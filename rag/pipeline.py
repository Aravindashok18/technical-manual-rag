"""Question in -> retrieve -> guardrail -> prompt -> answer with cited pages."""
import re
import time
from dataclasses import dataclass, field

from . import prompts

CITE_RE = re.compile(r"\[?\s*(FIST\s+\d+-\d+[AB]?)\s*,?\s*(?:p\.?|page)\s*(\d+)", re.I)
DIAGRAM_WORDS = re.compile(r"\b(diagram|figure|wiring|schematic|circuit drawing|connection|connect(ed)?|"
                           r"drawn|shown|illustrat\w+)\b", re.I)


@dataclass
class Answer:
    question: str
    answer: str
    declined: bool
    reason: str = ""
    sources: list = field(default_factory=list)     # [{manual, page, score, snippet}]
    cited: list = field(default_factory=list)       # [(manual, page)] found in the model's text
    timings: dict = field(default_factory=dict)     # stage -> ms
    used_vision: bool = False


def parse_citations(text):
    seen, out = set(), []
    for manual, page in CITE_RE.findall(text):
        key = (re.sub(r"\s+", " ", manual.upper()), int(page))
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out


def is_decline(text):
    t = text.lower()
    return "could not find" in t or "couldn't find" in t or "not found in the manual" in t


class RagPipeline:
    def __init__(self, index, embedder, llm, prompt="concise_structured", k=4, threshold=None,
                 vision=None):
        """threshold: minimum cosine score of the best chunk; below it the question is declined
        without calling the LLM. None switches the gate off."""
        self.index, self.embedder, self.llm = index, embedder, llm
        self.prompt, self.k, self.threshold, self.vision = prompt, k, threshold, vision

    def retrieve(self, question, k=None):
        t0 = time.perf_counter()
        hits = self.index.search(self.embedder.encode_query(question), k or self.k)
        return hits, (time.perf_counter() - t0) * 1000

    def ask(self, question, k=None):
        t_start = time.perf_counter()
        hits, retrieve_ms = self.retrieve(question, k)
        sources = [{"manual": c.manual, "page": c.page, "score": round(s, 4), "snippet": c.text[:240]}
                   for c, s in hits]
        timings = {"retrieve_ms": retrieve_ms}

        if not hits or (self.threshold is not None and hits[0][1] < self.threshold):
            timings["total_ms"] = (time.perf_counter() - t_start) * 1000
            return Answer(question, prompts.DECLINE, True, "best passage below similarity threshold",
                          sources, [], timings)

        used_vision = False
        messages = prompts.build_messages(self.prompt, question, hits)
        if self.vision and DIAGRAM_WORDS.search(question):
            t_v = time.perf_counter()
            note = self.vision.read_diagram(question, [c for c, _ in hits if c.has_figure][:1])
            timings["vision_ms"] = (time.perf_counter() - t_v) * 1000
            if note:
                used_vision = True
                messages[-1]["content"] += f"\n\nDiagram reading from the page image (vision model, may be imperfect):\n{note}"

        t_g = time.perf_counter()
        text, stats = self.llm.chat(messages)
        timings["generate_ms"] = (time.perf_counter() - t_g) * 1000
        timings["output_tokens"] = stats.get("output_tokens")
        timings["total_ms"] = (time.perf_counter() - t_start) * 1000
        declined = is_decline(text)
        return Answer(question, text, declined, "model declined" if declined else "", sources,
                      parse_citations(text), timings, used_vision)
