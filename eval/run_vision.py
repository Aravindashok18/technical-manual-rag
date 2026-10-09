"""Diagram questions: text-only RAG against RAG with a LLaVA reading of the page image.

13 questions about figures in the relay and breaker manuals (eval/diagrams.jsonl). Each is answered
end to end: retrieve, pick the best retrieved page that holds a figure, ask LLaVA about the page
image, hand its reading to the text model along with the retrieved passages.

    python -m eval.run_vision
"""
import json

import numpy as np

from eval.common import EVAL, REPORTS, answer_correct
from eval.run_prompts import pipeline_for
from rag.llm import OllamaClient, VISION_MODEL
from rag.vision import DiagramReader


class CachedVision:
    """Replays LLaVA readings computed in a first pass, so the two models are not swapped per question."""
    def __init__(self, notes):
        self.notes = notes
        self.current = None

    def read_diagram(self, question, chunks):
        return self.notes.get(question, "")


def main():
    gen = json.loads((REPORTS / "generation.json").read_text())
    qs = [json.loads(l) for l in open(EVAL / "diagrams.jsonl") if l.strip()]
    text_llm = OllamaClient()
    pipe = pipeline_for(gen["retrieval_config"], gen["chosen_prompt"], text_llm)

    # pass 1: retrieval + text-only answers (text model loaded)
    plain, targets = [], {}
    for q in qs:
        hits, _ = pipe.retrieve(q["question"])
        gold = {tuple(p) for p in q["pages"]}
        fig = [c for c, _ in hits if c.has_figure][:1]
        targets[q["question"]] = fig
        a = pipe.ask(q["question"])
        plain.append({"id": q["id"], "answer": a.answer, "correct": (not a.declined) and answer_correct(a.answer, q["keys"]),
                      "gold_page_retrieved": any((c.manual, c.page) in gold for c, _ in hits),
                      "figure_page_is_gold": bool(fig) and (fig[0].manual, fig[0].page) in gold})
        print(q["id"], "text-only", "OK " if plain[-1]["correct"] else "bad", flush=True)

    # pass 2: LLaVA readings (vision model loaded)
    reader = DiagramReader(OllamaClient(VISION_MODEL))
    notes = {}
    for q in qs:
        notes[q["question"]] = reader.read_diagram(q["question"], targets[q["question"]])
        print(q["id"], "llava:", notes[q["question"]][:120].replace("\n", " "), flush=True)

    # pass 3: text model again, with the vision notes
    pipe.vision = CachedVision(notes)
    withv = []
    for q in qs:
        a = pipe.ask(q["question"])
        withv.append({"id": q["id"], "answer": a.answer, "used_vision": a.used_vision,
                      "correct": (not a.declined) and answer_correct(a.answer, q["keys"])})
        print(q["id"], "with vision", "OK " if withv[-1]["correct"] else "bad", flush=True)

    out = {"n": len(qs), "text_model": text_llm.model, "vision_model": VISION_MODEL,
           "text_only_accuracy": float(np.mean([r["correct"] for r in plain])),
           "with_vision_accuracy": float(np.mean([r["correct"] for r in withv])),
           "gold_page_retrieved": float(np.mean([r["gold_page_retrieved"] for r in plain])),
           "figure_page_is_gold": float(np.mean([r["figure_page_is_gold"] for r in plain])),
           "text_only": plain, "with_vision": withv, "llava_notes": notes}
    (REPORTS / "vision.json").write_text(json.dumps(out, indent=2))
    print({k: v for k, v in out.items() if k.endswith("accuracy") or k.endswith("gold") or k.endswith("retrieved")})


if __name__ == "__main__":
    main()
