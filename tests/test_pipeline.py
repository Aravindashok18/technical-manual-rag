from rag.llm import FakeLLM
from rag.pipeline import RagPipeline, is_decline, parse_citations
from rag.prompts import DECLINE, TEMPLATES, build_messages


def test_parse_citations_handles_common_formats():
    text = "Every 5 years [FIST 4-1B, p.158]. Also (FIST 3-16, page 53) and FIST 4-1B p. 158."
    assert parse_citations(text) == [("FIST 4-1B", 158), ("FIST 3-16", 53)]


def test_is_decline():
    assert is_decline(DECLINE) and is_decline("Sorry, I couldn't find that.")
    assert not is_decline("Every 5 years.")


def test_all_templates_build_messages_with_sources(index, embedder):
    hits = index.search(embedder.encode_query("Buchholz relay"), 2)
    for name in TEMPLATES:
        msgs = build_messages(name, "How often?", hits)
        assert msgs[0]["role"] == "system" and "How often?" in msgs[-1]["content"]
        assert "[FIST 4-1B, p.158]" in msgs[-1]["content"]


def test_ask_returns_sources_with_pages_and_cited_pages(index, embedder):
    llm = FakeLLM("Test it every 5 years. Source: [FIST 4-1B, p.158]")
    pipe = RagPipeline(index, embedder, llm, k=2)
    ans = pipe.ask("How often should a Buchholz relay on the transformer be tested?")
    assert not ans.declined
    assert (ans.sources[0]["manual"], ans.sources[0]["page"]) == ("FIST 4-1B", 158)
    assert ans.cited == [("FIST 4-1B", 158)]
    assert {"retrieve_ms", "generate_ms", "total_ms"} <= set(ans.timings)


def test_threshold_declines_without_calling_the_llm(index, embedder):
    llm = FakeLLM()
    pipe = RagPipeline(index, embedder, llm, k=2, threshold=0.99)
    ans = pipe.ask("What is the capital of Australia?")
    assert ans.declined and ans.answer == DECLINE
    assert llm.calls == []
    assert "generate_ms" not in ans.timings


def test_model_decline_is_recognised(index, embedder):
    pipe = RagPipeline(index, embedder, FakeLLM(DECLINE), k=2)
    assert pipe.ask("Buchholz relay test interval?").declined


def test_vision_note_is_added_for_diagram_questions_on_figure_pages(index, embedder):
    class FakeVision:
        def __init__(self): self.seen = None
        def read_diagram(self, question, chunks):
            self.seen = [(c.manual, c.page) for c in chunks]
            return "A jumper joins H1 and X1."
    vision = FakeVision()
    llm = FakeLLM("H1 and X1")
    pipe = RagPipeline(index, embedder, llm, k=4, vision=vision)
    ans = pipe.ask("In the wiring diagram, which current transformer excitation terminals are connected?")
    assert ans.used_vision and vision.seen == [("FIST 3-8", 29)]
    assert "A jumper joins H1 and X1." in llm.calls[0]["messages"][-1]["content"]


def test_vision_not_used_for_plain_questions(index, embedder):
    class Boom:
        def read_diagram(self, *a): raise AssertionError("should not be called")
    pipe = RagPipeline(index, embedder, FakeLLM(), k=3, vision=Boom())
    assert not pipe.ask("How often are current transformer excitation tests done?").used_vision
