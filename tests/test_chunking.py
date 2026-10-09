from rag.chunking import chunk_pages, fixed_windows, recursive_lines
from rag.ingest import Page, clean, strip_boilerplate


def test_clean_joins_hyphenated_words_and_drops_blank_lines():
    assert clean("main-\ntenance  of \n\n  breakers") == "maintenance of\nbreakers"


def test_fixed_windows_cover_the_text_with_overlap():
    text = " ".join(f"word{i}" for i in range(300))
    chunks = fixed_windows(text, 200, 40)
    assert all(len(c) <= 200 for c in chunks)
    assert chunks[0].startswith("word0") and chunks[-1].endswith("word299")
    assert set(text.split()) <= set(" ".join(chunks).split())


def test_recursive_lines_respects_size_and_keeps_every_line():
    lines = [f"line number {i} with some words" for i in range(60)]
    chunks = recursive_lines("\n".join(lines), 300, 60)
    assert all(len(c) <= 300 for c in chunks)
    joined = "\n".join(chunks)
    assert all(l in joined for l in lines)


def test_recursive_lines_splits_a_single_huge_line():
    chunks = recursive_lines("word " * 500, 200, 20)
    assert len(chunks) > 5 and all(len(c) <= 200 for c in chunks)


def test_chunks_keep_their_page_and_skip_blank_pages():
    pages = [Page("FIST 3-16", 5, "x" * 100, False), Page("FIST 3-16", 6, "", False),
             Page("FIST 4-1A", 7, "y " * 400, True)]
    for strategy in ("fixed500", "recursive1000", "page"):
        chunks = chunk_pages(pages, strategy)
        assert {(c.manual, c.page) for c in chunks} == {("FIST 3-16", 5), ("FIST 4-1A", 7)}
        assert [c.id for c in chunks] == list(range(len(chunks)))


def test_page_strategy_is_one_chunk_per_page():
    pages = [Page("FIST 3-8", i, "some page text " * 20, False) for i in range(1, 5)]
    assert len(chunk_pages(pages, "page")) == 4


def test_strip_boilerplate_removes_running_headers_only():
    pages = [Page("M", i, f"FIST Volume 3-16\nunique content {i}", False) for i in range(10)]
    strip_boilerplate(pages)
    assert all("FIST Volume" not in p.text and "unique content" in p.text for p in pages)
