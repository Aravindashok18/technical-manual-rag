"""Question set, dev/test split, gold pages and answer scoring shared by the eval scripts."""
import json
import random
import re
from pathlib import Path

from rag.ingest import load_corpus

EVAL = Path(__file__).resolve().parent
ROOT = EVAL.parent
REPORTS = ROOT / "reports"
INDEX_ROOT = ROOT / "data" / "index"


def norm(s):
    """Lower-case and drop whitespace and commas so '1,500 V' and '1500v' compare equal."""
    return re.sub(r"[\s,]+", "", s.lower())


def load_questions(name="questions.jsonl"):
    return [json.loads(line) for line in open(EVAL / name) if line.strip()]


def load_split(questions):
    """60 development questions (used to choose settings) and 60 test questions, fixed by seed."""
    path = EVAL / "split.json"
    if not path.exists():
        ids = [q["id"] for q in questions]
        random.Random(11).shuffle(ids)
        path.write_text(json.dumps({"dev": sorted(ids[:60]), "test": sorted(ids[60:])}, indent=1))
    s = json.loads(path.read_text())
    return set(s["dev"]), set(s["test"])


def _is_toc(text):
    return text.count("....") >= 3


_corpus = None


def acceptable_pages(q):
    """Pages that count as a correct retrieval: the authored page plus any other page that holds
    the question's anchor phrase and every answer key (table-of-contents pages excluded)."""
    global _corpus
    if _corpus is None:
        _corpus = load_corpus()
    ok = {tuple(p) for p in q["pages"]}
    anchors = [norm(a) for a in q.get("anchor", [])]
    if not anchors:
        return ok
    for p in _corpus:
        if _is_toc(p.text):
            continue
        t = norm(p.text)
        if all(a in t for a in anchors) and all(any(norm(k) in t for k in grp) for grp in q["keys"]):
            ok.add((p.manual, p.page))
    return ok


def answer_correct(answer, keys):
    """Every key group must have at least one alternative present in the answer."""
    a = norm(answer)
    return all(any(norm(k) in a for k in group) for group in keys)
