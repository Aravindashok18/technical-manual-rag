"""Read the manual PDFs page by page."""
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pymupdf

MANUAL_DIR = Path(__file__).resolve().parent.parent / "data" / "manuals"

TITLES = {
    "FIST 3-16": "Maintenance of Power Circuit Breakers",
    "FIST 3-8": "Operation, Maintenance, and Field Test Procedures for Protective Relays and Associated Circuits",
    "FIST 4-1A": "Maintenance Scheduling for Mechanical Equipment",
    "FIST 4-1B": "Maintenance Scheduling for Electrical Equipment",
}


@dataclass
class Page:
    manual: str          # e.g. "FIST 3-16"
    page: int            # 1-based PDF page number, the one a viewer shows
    text: str
    has_figure: bool     # page carries an embedded image or a lot of vector drawing


def manual_name(path):
    m = re.match(r"FIST_(\d+-\d+[A-Z]?)", Path(path).name)
    return f"FIST {m.group(1)}" if m else Path(path).stem


def clean(text):
    """Undo hyphenated line breaks, collapse runs of spaces and drop blank lines."""
    text = text.replace("­", "").replace(" ", " ")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in text.split("\n")]
    lines = [ln for ln in lines if ln]
    return "\n".join(lines)


def load_manual(path):
    doc = pymupdf.open(path)
    name = manual_name(path)
    pages = []
    for i, pg in enumerate(doc, start=1):
        fig = bool(pg.get_images()) or len(pg.get_drawings()) > 50
        pages.append(Page(name, i, clean(pg.get_text()), fig))
    return pages


def strip_boilerplate(pages, min_share=0.3):
    """Drop lines that repeat on at least `min_share` of a manual's pages (running headers)."""
    by_manual = {}
    for p in pages:
        by_manual.setdefault(p.manual, []).append(p)
    for manual_pages in by_manual.values():
        counts = Counter(ln for p in manual_pages for ln in set(p.text.split("\n")))
        cutoff = min_share * len(manual_pages)
        common = {ln for ln, c in counts.items() if c >= cutoff and len(ln) > 3}
        for p in manual_pages:
            p.text = "\n".join(ln for ln in p.text.split("\n") if ln not in common)
    return pages


def load_corpus(directory=MANUAL_DIR, strip=False):
    pages = []
    for p in sorted(Path(directory).glob("*.pdf")):
        if "6-4" in p.name:      # acronym list, not an equipment manual
            continue
        pages += load_manual(p)
    return strip_boilerplate(pages) if strip else pages
