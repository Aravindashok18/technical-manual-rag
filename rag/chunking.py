"""Three ways to cut manual pages into chunks. A chunk never crosses a page boundary,
so every chunk can be cited with an exact page number."""
import re
from dataclasses import asdict, dataclass

from .ingest import TITLES


@dataclass
class Chunk:
    id: int
    manual: str
    page: int
    text: str
    has_figure: bool = False

    @property
    def embed_text(self):
        """Text that is embedded: the manual title gives the model context that page text lacks."""
        return f"{TITLES.get(self.manual, self.manual)}. {self.text}"

    def to_dict(self):
        return asdict(self)


def _split_long(line, size):
    """Break one over-long line at sentence ends, then at spaces."""
    parts = re.split(r"(?<=[.;:])\s+", line)
    out, cur = [], ""
    for part in parts:
        while len(part) > size:
            cut = part.rfind(" ", 0, size)
            cut = cut if cut > 0 else size
            if cur:
                out.append(cur)
                cur = ""
            out.append(part[:cut])
            part = part[cut:].lstrip()
        if cur and len(cur) + 1 + len(part) > size:
            out.append(cur)
            cur = part
        else:
            cur = f"{cur} {part}".strip()
    if cur:
        out.append(cur)
    return out


def fixed_windows(text, size, overlap):
    """Character windows of `size` with `overlap`, moved to word boundaries."""
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            cut = text.rfind(" ", start + size // 2, end)
            end = cut if cut > 0 else end
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
        nxt = text.find(" ", start)
        start = nxt + 1 if 0 <= nxt < end else start
    return [c for c in chunks if c]


def recursive_lines(text, size, overlap):
    """Pack whole lines into chunks up to `size`, carrying the last lines (<= `overlap`) forward."""
    lines = []
    for ln in text.split("\n"):
        lines += _split_long(ln, size) if len(ln) > size else [ln]
    chunks, cur = [], []
    for ln in lines:
        if cur and sum(len(x) + 1 for x in cur) + len(ln) > size:
            chunks.append("\n".join(cur))
            keep, total = [], 0
            for x in reversed(cur):
                if total + len(x) + 1 > overlap:
                    break
                keep.insert(0, x)
                total += len(x) + 1
            cur = keep
        cur.append(ln)
    if cur:
        chunks.append("\n".join(cur))
    return chunks


STRATEGIES = {
    "fixed500": lambda t: fixed_windows(t.replace("\n", " "), 500, 50),
    "recursive1000": lambda t: recursive_lines(t, 1000, 100),
    "page": lambda t: [t],
}


def chunk_pages(pages, strategy):
    split = STRATEGIES[strategy]
    chunks = []
    for p in pages:
        if len(p.text) < 40:          # blank or near-blank page
            continue
        for piece in split(p.text):
            chunks.append(Chunk(len(chunks), p.manual, p.page, piece, p.has_figure))
    return chunks
