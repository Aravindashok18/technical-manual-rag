"""Read a diagram on a manual page with a vision model (LLaVA through Ollama)."""
import pymupdf

from .ingest import MANUAL_DIR

PROMPT = ("This is a page from an equipment maintenance manual. It contains a figure, such as a wiring or "
          "connection diagram. Look at the figure and answer the question about it in two or three sentences. "
          "Say only what you can see. Question: {question}")


def render_page(manual, page, dpi=130):
    """PNG bytes of one manual page."""
    stem = manual.replace("FIST ", "FIST_")
    path = next(MANUAL_DIR.glob(f"{stem}_*.pdf"))
    doc = pymupdf.open(path)
    return doc[page - 1].get_pixmap(dpi=dpi).tobytes("png")


class DiagramReader:
    def __init__(self, vision_llm, dpi=130):
        self.llm, self.dpi = vision_llm, dpi

    def read_diagram(self, question, chunks):
        if not chunks:
            return ""
        c = chunks[0]
        png = render_page(c.manual, c.page, self.dpi)
        text, _ = self.llm.chat([{"role": "user", "content": PROMPT.format(question=question)}],
                                images=[png], max_tokens=200)
        return text
