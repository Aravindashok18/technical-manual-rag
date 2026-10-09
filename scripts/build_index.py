"""Build the index the API serves (data/index/default).

    python scripts/build_index.py [chunking] [embedder]

Defaults to the configuration chosen on the development questions in reports/retrieval.json.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.chunking import chunk_pages  # noqa: E402
from rag.embeddings import Embedder  # noqa: E402
from rag.index import VectorIndex  # noqa: E402
from rag.ingest import load_corpus  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main():
    if len(sys.argv) >= 3:
        strategy, emb_key = sys.argv[1], sys.argv[2]
    else:
        chosen = json.loads((ROOT / "reports" / "retrieval.json").read_text())["chosen"]
        strategy, emb_key = chosen.split("__")
    pages = load_corpus(strip=True)
    chunks = chunk_pages(pages, strategy)
    index = VectorIndex.build(chunks, Embedder(emb_key), strategy)
    index.save(ROOT / "data" / "index" / "default")
    print(f"indexed {len(pages)} pages into {len(chunks)} chunks ({strategy}, {emb_key})")


if __name__ == "__main__":
    main()
