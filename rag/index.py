"""FAISS index over chunk embeddings, with the chunk records stored beside it."""
import json
from pathlib import Path

import faiss
import numpy as np

from .chunking import Chunk


class VectorIndex:
    def __init__(self, index, chunks, embedder_key, strategy):
        self.index, self.chunks = index, chunks
        self.embedder_key, self.strategy = embedder_key, strategy

    @classmethod
    def build(cls, chunks, embedder, strategy):
        vecs = embedder.encode_docs([c.embed_text for c in chunks])
        index = faiss.IndexFlatIP(vecs.shape[1])
        index.add(vecs)
        return cls(index, chunks, embedder.key, strategy)

    def search(self, query_vec, k=4):
        """Return [(chunk, cosine score)] best first."""
        scores, ids = self.index.search(query_vec, min(k, len(self.chunks)))
        return [(self.chunks[i], float(s)) for i, s in zip(ids[0], scores[0]) if i >= 0]

    def save(self, directory):
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(d / "faiss.index"))
        with open(d / "chunks.jsonl", "w") as f:
            for c in self.chunks:
                f.write(json.dumps(c.to_dict()) + "\n")
        (d / "meta.json").write_text(json.dumps({"embedder": self.embedder_key, "strategy": self.strategy,
                                                 "chunks": len(self.chunks)}))

    @classmethod
    def load(cls, directory):
        d = Path(directory)
        meta = json.loads((d / "meta.json").read_text())
        chunks = [Chunk(**json.loads(line)) for line in open(d / "chunks.jsonl")]
        return cls(faiss.read_index(str(d / "faiss.index")), chunks, meta["embedder"], meta["strategy"])
