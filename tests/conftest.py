import hashlib
import re

import numpy as np
import pytest

from rag.chunking import Chunk
from rag.index import VectorIndex


class HashEmbedder:
    """Bag-of-words hashing embedder: deterministic, no model download."""
    key = "hash"
    dim = 128

    def _vec(self, text):
        v = np.zeros(self.dim, np.float32)
        for tok in re.findall(r"[a-z0-9]+", text.lower()):
            v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim] += 1
        n = np.linalg.norm(v)
        return v / n if n else v

    def encode_docs(self, texts, batch_size=64):
        return np.stack([self._vec(t) for t in texts])

    def encode_query(self, q):
        return self._vec(q)[None, :]


CHUNKS = [
    Chunk(0, "FIST 4-1B", 158, "Test Buchholz relay on the transformer every 5 years."),
    Chunk(1, "FIST 3-16", 53, "Vacuum circuit breaker contacts move no more than 10 to 20 mm."),
    Chunk(2, "FIST 3-8", 29, "Current transformer excitation tests are performed every 6 years.", has_figure=True),
    Chunk(3, "FIST 4-1A", 130, "Check the oil or grease cups of the compressor weekly."),
]


@pytest.fixture
def embedder():
    return HashEmbedder()


@pytest.fixture
def index(embedder):
    return VectorIndex.build(CHUNKS, embedder, "test")
