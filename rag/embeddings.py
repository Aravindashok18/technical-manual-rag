"""Sentence-embedding wrappers. Vectors are L2-normalised, so inner product = cosine similarity."""
import numpy as np

MODELS = {
    "minilm": {"name": "sentence-transformers/all-MiniLM-L6-v2", "query_prefix": ""},
    "bge-small": {"name": "BAAI/bge-small-en-v1.5",
                  "query_prefix": "Represent this sentence for searching relevant passages: "},
}


class Embedder:
    def __init__(self, key="bge-small", device="cpu"):
        from sentence_transformers import SentenceTransformer
        self.key = key
        spec = MODELS[key]
        self.model = SentenceTransformer(spec["name"], device=device)
        self.query_prefix = spec["query_prefix"]
        self.dim = self.model.get_sentence_embedding_dimension()

    def encode_docs(self, texts, batch_size=64):
        return self.model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                                 convert_to_numpy=True, show_progress_bar=False).astype(np.float32)

    def encode_query(self, query):
        v = self.model.encode([self.query_prefix + query], normalize_embeddings=True,
                              convert_to_numpy=True, show_progress_bar=False)
        return v.astype(np.float32)
