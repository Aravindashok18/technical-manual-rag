"""Compare 3 chunking strategies x 2 embedding models on retrieval.

For every question we look at the ranked chunks and find the first one that sits on a gold page.
    hit@k  = share of questions with a gold page among the top k chunks
    MRR    = mean of 1 / rank of the first gold chunk (0 if not in the top 10)

    python -m eval.run_retrieval
"""
import json

import numpy as np

from eval.common import INDEX_ROOT, REPORTS, acceptable_pages, load_questions, load_split
from rag.chunking import STRATEGIES, chunk_pages
from rag.embeddings import MODELS, Embedder
from rag.index import VectorIndex
from rag.ingest import load_corpus

KS = (1, 3, 5, 10)


def first_gold_rank(index, embedder, q, gold, depth=10):
    hits = index.search(embedder.encode_query(q["question"]), depth)
    for rank, (c, _) in enumerate(hits, start=1):
        if (c.manual, c.page) in gold:
            return rank
    return None


def summarise(ranks):
    n = len(ranks)
    out = {"n": n, "mrr": float(np.mean([1 / r if r else 0 for r in ranks]))}
    for k in KS:
        out[f"hit@{k}"] = float(np.mean([bool(r and r <= k) for r in ranks]))
    return out


def main():
    questions = load_questions()
    dev, test = load_split(questions)
    gold = {q["id"]: acceptable_pages(q) for q in questions}
    pages = load_corpus(strip=True)
    results = {"corpus": {"manuals": sorted({p.manual for p in pages}), "pages": len(pages)}, "configs": {}}
    for emb_key in MODELS:
        embedder = Embedder(emb_key)
        for strat in STRATEGIES:
            name = f"{strat}__{emb_key}"
            chunks = chunk_pages(pages, strat)
            index = VectorIndex.build(chunks, embedder, strat)
            index.save(INDEX_ROOT / name)
            ranks = {q["id"]: first_gold_rank(index, embedder, q, gold[q["id"]]) for q in questions}
            res = {"chunks": len(chunks), "mean_chunk_chars": float(np.mean([len(c.text) for c in chunks])),
                   "all": summarise(list(ranks.values())),
                   "dev": summarise([ranks[i] for i in sorted(dev)]),
                   "test": summarise([ranks[i] for i in sorted(test)])}
            results["configs"][name] = res
            print(f"{name:28s} chunks={len(chunks):5d}  dev hit@3={res['dev']['hit@3']:.3f} hit@5={res['dev']['hit@5']:.3f} "
                  f"mrr={res['dev']['mrr']:.3f} | test hit@3={res['test']['hit@3']:.3f} hit@5={res['test']['hit@5']:.3f} "
                  f"mrr={res['test']['mrr']:.3f}", flush=True)
    # the choice is made on the dev questions only: best hit@5, ties broken by MRR
    best = max(results["configs"], key=lambda n: (results["configs"][n]["dev"]["hit@5"], results["configs"][n]["dev"]["mrr"]))
    results["chosen"] = best
    results["baseline"] = "fixed500__minilm"
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "retrieval.json").write_text(json.dumps(results, indent=2))
    print("chosen on dev:", best)


if __name__ == "__main__":
    main()
