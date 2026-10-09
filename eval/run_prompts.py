"""Answer accuracy with the local LLM.

Phase A  4 prompt templates on the 60 DEV questions, retrieval fixed to the config chosen on dev.
Phase B  on the 60 TEST questions (and, for the headline, all 120):
           baseline                         fixed500 chunks + MiniLM + minimal prompt
           chosen_retrieval_minimal_prompt  chosen chunking/embedder + minimal prompt
           final                            chosen chunking/embedder + chosen prompt
A question counts as answered correctly when the answer contains every key group
(eval/common.answer_correct). A decline is never correct. Answers are cached in
reports/gen_cache.jsonl so the run can be stopped and resumed.

    python -m eval.run_prompts
"""
import json
import sys
from pathlib import Path

import numpy as np

from eval.common import INDEX_ROOT, REPORTS, acceptable_pages, answer_correct, load_questions, load_split
from rag.embeddings import Embedder
from rag.index import VectorIndex
from rag.llm import OllamaClient
from rag.pipeline import RagPipeline
from rag.prompts import TEMPLATES

CACHE = REPORTS / "gen_cache.jsonl"
K = 4
_cache = {}
_embedders, _indexes = {}, {}


def load_cache():
    if CACHE.exists():
        for line in open(CACHE):
            r = json.loads(line)
            _cache[r["key"]] = r


def pipeline_for(config, prompt, llm):
    emb_key = config.split("__")[1]
    if emb_key not in _embedders:
        _embedders[emb_key] = Embedder(emb_key)
    if config not in _indexes:
        _indexes[config] = VectorIndex.load(INDEX_ROOT / config)
    return RagPipeline(_indexes[config], _embedders[emb_key], llm, prompt=prompt, k=K)


def run(config, prompt, questions, llm, label):
    pipe = pipeline_for(config, prompt, llm)
    rows = []
    for i, q in enumerate(questions, start=1):
        key = f"{config}|{prompt}|k{K}|{q['id']}"
        if key not in _cache:
            ans = pipe.ask(q["question"])
            gold = acceptable_pages(q)
            rec = {"key": key, "id": q["id"], "answer": ans.answer, "declined": ans.declined,
                   "correct": (not ans.declined) and answer_correct(ans.answer, q["keys"]),
                   "cited": [list(c) for c in ans.cited],
                   "cited_ok": any(tuple(c) in gold for c in ans.cited),
                   "source_ok": any((s["manual"], s["page"]) in gold for s in ans.sources),
                   "generate_ms": ans.timings.get("generate_ms"), "retrieve_ms": ans.timings["retrieve_ms"]}
            _cache[key] = rec
            with open(CACHE, "a") as f:
                f.write(json.dumps(rec) + "\n")
        rows.append(_cache[key])
        if i % 10 == 0:
            print(f"  {label}: {i}/{len(questions)}  acc so far {np.mean([r['correct'] for r in rows]):.3f}", flush=True)
    return rows


def summarise(rows):
    n = len(rows)
    answered = [r for r in rows if not r["declined"]]
    cited = [r for r in answered if r["cited"]]
    return {"n": n, "accuracy": float(np.mean([r["correct"] for r in rows])),
            "declined": float(np.mean([r["declined"] for r in rows])),
            "gold_page_in_sources": float(np.mean([r["source_ok"] for r in rows])),
            "answers_with_inline_citation": float(len(cited) / len(answered)) if answered else 0.0,
            "inline_citation_on_gold_page": float(np.mean([r["cited_ok"] for r in cited])) if cited else None,
            "median_generate_s": float(np.median([r["generate_ms"] for r in rows if r["generate_ms"]]) / 1000)}


def main():
    REPORTS.mkdir(exist_ok=True)
    load_cache()
    qs = load_questions()
    dev_ids, test_ids = load_split(qs)
    dev = [q for q in qs if q["id"] in dev_ids]
    test = [q for q in qs if q["id"] in test_ids]
    ret = json.loads((REPORTS / "retrieval.json").read_text())
    chosen_cfg, base_cfg = ret["chosen"], ret["baseline"]
    llm = OllamaClient()
    if not llm.available():
        sys.exit(f"Ollama model {llm.model} is not available at {llm.url}")
    out = {"model": llm.model, "k": K, "retrieval_config": chosen_cfg, "baseline_config": base_cfg}

    print("phase A: prompts on dev", flush=True)
    out["prompts_dev"] = {}
    for name in TEMPLATES:
        rows = run(chosen_cfg, name, dev, llm, f"dev/{name}")
        out["prompts_dev"][name] = summarise(rows)
        print(f"{name:20s} dev {out['prompts_dev'][name]}", flush=True)
    # Highest dev accuracy; ties go to the prompt whose answers cite a page more often, because the
    # assistant has to cite its sources. (The first run broke ties by retrieval hit rate, which is
    # identical for every prompt, so it fell through to "minimal"; see README.)
    best_prompt = max(out["prompts_dev"], key=lambda n: (round(out["prompts_dev"][n]["accuracy"], 4),
                                                         out["prompts_dev"][n]["answers_with_inline_citation"]))
    out["chosen_prompt"] = best_prompt
    print("chosen prompt (dev):", best_prompt, flush=True)

    setups = {"baseline": (base_cfg, "minimal"), "chosen_retrieval_minimal_prompt": (chosen_cfg, "minimal"),
              "final": (chosen_cfg, best_prompt)}
    out["setups"] = {k: {"retrieval": v[0], "prompt": v[1]} for k, v in setups.items()}
    out["test"], out["all"] = {}, {}
    for name, (cfg, prompt) in setups.items():
        rows_test = run(cfg, prompt, test, llm, f"test/{name}")
        out["test"][name] = summarise(rows_test)
        print(f"{name:15s} test {out['test'][name]}", flush=True)
    for name in setups:
        cfg, prompt = setups[name]
        rows_all = run(cfg, prompt, dev, llm, f"dev/{name}") + run(cfg, prompt, test, llm, f"test/{name}")
        out["all"][name] = summarise(rows_all)
        print(f"{name:15s} all120 {out['all'][name]}", flush=True)
    (REPORTS / "generation.json").write_text(json.dumps(out, indent=2))
    print("done")


if __name__ == "__main__":
    main()
