"""Paired comparison of setups using the cached answers: how many questions flip, and a
bootstrap interval (over questions) for the accuracy difference.

    python -m eval.analyze
"""
import json

import numpy as np

from eval.common import REPORTS, load_questions, load_split


def load_rows():
    rows = {}
    for line in open(REPORTS / "gen_cache.jsonl"):
        r = json.loads(line)
        rows[r["key"]] = r
    return rows


def correct_vector(rows, config, prompt, ids):
    return np.array([rows[f"{config}|{prompt}|k4|{i}"]["correct"] for i in ids], dtype=float)


def paired(a, b, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), (n, len(a)))
    diff = (a[idx] - b[idx]).mean(axis=1)
    return {"diff": float(a.mean() - b.mean()), "ci95": [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))],
            "a_only_correct": int(((a == 1) & (b == 0)).sum()), "b_only_correct": int(((a == 0) & (b == 1)).sum())}


def main():
    gen = json.loads((REPORTS / "generation.json").read_text())
    rows = load_rows()
    qs = load_questions()
    dev, test = load_split(qs)
    chosen, base, prompt = gen["retrieval_config"], gen["baseline_config"], gen["chosen_prompt"]
    arms = {"baseline": (base, "minimal"), "chosen_retrieval_minimal_prompt": (chosen, "minimal"), "final": (chosen, prompt)}
    out = {}
    for split_name, ids in (("test", sorted(test)), ("all", sorted(dev | test))):
        vec = {k: correct_vector(rows, c, p, ids) for k, (c, p) in arms.items()}
        out[split_name] = {"accuracy": {k: float(v.mean()) for k, v in vec.items()},
                           "final_vs_baseline": paired(vec["final"], vec["baseline"]),
                           "chosen_retrieval_minimal_vs_baseline": paired(vec["chosen_retrieval_minimal_prompt"], vec["baseline"]),
                           "final_vs_chosen_retrieval_minimal": paired(vec["final"], vec["chosen_retrieval_minimal_prompt"])}
    (REPORTS / "generation_paired.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
