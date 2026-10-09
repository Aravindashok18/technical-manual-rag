"""Does the system decline when the manuals do not contain the answer?

40 questions the manuals cannot answer: 10 off-topic, 30 about equipment, makes and procedures that
are not in these four manuals (eval/unanswerable.jsonl). Half of each kind goes to dev, half to test.

Three setups, each answering all 40:
  baseline        minimal prompt, baseline retrieval, no gate      (what a first version would do)
  prompt_only     chosen prompt (it tells the model to decline), no gate
  prompt_and_gate chosen prompt + similarity gate: decline when the best chunk scores below a threshold

The threshold is the highest value that still lets 95% of the DEV answerable questions through.
An answer counts as unsupported when it is not a decline.

    python -m eval.run_guardrail
"""
import json
import random
import re

import numpy as np

from eval.common import INDEX_ROOT, REPORTS, EVAL, load_questions, load_split
from eval.run_prompts import K, pipeline_for
from rag.llm import OllamaClient

BROAD_DECLINE = re.compile(
    r"could not find|couldn't find|can(?:no|')t find|not found|does(?: not|n't) (?:contain|mention|provide|specify|"
    r"include|say|cover|address|have|give)|do(?: not|n't) (?:contain|mention|provide|specify|include|have)|"
    r"no (?:specific )?information|not (?:mentioned|specified|provided|stated|available|addressed|covered)|"
    r"(?:cannot|can't|unable to) (?:answer|determine|find|provide)|not in the (?:context|manuals?|provided)|"
    r"does not appear", re.I)


def is_declined(text):
    return bool(BROAD_DECLINE.search(text))


def load_unanswerable():
    qs = [json.loads(l) for l in open(EVAL / "unanswerable.jsonl") if l.strip()]
    rng = random.Random(5)
    dev, test = set(), set()
    for kind in ("off_topic", "near_topic"):
        ids = [q["id"] for q in qs if q["kind"] == kind]
        rng.shuffle(ids)
        half = len(ids) // 2
        dev |= set(ids[:half])
        test |= set(ids[half:])
    return qs, dev, test


def top_score(pipe, question):
    hits, _ = pipe.retrieve(question, 1)
    return hits[0][1]


def main():
    gen = json.loads((REPORTS / "generation.json").read_text())
    chosen_cfg, base_cfg, prompt = gen["retrieval_config"], gen["baseline_config"], gen["chosen_prompt"]
    llm = OllamaClient()
    answerable = load_questions()
    dev_a, test_a = load_split(answerable)
    un, dev_u, test_u = load_unanswerable()

    pipe_final = pipeline_for(chosen_cfg, prompt, llm)
    pipe_base = pipeline_for(base_cfg, "minimal", llm)

    # --- threshold from dev scores only
    sc_ans = {q["id"]: top_score(pipe_final, q["question"]) for q in answerable}
    sc_un = {q["id"]: top_score(pipe_final, q["question"]) for q in un}
    thr = float(np.percentile([sc_ans[i] for i in dev_a], 5))      # 95% of dev answerable score >= thr
    print(f"threshold {thr:.4f}")

    def gate_stats(ids_a, ids_u):
        return {"answerable_declined": float(np.mean([sc_ans[i] < thr for i in ids_a])),
                "unanswerable_declined": float(np.mean([sc_un[i] < thr for i in ids_u]))}

    out = {"threshold": thr, "prompt": prompt, "retrieval_config": chosen_cfg,
           "score_summary": {"answerable_median": float(np.median(list(sc_ans.values()))),
                             "unanswerable_off_topic_median": float(np.median([sc_un[q["id"]] for q in un if q["kind"] == "off_topic"])),
                             "unanswerable_near_topic_median": float(np.median([sc_un[q["id"]] for q in un if q["kind"] == "near_topic"]))},
           "gate_only": {"dev": gate_stats(dev_a, dev_u), "test": gate_stats(test_a, test_u)}, "setups": {}}

    rows = {"baseline": [], "prompt_only": [], "prompt_and_gate": []}
    for q in un:
        a_base = pipe_base.ask(q["question"])
        a_final = pipe_final.ask(q["question"])
        gated = sc_un[q["id"]] < thr
        rec = {"id": q["id"], "kind": q["kind"], "split": "dev" if q["id"] in dev_u else "test",
               "question": q["question"], "score": sc_un[q["id"]]}
        rows["baseline"].append({**rec, "answer": a_base.answer, "unsupported": not is_declined(a_base.answer)})
        rows["prompt_only"].append({**rec, "answer": a_final.answer, "unsupported": not is_declined(a_final.answer)})
        rows["prompt_and_gate"].append({**rec, "answer": "(gate) " + a_final.answer if gated else a_final.answer,
                                        "unsupported": (not gated) and not is_declined(a_final.answer)})
        print(q["id"], q["kind"], f"score={sc_un[q['id']]:.3f}", "gated" if gated else "      ",
              "| base:", "ANSWERED" if rows["baseline"][-1]["unsupported"] else "declined",
              "| final:", "ANSWERED" if rows["prompt_only"][-1]["unsupported"] else "declined", flush=True)

    def rate(rs, split=None, kind=None):
        rs = [r for r in rs if (split is None or r["split"] == split) and (kind is None or r["kind"] == kind)]
        return {"n": len(rs), "unsupported_answer_rate": float(np.mean([r["unsupported"] for r in rs]))}

    for name, rs in rows.items():
        out["setups"][name] = {"all": rate(rs), "dev": rate(rs, "dev"), "test": rate(rs, "test"),
                               "off_topic": rate(rs, kind="off_topic"), "near_topic": rate(rs, kind="near_topic")}
    (REPORTS / "guardrail.json").write_text(json.dumps(out, indent=2))
    (REPORTS / "guardrail_answers.json").write_text(json.dumps(rows, indent=1))
    for name, r in out["setups"].items():
        print(f"{name:16s} all {r['all']['unsupported_answer_rate']:.3f}  test {r['test']['unsupported_answer_rate']:.3f}  "
              f"off-topic {r['off_topic']['unsupported_answer_rate']:.3f}  near-topic {r['near_topic']['unsupported_answer_rate']:.3f}")
    print("gate on answerable dev/test declined:", out["gate_only"]["dev"]["answerable_declined"], out["gate_only"]["test"]["answerable_declined"])


if __name__ == "__main__":
    main()
