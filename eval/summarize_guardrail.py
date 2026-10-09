"""Unsupported-answer rates using the hand labels in eval/guardrail_labels.json for the baseline
and exact-match on the fixed decline sentence for the two setups that were told to use it.

    python -m eval.summarize_guardrail
"""
import json

from eval.common import EVAL, REPORTS
from rag.prompts import DECLINE


def rate(rows, flag):
    return {"n": len(rows), "unsupported": sum(flag(r) for r in rows),
            "rate": sum(flag(r) for r in rows) / len(rows)}


def main():
    answers = json.loads((REPORTS / "guardrail_answers.json").read_text())
    labels = json.loads((EVAL / "guardrail_labels.json").read_text())
    bad = set(labels["baseline_unsupported"])
    flags = {"baseline": lambda r: r["id"] in bad,
             "prompt_only": lambda r: r["answer"].strip() != DECLINE,
             "prompt_and_gate": lambda r: not r["answer"].startswith("(gate)") and r["answer"].strip() != DECLINE}
    out = {}
    for name, rows in answers.items():
        out[name] = {"all": rate(rows, flags[name]),
                     "off_topic": rate([r for r in rows if r["kind"] == "off_topic"], flags[name]),
                     "near_topic": rate([r for r in rows if r["kind"] == "near_topic"], flags[name])}
    (REPORTS / "guardrail_summary.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
