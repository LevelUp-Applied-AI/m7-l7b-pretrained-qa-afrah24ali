"""
Module 7 Week B — Tuesday Stretch (Honors): Adversarial QA Probe.

Reuses the QA pipeline + EM/F1 functions from `lab.py`. Implement the TODO
functions below; see the stretch page for full task description.
"""

import json
import os
import sys

import pandas as pd

# allow import of lab.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import lab  # noqa: E402

def load_adversarial_set(path: str = "stretch/tuesday/adversarial_set.csv") -> pd.DataFrame:
    df = pd.read_csv(path)

    required_cols = {"qid", "question", "context", "gold_answer", "pattern_tag"}
    missing = required_cols - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    return df


# -----------------------------
# Evaluation logic
# -----------------------------
def evaluate_adversarial(qa, df: pd.DataFrame) -> dict:
    predictions = []

    total_em = 0.0
    total_f1 = 0.0
    n = len(df)

    per_pattern = {}

    for _, row in df.iterrows():
        qid = row["qid"]
        question = row["question"]
        context = row["context"]
        gold = row["gold_answer"]
        tag = row["pattern_tag"]

        # fix missing tags
        if pd.isna(tag):
            tag = "control"

        # -------------------------
        # FIXED QA CALL (IMPORTANT)
        # -------------------------
        pred = qa(question=question, context=context)["answer"]

        # metrics
        em = lab.exact_match(pred, gold)
        f1 = lab.token_f1(pred, gold)

        total_em += em
        total_f1 += f1

        predictions.append({
            "qid": qid,
            "question": question,
            "context_excerpt": context[:80],
            "gold_answer": gold,
            "predicted_answer": pred,
            "pattern_tag": tag,
            "em": em,
            "f1": f1
        })

        # init bucket
        if tag not in per_pattern:
            per_pattern[tag] = {"em": 0.0, "f1": 0.0, "n": 0}

        per_pattern[tag]["em"] += em
        per_pattern[tag]["f1"] += f1
        per_pattern[tag]["n"] += 1

    # normalize per pattern
    for tag in per_pattern:
        per_pattern[tag]["em"] /= per_pattern[tag]["n"]
        per_pattern[tag]["f1"] /= per_pattern[tag]["n"]

    return {
        "em": total_em / n,
        "f1": total_f1 / n,
        "n": n,
        "per_pattern": per_pattern,
        "predictions": predictions
    }


def main() -> None:
    """Load adversarial set, run evaluation, write predictions + metrics."""
    df = load_adversarial_set()

    qa = lab.build_qa_pipeline(lab.get_qa_model_name())

    result = evaluate_adversarial(qa, df)

    # save predictions
    pd.DataFrame(result["predictions"]).to_csv(
        "stretch/tuesday/adversarial_predictions.csv",
        index=False
    )

    # save metrics
    metrics = {
        "em": result["em"],
        "f1": result["f1"],
        "n": result["n"],
        "per_pattern": result["per_pattern"],
        "model": lab.get_qa_model_name()
    }

    with open("stretch/tuesday/adversarial_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Aggregate EM = {result['em']:.4f}")
    print(f"Aggregate F1 = {result['f1']:.4f}")
    print(f"n = {result['n']}")
    print(f"Patterns = {list(result['per_pattern'].keys())}")


if __name__ == "__main__":
    main()
