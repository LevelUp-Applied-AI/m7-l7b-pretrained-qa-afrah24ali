"""
Module 7 Week B — Tuesday Stretch (Honors): Adversarial QA Probe.

Reuses the QA pipeline + EM/F1 functions from `lab.py`. Implement the TODO
functions below; see the stretch page for full task description.
"""

import json
import os
import sys

import pandas as pd


# Import the lab's existing functions (we reuse build_qa_pipeline, predict_one,
# evaluate_qa, normalize_answer, exact_match, token_f1)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import lab  # noqa: E402


def load_adversarial_set(path: str = "stretch/tuesday/adversarial_set.csv") -> pd.DataFrame:
    """
    Load the adversarial test set CSV.

    Verifies columns: qid, question, context, gold_answer, pattern_tag.
    """
    df= pd.read_csv(path)
    required_cols={"qid", "question", "context", "gold_answer", "pattern_tag"}
    missing= required_cols-set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns in adversarial set {path}: {missing}"
                         f"EXPECTED exactly: {required_cols}")
    
    return df


# -----------------------------
# Evaluation logic
# -----------------------------
def evaluate_adversarial(qa, df: pd.DataFrame) -> dict:
    """
    Run the QA pipeline on the adversarial set; compute aggregate + per-pattern metrics.

    Returns:
        {
          "em": float, "f1": float, "n": int,
          "per_pattern": { tag: {"em": float, "f1": float, "n": int}, ... },
          "predictions": [ ... lab.evaluate_qa-shaped entries plus pattern_tag ... ],
        }
    """
    
    result = lab.evaluate_qa(qa, df)
    predictions = result["predictions"]

    for prediction, (_, row) in zip(predictions, df.iterrows()):
        prediction["pattern_tag"] = row["pattern_tag"]

    per_pattern = {}
    for p in predictions:
        tag = p["pattern_tag"]
        if tag not in per_pattern:
            per_pattern[tag] = {"em": 0.0, "f1": 0.0, "n": 0}
        per_pattern[tag]["em"] += p["em"]
        per_pattern[tag]["f1"] += p["f1"]
        per_pattern[tag]["n"] += 1

    for scores in per_pattern.values():
        scores["em"] /= scores["n"]
        scores["f1"] /= scores["n"]

    return {
        "em": result["em"],
        "f1": result["f1"],
        "n": result["n"],
        "per_pattern": per_pattern,
        "predictions": predictions
    }



    

def main() -> None:
    """Load adversarial set, run evaluation, write predictions + metrics."""
    df = load_adversarial_set()

    qa = lab.build_qa_pipeline(lab.get_qa_model_name())

    result = evaluate_adversarial(qa, df)

    pred_df = pd.DataFrame(result["predictions"])
    pred_df.to_csv("stretch/tuesday/adversarial_predictions.csv", index=False)

    # save metrics
    metrics = {
        "em": result["em"],
        "f1": result["f1"],
        "n": result["n"],
        "per_pattern": result["per_pattern"],
        "model": lab.get_qa_model_name(),
    }

    with open("stretch/tuesday/adversarial_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Aggregate EM = {result['em']:.4f}")
    print(f"Aggregate F1 = {result['f1']:.4f}")
    print(f"n = {result['n']}")
    print(f"Per-pattern: {list(result['per_pattern'].keys())}")


if __name__ == "__main__":
    main()
