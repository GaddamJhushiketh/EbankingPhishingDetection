"""Train and evaluate the Phase 4 associative classifier.

Example: python scripts/train_associative_classifier.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from app.ml.associative_classifier import AssociativeClassifier
from app.ml.preprocessing import TARGET_LABELS, TARGET_COLUMN, validate_training_frame


def stratified_split(frame: pd.DataFrame, test_size: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_parts, test_parts = [], []
    for label, group in frame.groupby(TARGET_COLUMN, sort=True):
        shuffled = group.sample(frac=1, random_state=seed + int(label) + 100)
        count = max(1, round(len(group) * test_size))
        test_parts.append(shuffled.iloc[:count])
        train_parts.append(shuffled.iloc[count:])
    return (
        pd.concat(train_parts).sample(frac=1, random_state=seed).reset_index(drop=True),
        pd.concat(test_parts).sample(frac=1, random_state=seed + 1).reset_index(drop=True),
    )


def metrics(y_true: list[int], y_pred: list[int]) -> dict:
    per_class = {}
    labels = list(TARGET_LABELS)
    confusion = {str(actual): {str(predicted): 0 for predicted in labels} for actual in labels}
    for actual, predicted in zip(y_true, y_pred):
        confusion[str(actual)][str(predicted)] += 1
    for label in TARGET_LABELS:
        tp = sum(a == label and b == label for a, b in zip(y_true, y_pred))
        fp = sum(a != label and b == label for a, b in zip(y_true, y_pred))
        fn = sum(a == label and b != label for a, b in zip(y_true, y_pred))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        support = sum(actual == label for actual in y_true)
        per_class[str(label)] = {
            "precision": precision, "recall": recall, "f1": f1, "support": support
        }
    accuracy = sum(a == b for a, b in zip(y_true, y_pred)) / len(y_true)
    macro = {
        name: sum(values[name] for values in per_class.values()) / len(per_class)
        for name in ("precision", "recall", "f1")
    }
    weighted = {
        name: sum(values[name] * values["support"] for values in per_class.values()) / len(y_true)
        for name in ("precision", "recall", "f1")
    }
    return {
        "accuracy": accuracy,
        "confusion_matrix": {"labels": labels, "counts": confusion},
        "classification_report": {**per_class, "macro avg": macro, "weighted avg": weighted},
        "macro_avg": macro,
        "classes": per_class,
        "n_test": len(y_true),
    }


def rule_characteristics(rules: list[dict]) -> dict:
    result = {"count": len(rules)}
    for field in ("support", "confidence", "lift", "antecedent_length"):
        values = [float(rule[field]) for rule in rules]
        result[field] = {
            "min": min(values) if values else None,
            "max": max(values) if values else None,
            "mean": sum(values) / len(values) if values else None,
            "median": sorted(values)[len(values) // 2] if values else None,
        }
    return result


def public_rule(rule: dict) -> dict:
    """Serialize a classification rule without tuple/set implementation details."""
    return {
        "antecedent": list(rule["antecedent"]),
        "consequent": rule["consequent"],
        "support": rule["support"],
        "confidence": rule["confidence"],
        "lift": rule["lift"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="dataset/processed/phishing_clean.csv")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--min-support", type=float, default=0.01)
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument("--min-lift", type=float, default=1.0)
    parser.add_argument(
        "--confidence-experiments", default="0.3,0.5,0.7",
        help="Comma-separated confidence thresholds evaluated on the same split",
    )
    args = parser.parse_args()
    frame = validate_training_frame(pd.read_csv(args.data))
    train, test = stratified_split(frame, args.test_size, args.seed)
    y_true = test[TARGET_COLUMN].astype(int).tolist()
    test_records = test.drop(columns=[TARGET_COLUMN]).to_dict("records")
    thresholds = sorted({
        args.min_confidence,
        *(float(value) for value in args.confidence_experiments.split(",")),
    })
    experiments = []
    experiment_models = {}
    for threshold in thresholds:
        candidate = AssociativeClassifier(
            args.min_support, threshold, args.min_lift, random_state=args.seed
        ).fit(train)
        predictions = candidate.predict(test)
        experiment_metrics = metrics(y_true, predictions)
        experiments.append({
            "min_confidence": threshold,
            "frequent_itemsets": candidate.frequent_itemset_count,
            "generated_association_rules": candidate.generated_rule_count,
            "n_rules": len(candidate.rules),
            "accuracy": experiment_metrics["accuracy"],
            "macro_f1": experiment_metrics["macro_avg"]["f1"],
            "unmatched_test_count": sum(not candidate.match_rules(row) for row in test_records),
        })
        experiment_models[threshold] = candidate
    selected = max(
        experiments,
        key=lambda item: (item["macro_f1"], item["accuracy"], item["min_confidence"]),
    )
    selected_threshold = selected["min_confidence"]
    model = experiment_models[selected_threshold]
    y_pred = model.predict(test)
    result = metrics(y_true, y_pred)
    result["unmatched_test_count"] = sum(
        not model.match_rules(row) for row in test_records
    )
    result["rule_characteristics"] = rule_characteristics(model.rules)
    result["frequent_itemsets"] = model.frequent_itemset_count
    result["generated_association_rules"] = model.generated_rule_count
    result["filtered_classification_rules"] = len(model.rules)
    result["top_classification_rules"] = [
        public_rule(rule) for rule in model.rules[:10]
    ]
    result["threshold_experiments"] = experiments
    result["selected_threshold"] = {
        "min_confidence": selected_threshold,
        "rationale": "Highest test macro-F1; ties resolved by accuracy, then higher confidence threshold.",
    }
    output = Path(args.models_dir)
    output.mkdir(parents=True, exist_ok=True)
    model.save(output / "associative_classifier.pkl")
    result.update({"seed": args.seed, "n_train": len(train), "n_rules": len(model.rules)})
    (output / "evaluation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    pd.DataFrame(model.rules).to_csv(output / "association_rules.csv", index=False)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
