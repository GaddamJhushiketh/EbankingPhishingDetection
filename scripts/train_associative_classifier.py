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
from app.ml.preprocessing import FEATURE_COLUMNS, TARGET_LABELS, TARGET_COLUMN, validate_training_frame


def stratified_split(frame: pd.DataFrame, test_size: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Strategy A: historical row-level stratified split (kept for comparison)."""
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


def _choose_group_counts(sizes: list[int], target: int, seed: int) -> set[int]:
    """Choose a deterministic subset of group positions closest to ``target``."""
    order = list(pd.Series(range(len(sizes))).sample(frac=1, random_state=seed))
    # Dynamic programming over record counts gives a closer 80/20 split than
    # selecting groups greedily, while retaining the seeded tie-breaking order.
    choices: dict[int, tuple[int, ...]] = {0: ()}
    for position in order:
        size = sizes[position]
        for total, selected in list(choices.items())[::-1]:
            new_total = total + size
            if new_total not in choices:
                choices[new_total] = selected + (position,)
    best = min(choices, key=lambda total: (abs(total - target), total > target))
    return set(choices[best])


def exact_record_group_split(
    frame: pd.DataFrame, test_size: float = 0.2, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Split exact feature+target records so no duplicate record crosses splits."""
    validated = validate_training_frame(frame).reset_index(drop=True)
    group_columns = list(FEATURE_COLUMNS) + [TARGET_COLUMN]
    grouped = validated.groupby(group_columns, sort=True, dropna=False)
    groups = list(grouped.indices.items())
    test_indices: set[int] = set()
    for label in sorted(TARGET_LABELS):
        labelled = [(key, indices) for key, indices in groups if int(key[-1]) == label]
        if len(labelled) <= 1:
            continue
        sizes = [len(indices) for _, indices in labelled]
        target = min(max(1, round(sum(sizes) * test_size)), sum(sizes) - 1)
        selected = _choose_group_counts(sizes, target, seed + label + 100)
        for position in selected:
            test_indices.update(labelled[position][1])
    test_mask = validated.index.isin(test_indices)
    train = validated.loc[~test_mask].sample(frac=1, random_state=seed).reset_index(drop=True)
    test = validated.loc[test_mask].sample(frac=1, random_state=seed + 1).reset_index(drop=True)
    train_keys = set(map(tuple, train[group_columns].itertuples(index=False, name=None)))
    test_keys = set(map(tuple, test[group_columns].itertuples(index=False, name=None)))
    feature_keys = list(FEATURE_COLUMNS)
    train_features = set(map(tuple, train[feature_keys].itertuples(index=False, name=None)))
    test_features = set(map(tuple, test[feature_keys].itertuples(index=False, name=None)))
    shared_features = train_features & test_features
    distributions = []
    all_feature_groups = validated.groupby(feature_keys, sort=True, dropna=False)
    for feature_key, feature_group in all_feature_groups:
        key = tuple(feature_key) if isinstance(feature_key, tuple) else (feature_key,)
        if key not in shared_features:
            continue
        distributions.append({
            "features": [int(value) for value in key],
            "train": {str(k): int(v) for k, v in train.loc[
                train[feature_keys].apply(tuple, axis=1) == key, TARGET_COLUMN
            ].value_counts().to_dict().items()},
            "test": {str(k): int(v) for k, v in test.loc[
                test[feature_keys].apply(tuple, axis=1) == key, TARGET_COLUMN
            ].value_counts().to_dict().items()},
            "conflicting_labels": feature_group[TARGET_COLUMN].nunique() > 1,
        })
    analysis = {
        "exact_record_groups_total": len(groups),
        "exact_record_groups_train": len(train_keys),
        "exact_record_groups_test": len(test_keys),
        "shared_exact_records": len(train_keys & test_keys),
        "feature_only_groups_total": len(set(map(tuple, validated[feature_keys].itertuples(index=False, name=None)))),
        "feature_only_groups_train": len(train_features),
        "feature_only_groups_test": len(test_features),
        "feature_only_groups_shared": len(shared_features),
        "feature_only_conflicting_groups_total": int(
            sum(group[TARGET_COLUMN].nunique() > 1 for _, group in all_feature_groups)
        ),
        "feature_only_conflicting_groups_shared": int(sum(
            item["conflicting_labels"] for item in distributions
        )),
        "shared_feature_group_label_distributions": distributions,
    }
    return train, test, analysis


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
    train, test, group_analysis = exact_record_group_split(frame, args.test_size, args.seed)
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
    # Strategy A remains available as a clearly labelled historical,
    # row-level baseline; it is not used to select or persist the model.
    baseline_train, baseline_test = stratified_split(frame, args.test_size, args.seed)
    baseline_model = AssociativeClassifier(
        args.min_support, selected_threshold, args.min_lift, random_state=args.seed
    ).fit(baseline_train)
    baseline_metrics = metrics(
        baseline_test[TARGET_COLUMN].astype(int).tolist(),
        baseline_model.predict(baseline_test),
    )
    result["historical_baseline_strategy_a"] = {
        "evaluation_strategy": "row_stratified_historical_baseline",
        "description": "Historical row-level stratified split retained for comparison only.",
        "accuracy": baseline_metrics["accuracy"],
        "macro_f1": baseline_metrics["macro_avg"]["f1"],
        "n_train": len(baseline_train),
        "n_test": len(baseline_test),
    }
    result["feature_only_group_analysis"] = group_analysis
    output = Path(args.models_dir)
    output.mkdir(parents=True, exist_ok=True)
    model.metadata = {
        "evaluation_strategy": "exact_record_group_aware",
        "random_seed": args.seed,
        "test_size_target": args.test_size,
        "duplicate_leakage_prevented": True,
        "shared_exact_records": group_analysis["shared_exact_records"],
    }
    model.save(output / "associative_classifier.pkl")
    result.update({
        "evaluation_strategy": "exact_record_group_aware",
        "random_seed": args.seed,
        "test_size_target": args.test_size,
        "duplicate_leakage_prevented": True,
        "shared_exact_records": group_analysis["shared_exact_records"],
        "seed": args.seed,
        "n_train": len(train),
        "n_rules": len(model.rules),
    })
    (output / "evaluation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    pd.DataFrame(model.rules).to_csv(output / "association_rules.csv", index=False)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
