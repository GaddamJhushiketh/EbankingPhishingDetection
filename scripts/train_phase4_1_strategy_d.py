"""Train and evaluate the isolated Phase 4.1 Strategy D model."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split

from app.ml.preprocessing import FEATURE_COLUMNS, TARGET_COLUMN, load_training_frame
from app.ml.strategy_d import StrategyDClassifier


def evaluate(model, frame: pd.DataFrame) -> dict:
    y_true = frame[TARGET_COLUMN].astype(int).tolist()
    y_pred = model.predict(frame)
    labels = [-1, 0, 1]
    feature_records = frame.loc[:, list(FEATURE_COLUMNS)].to_dict("records")
    records = []
    errors = []
    matched = 0
    default = 0
    for index, (actual, predicted) in enumerate(zip(y_true, y_pred)):
        matches = model.match_rules(feature_records[index])
        selected = matches[0] if matches else None
        matched += bool(matches)
        default += not matches
        record = {
            "index": int(frame.index[index]),
            "actual": actual,
            "predicted": predicted,
            "matched": bool(matches),
            "default_rule": not matches,
            "matching_rule_count": len(matches),
            "selected_rule": selected,
        }
        records.append(record)
        if actual != predicted:
            errors.append(record)
    by_actual = {
        str(label): {
            "total": sum(actual == label for actual in y_true),
            "matched": sum(
                actual == label and bool(model.match_rules(feature_records[i]))
                for i, actual in enumerate(y_true)
            ),
            "unmatched": sum(
                actual == label and not model.match_rules(feature_records[i])
                for i, actual in enumerate(y_true)
            ),
        }
        for label in labels
    }
    by_predicted = {
        str(label): sum(
            predicted == label and bool(model.match_rules(feature_records[i]))
            for i, predicted in enumerate(y_pred)
        )
        for label in labels
    }
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(
            precision_score(
                y_true, y_pred, labels=labels, average="macro", zero_division=0
            )
        ),
        "macro_recall": float(
            recall_score(
                y_true, y_pred, labels=labels, average="macro", zero_division=0
            )
        ),
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "weighted_precision": float(
            precision_score(
                y_true, y_pred, labels=labels, average="weighted", zero_division=0
            )
        ),
        "weighted_recall": float(
            recall_score(
                y_true, y_pred, labels=labels, average="weighted", zero_division=0
            )
        ),
        "weighted_f1": float(
            f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "labels": labels,
        "classification_report": classification_report(
            y_true, y_pred, labels=labels, output_dict=True, zero_division=0
        ),
        "n_records": len(frame),
        "rule_matched_records": matched,
        "unmatched_records": len(frame) - matched,
        "rule_coverage": matched / len(frame),
        "coverage_by_actual_class": by_actual,
        "coverage_by_predicted_class": by_predicted,
        "default_rule_predictions": default,
        "default_rule_correct": sum(
            item["default_rule"] and item["actual"] == item["predicted"]
            for item in records
        ),
        "records": records,
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="dataset/raw/phishing_dataset.csv")
    parser.add_argument("--models-dir", default="models")
    args = parser.parse_args()
    frame = load_training_frame(args.data)
    train, test = train_test_split(
        frame, test_size=0.30, stratify=frame[TARGET_COLUMN], random_state=42
    )
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
    params = {
        "support": 0.002,
        "confidence": 0.50,
        "max_antecedent_length": 4,
        "min_class_relative_support": 0.05,
        "max_rules_per_class": 50,
        "min_lift": 1.0,
    }
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    folds = []
    for fold, (fit_idx, valid_idx) in enumerate(
        splitter.split(train, train[TARGET_COLUMN]), 1
    ):
        candidate = StrategyDClassifier(**params).fit(train.iloc[fit_idx])
        result = evaluate(candidate, train.iloc[valid_idx])
        result.update(
            {"fold": fold, "n_fit": len(fit_idx), "n_validation": len(valid_idx)}
        )
        folds.append(result)
    model = StrategyDClassifier(**params).fit(train)
    output = Path(args.models_dir)
    output.mkdir(parents=True, exist_ok=True)
    model_path = output / "phase4_1_associative_classifier.joblib"
    model.save(model_path)
    rules_path = output / "phase4_1_classification_rules.json"
    report = {
        "strategy": "Phase 4.1 Strategy D",
        "data_source": str(Path(args.data)),
        "raw_dataset_only": True,
        "random_state": 42,
        "split": {
            "test_size": 0.30,
            "stratified": True,
            "n_total": len(frame),
            "n_train": len(train),
            "n_test": len(test),
            "train_class_distribution": {
                str(k): int(v)
                for k, v in train[TARGET_COLUMN].value_counts().to_dict().items()
            },
            "test_class_distribution": {
                str(k): int(v)
                for k, v in test[TARGET_COLUMN].value_counts().to_dict().items()
            },
        },
        "parameters": params,
        "cross_validation": {
            "method": "StratifiedKFold",
            "n_splits": 5,
            "shuffle": True,
            "random_state": 42,
            "training_records_only": True,
            "folds": folds,
            "mean_accuracy": float(sum(x["accuracy"] for x in folds) / len(folds)),
            "mean_macro_f1": float(sum(x["macro_f1"] for x in folds) / len(folds)),
        },
        "final_model": {
            "n_samples": model.n_samples,
            "n_rules": len(model.rules),
            "rules_per_class": {
                str(c): sum(r["class"] == c for r in model.rules) for c in (-1, 0, 1)
            },
        },
        "test_evaluation": evaluate(model, test),
    }
    report["historical_reference"] = {
        "cv_macro_f1": 0.7860,
        "test_accuracy": 0.8571,
        "test_macro_f1": 0.7344,
        "rule_counts": {"1": 50, "0": 15, "-1": 50},
        "status": "reference_only; not used for tuning or validation",
    }
    report["artifact_paths"] = {
        "model": str(model_path),
        "evaluation": "models/phase4_1_evaluation_results.json",
        "rules": str(rules_path),
        "metadata": "models/phase4_1_reproducibility_metadata.json",
    }
    (output / "phase4_1_evaluation_results.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    rules_payload = {
        "strategy": "Phase 4.1 Strategy D",
        "ranking": "confidence desc, relative_support desc, lift desc, antecedent_length asc, antecedent lexicographic asc",
        "rules": [
            {**rule, "antecedent": list(rule["antecedent"])} for rule in model.rules
        ],
    }
    rules_path.write_text(json.dumps(rules_payload, indent=2), encoding="utf-8")
    dataset_hash = hashlib.sha256(Path(args.data).read_bytes()).hexdigest()
    metadata = {
        "dataset": {
            "path": args.data,
            "sha256": dataset_hash,
            "rows": len(frame),
            "columns": list(frame.columns),
        },
        "class_mapping": {"-1": "Phishy", "0": "Suspicious", "1": "Legitimate"},
        "split": {
            "test_size": 0.30,
            "random_state": 42,
            "stratified": True,
            "train": len(train),
            "test": len(test),
        },
        "parameters": params,
        "cross_validation": {
            "n_splits": 5,
            "shuffle": True,
            "random_state": 42,
            "training_records_only": True,
        },
        "ranking": rules_payload["ranking"],
        "final_rule_counts": report["final_model"]["rules_per_class"],
        "artifacts": {
            "model": {
                "path": str(model_path),
                "sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
            },
            "evaluation": {
                "path": "models/phase4_1_evaluation_results.json",
                "sha256": hashlib.sha256(
                    (output / "phase4_1_evaluation_results.json").read_bytes()
                ).hexdigest(),
            },
            "rules": {
                "path": str(rules_path),
                "sha256": hashlib.sha256(rules_path.read_bytes()).hexdigest(),
            },
        },
    }
    (output / "phase4_1_reproducibility_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    test_eval = report["test_evaluation"]
    (Path("reports")).mkdir(exist_ok=True)
    (Path("reports") / "phase4_1_evaluation_report.md").write_text(
        "# Phase 4.1 Strategy D Evaluation\n\n"
        "This is an isolated reproduction; existing Phase 4 artifacts were not modified.\n\n"
        f"- Train/test: {len(train)}/{len(test)}\n"
        f"- CV macro F1: {report['cross_validation']['mean_macro_f1']:.6f}\n"
        f"- Test accuracy: {test_eval['accuracy']:.6f}\n"
        f"- Test macro F1: {test_eval['macro_f1']:.6f}\n"
        f"- Rule coverage: {test_eval['rule_coverage']:.6%}\n"
        f"- Unmatched/default predictions: {test_eval['unmatched_records']}\n\n"
        "The historical values supplied in the specification are reference-only and were not used for tuning.\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
