import json
from pathlib import Path

import pandas as pd

from app.ml.preprocessing import FEATURE_COLUMNS, TARGET_COLUMN, TARGET_LABELS
from app.ml.strategy_d import StrategyDClassifier


def test_strategy_d_artifact_contract():
    report = json.loads(Path("models/phase4_1_evaluation_results.json").read_text())
    assert report["split"]["n_train"] == 947
    assert report["split"]["n_test"] == 406
    assert report["parameters"] == {
        "support": 0.002,
        "confidence": 0.50,
        "max_antecedent_length": 4,
        "min_class_relative_support": 0.05,
        "max_rules_per_class": 50,
        "min_lift": 1.0,
    }
    cv = report["cross_validation"]
    assert cv["method"] == "StratifiedKFold"
    assert cv["n_splits"] == 5 and cv["shuffle"] and cv["random_state"] == 42
    assert cv["training_records_only"] is True
    assert report["test_evaluation"]["n_records"] == 406


def test_strategy_d_rules_are_bounded_and_ranked():
    model = StrategyDClassifier.load("models/phase4_1_associative_classifier.joblib")
    assert model.n_samples == 947
    assert all(len(rule["antecedent"]) <= 4 for rule in model.rules)
    assert all(rule["relative_support"] >= 0.05 for rule in model.rules)
    assert all(
        sum(rule["class"] == label for rule in model.rules) <= 50
        for label in (-1, 0, 1)
    )
    assert model.rules == sorted(model.rules, key=model._sort_key)


def test_exact_split_and_class_mapping_are_recorded():
    report = json.loads(Path("models/phase4_1_evaluation_results.json").read_text())
    assert report["split"]["train_class_distribution"] == {"-1": 491, "0": 72, "1": 384}
    assert report["split"]["test_class_distribution"] == {"-1": 211, "0": 31, "1": 164}
    assert TARGET_LABELS == {-1: "Phishy", 0: "Suspicious", 1: "Legitimate"}
    frame = pd.read_csv("dataset/raw/phishing_dataset.csv")
    assert frame.shape == (1353, 10)
    assert list(frame.columns) == [*FEATURE_COLUMNS, TARGET_COLUMN]


def test_cv_is_training_only_and_test_has_no_selection_fields():
    report = json.loads(Path("models/phase4_1_evaluation_results.json").read_text())
    cv = report["cross_validation"]
    assert cv["training_records_only"] is True
    assert cv["n_splits"] == 5
    assert cv["random_state"] == 42
    assert report["test_evaluation"]["n_records"] == 406
    assert "selected_threshold" not in report


def test_rule_metadata_contains_relative_support_and_deterministic_ties():
    rules = json.loads(Path("models/phase4_1_classification_rules.json").read_text())
    assert "relative_support" in rules["rules"][0]
    assert rules["ranking"].startswith("confidence desc")
    first = {
        "confidence": 0.5,
        "relative_support": 0.5,
        "lift": 1.0,
        "antecedent_length": 2,
        "antecedent": ("A=1", "B=0"),
    }
    second = {**first, "antecedent": ("A=1", "C=0")}
    assert StrategyDClassifier._sort_key(first) < StrategyDClassifier._sort_key(second)
