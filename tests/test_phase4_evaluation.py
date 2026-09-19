import json
from pathlib import Path

import pandas as pd

from app.ml.associative_classifier import AssociativeClassifier
from app.ml.preprocessing import FEATURE_COLUMNS, TARGET_COLUMN, validate_training_frame
from scripts.train_associative_classifier import exact_record_group_split


DATA = Path("dataset/processed/phishing_clean.csv")


def test_exact_group_split_is_deterministic_stratified_and_leak_free():
    frame = validate_training_frame(pd.read_csv(DATA))
    train_a, test_a, analysis_a = exact_record_group_split(frame, 0.2, 42)
    train_b, test_b, analysis_b = exact_record_group_split(frame, 0.2, 42)
    keys = list(FEATURE_COLUMNS) + [TARGET_COLUMN]
    assert train_a.equals(train_b)
    assert test_a.equals(test_b)
    assert analysis_a == analysis_b
    assert set(map(tuple, train_a[keys].itertuples(index=False, name=None))).isdisjoint(
        set(map(tuple, test_a[keys].itertuples(index=False, name=None)))
    )
    assert analysis_a["shared_exact_records"] == 0
    assert abs(len(test_a) / len(frame) - 0.2) <= 0.01
    assert set(train_a[TARGET_COLUMN]) == {-1, 0, 1}
    assert set(test_a[TARGET_COLUMN]) == {-1, 0, 1}


def test_persisted_evaluation_metadata_and_model_are_usable():
    report = json.loads(Path("models/evaluation.json").read_text(encoding="utf-8"))
    assert report["evaluation_strategy"] == "exact_record_group_aware"
    assert report["random_seed"] == 42
    assert report["test_size_target"] == 0.2
    assert report["duplicate_leakage_prevented"] is True
    assert report["shared_exact_records"] == 0
    assert report["feature_only_group_analysis"]["shared_feature_group_label_distributions"]
    model = AssociativeClassifier.load("models/associative_classifier.pkl")
    frame = validate_training_frame(pd.read_csv(DATA))
    assert len(model.predict(frame.loc[:, list(FEATURE_COLUMNS)].head(5))) == 5


def test_rules_are_mined_from_training_partition_only():
    frame = validate_training_frame(pd.read_csv(DATA))
    train, test, _ = exact_record_group_split(frame, 0.2, 42)
    model = AssociativeClassifier(min_support=0.01, min_confidence=0.7).fit(train)
    train_items = {
        f"{column}={int(row[column])}"
        for _, row in train.iterrows()
        for column in FEATURE_COLUMNS
    }
    assert model.n_samples == len(train)
    assert all(set(rule["antecedent"]).issubset(train_items) for rule in model.rules)
    assert len(test) > 0
