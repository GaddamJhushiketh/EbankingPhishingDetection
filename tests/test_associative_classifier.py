import pandas as pd

from app.ml.associative_classifier import AssociativeClassifier, filter_class_rules
from app.ml.preprocessing import FEATURE_COLUMNS


def frame(rows):
    return pd.DataFrame(
        [{**{column: values[i] for i, column in enumerate(FEATURE_COLUMNS)}, "Result": label}
         for values, label in rows]
    )


def test_training_preserves_duplicate_frequency_and_mines_class_rules():
    rows = [([1] * 9, 1)] * 3 + [([-1] * 9, -1)] * 2
    model = AssociativeClassifier(min_support=0.2, min_confidence=0.5).fit(frame(rows))
    assert model.n_samples == 5
    assert model.rules
    assert model.frequent_itemset_count >= 1
    assert model.generated_rule_count >= len(model.rules)
    assert all(rule["consequent"].startswith("Result=") for rule in model.rules)


def test_rule_filtering_and_deterministic_tie_breaking():
    rules = [
        {"confidence": 0.8, "lift": 1.1, "support": 0.4, "antecedent_length": 1,
         "class": 1, "antecedent": ("a",)},
        {"confidence": 0.8, "lift": 1.1, "support": 0.4, "antecedent_length": 1,
         "class": -1, "antecedent": ("b",)},
    ]
    assert filter_class_rules(rules, min_confidence=0.9) == []
    model = AssociativeClassifier()
    model.rules = sorted(rules, key=model._rule_sort_key)
    assert model.rules[0]["class"] == -1


def test_matching_no_rule_fallback_and_artifact_round_trip(tmp_path):
    rows = [([1] * 9, 1)] * 3 + [([-1] * 9, -1)]
    model = AssociativeClassifier(min_support=0.9, min_confidence=0.99).fit(frame(rows))
    record = {column: 0 for column in FEATURE_COLUMNS}
    assert model.match_rules(record) == []
    assert model.predict_one(record) == 1
    path = model.save(tmp_path / "model.pkl")
    loaded = AssociativeClassifier.load(path)
    assert loaded.predict_one(record) == model.predict_one(record)
    assert loaded.rules == model.rules
    assert loaded.frequent_itemset_count == model.frequent_itemset_count
    assert loaded.generated_rule_count == model.generated_rule_count
