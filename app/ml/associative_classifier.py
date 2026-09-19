"""Reproducible associative classification using mlxtend.

The classifier mines feature items and class items from the Phase 3
representation.  Rules are deliberately kept as plain dictionaries in the
persisted artifact so the model can be inspected without importing pandas.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

from .preprocessing import (
    ALLOWED_TARGET_VALUES,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    TARGET_LABELS,
    feature_items,
    validate_training_frame,
)


class AssociativeClassifier:
    """An associative-rule classifier with deterministic rule selection."""

    artifact_version = 2

    def __init__(
        self,
        min_support: float = 0.01,
        min_confidence: float = 0.5,
        min_lift: float = 1.0,
        max_len: int | None = None,
        random_state: int = 42,
    ) -> None:
        if not 0 < min_support <= 1 or not 0 <= min_confidence <= 1:
            raise ValueError("min_support must be in (0, 1] and min_confidence in [0, 1]")
        if min_lift < 0:
            raise ValueError("min_lift must be non-negative")
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.min_lift = min_lift
        self.max_len = max_len
        self.random_state = random_state
        self.rules: list[dict] = []
        self.majority_class: int | None = None
        self.class_counts: dict[int, int] = {}
        self.n_samples: int = 0
        self.frequent_itemset_count: int = 0
        self.generated_rule_count: int = 0
        # Training/evaluation provenance is populated by the Phase 4 training
        # script; classifier behavior and feature encoding remain unchanged.
        self.metadata: dict = {}

    @staticmethod
    def _transactions(frame: pd.DataFrame) -> list[list[str]]:
        return [
            [f"{column}={int(row[column])}" for column in FEATURE_COLUMNS]
            + [f"{TARGET_COLUMN}={TARGET_LABELS[int(row[TARGET_COLUMN])] }"]
            for _, row in frame.iterrows()
        ]

    def fit(self, frame: pd.DataFrame) -> "AssociativeClassifier":
        """Mine class-consequent rules from a duplicate-preserving frame."""
        validated = validate_training_frame(frame)
        self.n_samples = len(validated)
        counts = validated[TARGET_COLUMN].value_counts().to_dict()
        self.class_counts = {int(k): int(v) for k, v in counts.items()}
        self.majority_class = min(
            ALLOWED_TARGET_VALUES,
            key=lambda label: (-self.class_counts.get(label, 0), label),
        )
        transactions = self._transactions(validated)
        items = sorted({item for row in transactions for item in row})
        encoded = pd.DataFrame(
            [{item: item in row for item in items} for row in transactions],
            columns=items,
        )
        frequent = apriori(
            encoded,
            min_support=self.min_support,
            use_colnames=True,
            max_len=self.max_len,
        )
        self.frequent_itemset_count = len(frequent)
        if frequent.empty:
            self.rules = []
            self.generated_rule_count = 0
            return self
        mined = association_rules(frequent, metric="confidence", min_threshold=self.min_confidence)
        self.generated_rule_count = len(mined)
        selected: list[dict] = []
        for row in mined.itertuples(index=False):
            antecedent = frozenset(row.antecedents)
            consequent = frozenset(row.consequents)
            classes = [item for item in consequent if item.startswith("Result=")]
            if len(classes) != 1 or len(consequent) != 1 or row.lift < self.min_lift:
                continue
            class_text = classes[0].split("=", 1)[1]
            label = next((key for key, value in TARGET_LABELS.items() if value == class_text), None)
            if label is None or not antecedent:
                continue
            selected.append(
                {
                    "antecedent": tuple(sorted(antecedent)),
                    "consequent": classes[0],
                    "support": float(row.support),
                    "confidence": float(row.confidence),
                    "lift": float(row.lift),
                    "antecedent_length": len(antecedent),
                    "class": int(label),
                }
            )
        self.rules = sorted(selected, key=self._rule_sort_key)
        return self

    @staticmethod
    def _rule_sort_key(rule: Mapping) -> tuple:
        # Higher confidence/lift/support and longer rules win; lexical fields
        # make ties stable across pandas/mlxtend versions.
        return (
            -float(rule["confidence"]),
            -float(rule["lift"]),
            -float(rule["support"]),
            -int(rule["antecedent_length"]),
            int(rule["class"]),
            tuple(rule["antecedent"]),
        )

    def match_rules(self, record: Mapping[str, int]) -> list[dict]:
        """Return matching rules in the deterministic selection order."""
        items = set(feature_items(record))
        return [rule for rule in self.rules if set(rule["antecedent"]).issubset(items)]

    def predict_record(self, record: Mapping[str, int]) -> int:
        """Predict a class; use the deterministic training majority as fallback."""
        matches = self.match_rules(record)
        return int(matches[0]["class"] if matches else self.majority_class)

    predict_one = predict_record

    @property
    def default_class(self) -> int | None:
        """The class used when no mined antecedent matches."""
        return self.majority_class

    def predict(self, records: pd.DataFrame | Iterable[Mapping[str, int]]) -> list[int]:
        if isinstance(records, pd.DataFrame):
            values = records.loc[:, list(FEATURE_COLUMNS)].to_dict("records")
        else:
            values = list(records)
        return [self.predict_record(record) for record in values]

    def save(self, path: str | Path) -> Path:
        if self.majority_class is None:
            raise ValueError("Cannot save an unfitted classifier")
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as stream:
            pickle.dump(self, stream, protocol=pickle.HIGHEST_PROTOCOL)
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "AssociativeClassifier":
        with Path(path).open("rb") as stream:
            model = pickle.load(stream)
        if not isinstance(model, cls) or model.artifact_version != cls.artifact_version:
            raise ValueError("Unsupported associative classifier artifact")
        return model


def filter_class_rules(
    rules: Sequence[Mapping], *, min_confidence: float = 0.0, min_lift: float = 0.0
) -> list[dict]:
    """Filter already-mined rules without changing their deterministic order."""
    return [
        dict(rule)
        for rule in rules
        if float(rule["confidence"]) >= min_confidence and float(rule["lift"]) >= min_lift
    ]
