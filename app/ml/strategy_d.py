"""Phase 4.1 Strategy D associative classifier.

This module is intentionally separate from the Phase 4 classifier.  Strategy
D uses class-relative support and a bounded, deterministic rule list.
"""

from __future__ import annotations

import pickle
from collections.abc import Iterable, Mapping
from pathlib import Path

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

from .preprocessing import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    TARGET_LABELS,
    feature_items,
    validate_training_frame,
)


class StrategyDClassifier:
    """Apriori classifier with Strategy D's class-relative rule policy."""

    artifact_version = 1

    def __init__(
        self,
        support: float = 0.002,
        confidence: float = 0.50,
        max_antecedent_length: int = 4,
        min_class_relative_support: float = 0.05,
        max_rules_per_class: int = 50,
        min_lift: float = 1.0,
    ) -> None:
        self.support = support
        self.confidence = confidence
        self.max_antecedent_length = max_antecedent_length
        self.min_class_relative_support = min_class_relative_support
        self.max_rules_per_class = max_rules_per_class
        self.min_lift = min_lift
        # Familiar names make the isolated artifact easy to inspect alongside
        # the original Phase 4 classifier.
        self.min_support = support
        self.min_confidence = confidence
        self.max_len = max_antecedent_length + 1  # includes the class item
        self.rules: list[dict] = []
        self.majority_class: int | None = None
        self.class_counts: dict[int, int] = {}
        self.n_samples = 0
        self.frequent_itemset_count = 0
        self.generated_rule_count = 0

    @staticmethod
    def _sort_key(rule: Mapping) -> tuple:
        return (
            -float(rule["confidence"]),
            -float(rule["relative_support"]),
            -float(rule["lift"]),
            int(rule["antecedent_length"]),
            tuple(rule["antecedent"]),
        )

    def fit(self, frame: pd.DataFrame) -> StrategyDClassifier:
        data = validate_training_frame(frame)
        self.n_samples = len(data)
        counts = data[TARGET_COLUMN].value_counts().to_dict()
        self.class_counts = {int(k): int(v) for k, v in counts.items()}
        self.majority_class = min(
            (-1, 0, 1), key=lambda c: (-self.class_counts.get(c, 0), c)
        )
        transactions = [
            [f"{column}={int(row[column])}" for column in FEATURE_COLUMNS]
            + [f"Result={TARGET_LABELS[int(row[TARGET_COLUMN])]}"]
            for _, row in data.iterrows()
        ]
        items = sorted({item for tx in transactions for item in tx})
        encoded = pd.DataFrame(
            [{item: item in tx for item in items} for tx in transactions], columns=items
        )
        frequent = apriori(
            encoded,
            min_support=self.support,
            use_colnames=True,
            max_len=self.max_antecedent_length + 1,
        )
        self.frequent_itemset_count = len(frequent)
        if frequent.empty:
            self.rules = []
            self.generated_rule_count = 0
            return self
        mined = association_rules(
            frequent, metric="confidence", min_threshold=self.confidence
        )
        self.generated_rule_count = len(mined)
        selected: list[dict] = []
        for row in mined.itertuples(index=False):
            consequent = tuple(row.consequents)
            if len(consequent) != 1 or not consequent[0].startswith("Result="):
                continue
            antecedent = tuple(sorted(row.antecedents))
            if not antecedent or len(antecedent) > self.max_antecedent_length:
                continue
            label_text = consequent[0].split("=", 1)[1]
            label = next(
                (key for key, value in TARGET_LABELS.items() if value == label_text),
                None,
            )
            if label is None:
                continue
            if float(row.lift) < self.min_lift:
                continue
            class_support = self.class_counts[int(label)] / self.n_samples
            relative = float(row.support) / class_support
            if relative < self.min_class_relative_support:
                continue
            selected.append(
                {
                    "antecedent": antecedent,
                    "consequent": consequent[0],
                    "support": float(row.support),
                    "confidence": float(row.confidence),
                    "lift": float(row.lift),
                    "relative_support": relative,
                    "antecedent_length": len(antecedent),
                    "class": int(label),
                }
            )
        self.rules = sorted(
            [
                rule
                for label in (-1, 0, 1)
                for rule in sorted(
                    (r for r in selected if r["class"] == label), key=self._sort_key
                )[: self.max_rules_per_class]
            ],
            key=self._sort_key,
        )
        return self

    def match_rules(self, record: Mapping[str, int]) -> list[dict]:
        items = set(feature_items(record))
        return [rule for rule in self.rules if set(rule["antecedent"]).issubset(items)]

    def predict_record(self, record: Mapping[str, int]) -> int:
        matches = self.match_rules(record)
        return int(matches[0]["class"] if matches else self.majority_class)

    def predict(self, records: pd.DataFrame | Iterable[Mapping[str, int]]) -> list[int]:
        values = (
            records.loc[:, list(FEATURE_COLUMNS)].to_dict("records")
            if isinstance(records, pd.DataFrame)
            else list(records)
        )
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
    def load(cls, path: str | Path) -> StrategyDClassifier:
        with Path(path).open("rb") as stream:
            model = pickle.load(stream)
        if not isinstance(model, cls) or model.artifact_version != cls.artifact_version:
            raise ValueError("Unsupported Strategy D artifact")
        return model
