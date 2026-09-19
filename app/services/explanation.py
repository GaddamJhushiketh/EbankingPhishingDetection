"""Model-derived explanations for encoded Dataset 379 predictions."""
from __future__ import annotations

from collections.abc import Mapping

from app.ml.preprocessing import (
    FEATURE_ALLOWED_VALUES,
    FEATURE_COLUMNS,
    TARGET_LABELS,
    validate_feature_vector,
)


class ExplanationService:
    """Expose matching learned rules without changing classifier behavior."""

    def __init__(self, model, *, max_rules: int = 5):
        if max_rules < 1:
            raise ValueError("max_rules must be positive")
        self.model = model
        self.max_rules = max_rules

    @staticmethod
    def _rule_view(rule: Mapping) -> dict:
        antecedent = tuple(str(item) for item in rule["antecedent"])
        return {
            "antecedent": list(antecedent),
            "consequent": str(rule["consequent"]),
            "class": int(rule["class"]),
            "class_label": TARGET_LABELS[int(rule["class"])],
            "support": float(rule["support"]),
            "confidence": float(rule["confidence"]),
            "lift": float(rule["lift"]),
            "antecedent_length": int(rule["antecedent_length"]),
        }

    def explain(self, features: Mapping[str, int]) -> dict:
        """Return deterministic, model-derived evidence for one vector."""

        validated = validate_feature_vector(features)
        matching = sorted(
            self.model.match_rules(validated),
            key=lambda rule: (
                -float(rule["confidence"]),
                -float(rule["lift"]),
                -float(rule["support"]),
                -int(rule["antecedent_length"]),
                int(rule["class"]),
                tuple(rule["antecedent"]),
            ),
        )
        viewed = [self._rule_view(rule) for rule in matching]
        feature_evidence = []
        for feature in FEATURE_COLUMNS:
            item = f"{feature}={validated[feature]}"
            related = [rule for rule in viewed if item in rule["antecedent"]]
            feature_evidence.append(
                {
                    "name": feature,
                    "value": validated[feature],
                    "allowed_values": sorted(FEATURE_ALLOWED_VALUES[feature]),
                    "matching_rule_count": len(related),
                    "strongest_rule": related[0] if related else None,
                }
            )
        return {
            "matching_rule_count": len(viewed),
            "matching_rules": viewed[: self.max_rules],
            "feature_evidence": feature_evidence,
            "has_matching_rules": bool(viewed),
            "ranking": (
                "Existing classifier order: descending rule confidence, lift, "
                "support, antecedent length, class value, then lexical antecedent order."
            ),
            "limitations": (
                "This is model-derived rule evidence, not a causal explanation "
                "or a calibrated prediction probability. Feature names and encoded "
                "values are shown without inventing Dataset 379 semantics."
            ),
        }
