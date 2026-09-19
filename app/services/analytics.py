"""Read-only security analytics built from application history and metadata."""
from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from pathlib import Path

from sqlalchemy import func

from app.ml.preprocessing import FEATURE_ALLOWED_VALUES, FEATURE_COLUMNS
from app.models.db import PredictionHistory


class AnalyticsService:
    """Prepare safe, presentation-ready dashboard data."""

    def __init__(self, *, model, root: Path, integrity_configured: bool):
        self.model = model
        self.root = Path(root)
        self.integrity_configured = integrity_configured

    def _metadata(self, name: str) -> dict:
        path = self.root / "models" / name
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    @staticmethod
    def _filtered_query(filters: dict):
        query = PredictionHistory.query
        if filters.get("classification") in {"Phishy", "Suspicious", "Legitimate"}:
            query = query.filter(PredictionHistory.label == filters["classification"])
        if filters.get("status") in {"predicted", "feature_mapping_unavailable"}:
            query = query.filter(PredictionHistory.prediction_status == filters["status"])
        if filters.get("date_from"):
            query = query.filter(PredictionHistory.created_at >= filters["date_from"])
        if filters.get("date_to"):
            query = query.filter(PredictionHistory.created_at < filters["date_to"])
        return query

    @staticmethod
    def parse_filters(args) -> tuple[dict, str | None]:
        values = {
            "classification": (args.get("classification") or "").strip(),
            "status": (args.get("status") or "").strip(),
            "date_from": None,
            "date_to": None,
        }
        error = None
        for key, is_end in (("date_from", False), ("date_to", True)):
            raw = (args.get(key) or "").strip()
            if not raw:
                continue
            try:
                parsed = datetime.strptime(raw, "%Y-%m-%d")
                values[key] = parsed.replace(hour=23, minute=59, second=59, microsecond=999999) if is_end else parsed
            except ValueError:
                error = "Dates must use the YYYY-MM-DD format."
        if values["classification"] and values["classification"] not in {"Phishy", "Suspicious", "Legitimate"}:
            error = "Unsupported classification filter."
        if values["status"] and values["status"] not in {"predicted", "feature_mapping_unavailable"}:
            error = "Unsupported prediction status filter."
        return values, error

    def build_dashboard(self, filters: dict) -> dict:
        query = self._filtered_query(filters)
        total = query.with_entities(func.count(PredictionHistory.id)).scalar() or 0
        status_counts = {
            status: query.filter(PredictionHistory.prediction_status == status)
            .with_entities(func.count(PredictionHistory.id)).scalar() or 0
            for status in ("predicted", "feature_mapping_unavailable")
        }
        class_counts = {
            label: query.filter(PredictionHistory.label == label)
            .with_entities(func.count(PredictionHistory.id)).scalar() or 0
            for label in ("Phishy", "Suspicious", "Legitimate")
        }
        recent = query.order_by(PredictionHistory.created_at.desc()).limit(20).all()
        recent_activity = []
        rule_counts = []
        feature_counts = {name: Counter() for name in FEATURE_COLUMNS}
        vector_count = 0
        if self.model is not None:
            for row in query.filter(PredictionHistory.prediction_status == "predicted").all():
                try:
                    explanation = self.model.explain_features(row.features) if hasattr(self.model, "explain_features") else None
                    if explanation is not None:
                        rule_counts.append(int(explanation[2]["matching_rule_count"]))
                    for name in FEATURE_COLUMNS:
                        value = row.features.get(name)
                        if value in FEATURE_ALLOWED_VALUES[name]:
                            feature_counts[name][int(value)] += 1
                    vector_count += 1
                except (AttributeError, TypeError, ValueError):
                    continue
        for row in recent:
            count = None
            if row.prediction_status == "predicted" and self.model is not None:
                try:
                    count = self.model.explain_features(row.features)[2]["matching_rule_count"]
                except (AttributeError, TypeError, ValueError):
                    count = None
            recent_activity.append({
                "created_at": row.created_at,
                "url": row.url,
                "label": row.label,
                "status": row.prediction_status,
                "rule_count": count,
            })
        quality = self._metadata("dataset_quality.json")
        manifest = self._metadata("model_manifest.json")
        evaluation = self._metadata("evaluation.json")
        metadata = self._metadata("reproducibility_metadata.json")
        evaluation_report = evaluation
        return {
            "overview": {
                "total": int(total),
                "phishy": int(class_counts["Phishy"]),
                "suspicious": int(class_counts["Suspicious"]),
                "legitimate": int(class_counts["Legitimate"]),
                "mapping_unverified": int(status_counts["feature_mapping_unavailable"]),
            },
            "class_counts": class_counts,
            "recent_activity": recent_activity,
            "rule_evidence": {
                "predictions_with_rules": sum(count > 0 for count in rule_counts),
                "predictions_without_rules": sum(count == 0 for count in rule_counts),
                "average_matching_rules": (
                    sum(rule_counts) / len(rule_counts) if rule_counts else None
                ),
                "prediction_count": len(rule_counts),
            },
            "feature_distribution": {
                name: {
                    "observations": sum(counts.values()),
                    "values": [
                        {"value": value, "count": counts.get(value, 0)}
                        for value in sorted(FEATURE_ALLOWED_VALUES[name])
                        if counts.get(value, 0)
                    ],
                }
                for name, counts in feature_counts.items()
            } if vector_count else {},
            "evaluation": {
                "accuracy": evaluation_report.get("accuracy"),
                "macro_precision": evaluation_report.get("macro_avg", {}).get("precision"),
                "macro_recall": evaluation_report.get("macro_avg", {}).get("recall"),
                "macro_f1": evaluation_report.get("macro_avg", {}).get("f1"),
                "weighted_f1": evaluation_report.get("classification_report", {}).get("weighted avg", {}).get("f1"),
            },
            "dataset_quality": {
                "rows": quality.get("row_count"),
                "features": quality.get("feature_names", []),
                "duplicate_rows": quality.get("duplicate_rows"),
                "unique_rows": quality.get("unique_rows"),
                "conflicting_feature_groups": quality.get("conflicting_feature_group_count"),
                "missing_values": sum(quality.get("missing_values", {}).values()),
            },
            "model_status": {
                "name": manifest.get("model_name", "Associative classifier"),
                "type": manifest.get("model_type", "Associative Classification"),
                "loaded": self.model is not None,
                "integrity": (
                    "Verified"
                    if self.integrity_configured and self.model is not None
                    else "Development integrity verification not configured"
                ),
                "dataset_identity": "UCI Website Phishing Dataset 379",
                "artifact_sha256": manifest.get("artifact_sha256"),
                "schema": metadata.get("metadata_schema"),
            },
        }
