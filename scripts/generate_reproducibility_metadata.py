"""Generate hashes and reproducibility metadata from existing artifacts."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ml.preprocessing import FEATURE_COLUMNS, TARGET_COLUMN, TARGET_LABELS  # noqa: E402
from scripts.validate_dataset import validate_dataset  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]


def repo_path(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    dataset_path = ROOT / "dataset" / "raw" / "phishing_dataset.csv"
    processed_path = ROOT / "dataset" / "processed" / "phishing_clean.csv"
    model_path = ROOT / "models" / "associative_classifier.pkl"
    evaluation_path = ROOT / "models" / "evaluation.json"
    rules_path = ROOT / "models" / "association_rules.csv"
    _, quality = validate_dataset(dataset_path)
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    metadata = {
        "metadata_schema": "phase7-reproducibility-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "filename": repo_path(dataset_path),
            "sha256": sha256(dataset_path),
            "row_count": quality["row_count"],
            "column_count": quality["column_count"],
            "feature_names": list(FEATURE_COLUMNS),
            "target_column": TARGET_COLUMN,
            "target_mapping": {str(key): value for key, value in TARGET_LABELS.items()},
            "duplicate_rows": quality["duplicate_rows"],
            "unique_rows": quality["unique_rows"],
            "preprocessing": {
                "validated_copy": repo_path(processed_path),
                "validated_copy_sha256": sha256(processed_path),
                "primary_training_duplicates_retained": True,
                "deduplicated_sensitivity_artifact": "dataset/processed/phishing_dataset_deduplicated.csv",
                "conflicting_feature_groups": quality["conflicting_feature_group_count"],
            },
        },
        "training": {
            "command": "python scripts/train_associative_classifier.py --data dataset/processed/phishing_clean.csv --seed 42 --test-size 0.2 --min-support 0.01 --min-confidence 0.5 --min-lift 1.0",
            "min_support": 0.01,
            "min_confidence_argument": 0.5,
            "selected_min_confidence": evaluation["selected_threshold"]["min_confidence"],
            "min_lift": 1.0,
            "random_seed": evaluation["random_seed"],
            "test_size": evaluation["test_size_target"],
            "split_strategy": evaluation["evaluation_strategy"],
            "class_handling": "three-class target mapping; exact feature+target record groups cannot cross splits",
            "rule_mining_scope": "training partition only",
        },
        "artifacts": {
            "model": {
                "path": repo_path(model_path),
                "sha256": sha256(model_path),
            },
            "evaluation": {
                "path": repo_path(evaluation_path),
                "sha256": sha256(evaluation_path),
            },
            "association_rules": {
                "path": repo_path(rules_path),
                "sha256": sha256(rules_path),
            },
        },
    }
    output = ROOT / "models" / "reproducibility_metadata.json"
    output.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    manifest = {
        "model_name": "associative_classifier",
        "model_type": "Associative Classification",
        "artifact_path": "models/associative_classifier.pkl",
        "artifact_sha256": metadata["artifacts"]["model"]["sha256"],
        "dataset_sha256": metadata["dataset"]["sha256"],
        "training": metadata["training"],
        "evaluation_summary": {
            "accuracy": evaluation["accuracy"],
            "macro_f1": evaluation["macro_avg"]["f1"],
            "weighted_f1": evaluation["classification_report"]["weighted avg"]["f1"],
            "evaluation_strategy": evaluation["evaluation_strategy"],
        },
        "feature_names": list(FEATURE_COLUMNS),
        "target_mapping": metadata["dataset"]["target_mapping"],
        "generated_at_utc": metadata["generated_at_utc"],
    }
    (ROOT / "models" / "model_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
