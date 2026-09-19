"""Validate the project dataset without modifying it."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ml.preprocessing import (  # noqa: E402
    FEATURE_ALLOWED_VALUES,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    validate_training_frame,
)


def dataset_quality(frame: pd.DataFrame) -> dict:
    """Return deterministic quality statistics for a validated dataset."""

    feature_groups = frame.groupby(list(FEATURE_COLUMNS), dropna=False, sort=True)
    conflicting = [
        {
            "features": [int(value) for value in (key if isinstance(key, tuple) else (key,))],
            "target_distribution": {
                str(label): int(count)
                for label, count in group[TARGET_COLUMN].value_counts().sort_index().items()
            },
        }
        for key, group in feature_groups
        if group[TARGET_COLUMN].nunique() > 1
    ]
    return {
        "row_count": int(len(frame)),
        "column_count": int(len(frame.columns)),
        "columns": list(frame.columns),
        "feature_names": list(FEATURE_COLUMNS),
        "target_column": TARGET_COLUMN,
        "missing_values": {
            str(column): int(count) for column, count in frame.isna().sum().items()
        },
        "duplicate_rows": int(frame.duplicated().sum()),
        "unique_rows": int(len(frame.drop_duplicates())),
        "target_distribution": {
            str(label): int(count)
            for label, count in frame[TARGET_COLUMN].value_counts().sort_index().items()
        },
        "feature_only_group_count": int(frame.groupby(list(FEATURE_COLUMNS), dropna=False).ngroups),
        "conflicting_feature_group_count": len(conflicting),
        "conflicting_feature_groups": conflicting,
        "allowed_feature_values": {
            column: sorted(values) for column, values in FEATURE_ALLOWED_VALUES.items()
        },
        "allowed_target_values": [-1, 0, 1],
    }


def validate_dataset(path: str | Path) -> tuple[pd.DataFrame, dict]:
    """Load and validate a CSV, raising a clear error for invalid input."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Dataset not found: {source}")
    frame = pd.read_csv(source)
    validated = validate_training_frame(frame)
    for column, allowed in FEATURE_ALLOWED_VALUES.items():
        invalid = sorted(set(validated[column].tolist()) - allowed)
        if invalid:
            raise ValueError(
                f"Dataset feature column {column!r} contains invalid values {invalid}; "
                f"expected only {sorted(allowed)}"
            )
    return validated, dataset_quality(validated)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="dataset/raw/phishing_dataset.csv")
    parser.add_argument("--output", default="models/dataset_quality.json")
    args = parser.parse_args()
    _, report = validate_dataset(args.data)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
