"""Validation and item encoding for associative classification.

This module deliberately does not train a model. It defines the one encoding
contract shared by future training and live prediction code.
"""

from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd


FEATURE_COLUMNS = (
    "SFH",
    "popUpWidnow",
    "SSLfinal_State",
    "Request_URL",
    "URL_of_Anchor",
    "web_traffic",
    "URL_Length",
    "age_of_domain",
    "having_IP_Address",
)
TARGET_COLUMN = "Result"
ALLOWED_FEATURE_VALUES = frozenset({-1, 0, 1})
FEATURE_ALLOWED_VALUES = {
    "SFH": frozenset({-1, 0, 1}),
    "popUpWidnow": frozenset({-1, 0, 1}),
    "SSLfinal_State": frozenset({-1, 0, 1}),
    "Request_URL": frozenset({-1, 0, 1}),
    "URL_of_Anchor": frozenset({-1, 0, 1}),
    "web_traffic": frozenset({-1, 0, 1}),
    "URL_Length": frozenset({-1, 0, 1}),
    "age_of_domain": frozenset({-1, 1}),
    "having_IP_Address": frozenset({0, 1}),
}
ALLOWED_TARGET_VALUES = frozenset({-1, 0, 1})

# This is the explicit three-class contract for the UCI Result encoding.
TARGET_LABELS = {
    -1: "Phishy",
    0: "Suspicious",
    1: "Legitimate",
}


class PreprocessingError(ValueError):
    """Raised when a dataset or live feature record violates the contract."""


def _validate_columns(columns: Sequence[str]) -> None:
    missing = [column for column in (*FEATURE_COLUMNS, TARGET_COLUMN) if column not in columns]
    if missing:
        raise PreprocessingError(
            "Missing required columns: " + ", ".join(missing)
        )


def _validate_values(frame: pd.DataFrame, columns: Sequence[str], description: str) -> None:
    for column in columns:
        if frame[column].isna().any():
            raise PreprocessingError(
                f"{description} column {column!r} contains missing values"
            )
        invalid = sorted(set(frame[column].tolist()) - ALLOWED_FEATURE_VALUES)
        if invalid:
            raise PreprocessingError(
                f"{description} column {column!r} contains invalid values {invalid}; "
                f"expected only {sorted(ALLOWED_FEATURE_VALUES)}"
            )


def validate_training_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate and return a copy of a labelled training frame.

    Duplicate rows are intentionally retained because association-rule mining
    uses the original record frequencies.
    """

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Training data must be provided as a pandas DataFrame")
    _validate_columns(frame.columns)
    selected = frame.loc[:, [*FEATURE_COLUMNS, TARGET_COLUMN]].copy()
    _validate_values(selected, FEATURE_COLUMNS, "Feature")

    target = selected[TARGET_COLUMN]
    if target.isna().any():
        raise PreprocessingError("Target column 'Result' contains missing values")
    invalid_target = sorted(set(target.tolist()) - ALLOWED_TARGET_VALUES)
    if invalid_target:
        raise PreprocessingError(
            f"Target column 'Result' contains invalid values {invalid_target}; "
            f"expected only {sorted(ALLOWED_TARGET_VALUES)}"
        )
    return selected


def load_training_frame(path: str | Path) -> pd.DataFrame:
    """Load and validate a labelled CSV without changing row frequency."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Training dataset not found: {source}")
    return validate_training_frame(pd.read_csv(source))


def feature_items(record: Mapping[str, int]) -> list[str]:
    """Convert one unlabelled feature record to canonical association items."""

    missing = [column for column in FEATURE_COLUMNS if column not in record]
    if missing:
        raise PreprocessingError(
            "Live feature record is missing required features: " + ", ".join(missing)
        )
    extra = sorted(set(record) - set(FEATURE_COLUMNS))
    if extra:
        raise PreprocessingError(
            "Live feature record contains unsupported features: " + ", ".join(extra)
        )

    validated =     values = pd.DataFrame([dict(record)], columns=FEATURE_COLUMNS)
    _validate_values(values, FEATURE_COLUMNS, "Live feature")
    return [f"{column}={int(values.iloc[0][column])}" for column in FEATURE_COLUMNS]


def validate_feature_vector(record: Mapping[str, int]) -> dict[str, int]:
    """Validate an explicitly supplied, unlabelled Dataset 379 vector."""

    missing = [column for column in FEATURE_COLUMNS if column not in record]
    if missing:
        raise PreprocessingError(
            "Live feature record is missing required features: " + ", ".join(missing)
        )
    extra = sorted(set(record) - set(FEATURE_COLUMNS))
    if extra:
        raise PreprocessingError(
            "Live feature record contains unsupported features: " + ", ".join(extra)
        )
    validated: dict[str, int] = {}
    for column in FEATURE_COLUMNS:
        try:
            value = int(record[column])
        except (TypeError, ValueError) as exc:
            raise PreprocessingError(f"Feature {column!r} must be an integer") from exc
        if value not in FEATURE_ALLOWED_VALUES[column]:
            raise PreprocessingError(
                f"Feature {column!r} has invalid value {value}; expected "
                f"{sorted(FEATURE_ALLOWED_VALUES[column])}"
            )
        validated[column] = value
    return validated


def labelled_items(record: Mapping[str, int]) -> list[str]:
    """Convert one labelled record to feature items plus its class item."""

    items = feature_items({column: record[column] for column in FEATURE_COLUMNS})
    if TARGET_COLUMN not in record:
        raise PreprocessingError("Labelled record is missing target column 'Result'")
    target = record[TARGET_COLUMN]
    if target not in ALLOWED_TARGET_VALUES:
        raise PreprocessingError(
            f"Target value {target!r} is invalid; expected only "
            f"{sorted(ALLOWED_TARGET_VALUES)}"
        )
    return [*items, f"{TARGET_COLUMN}={TARGET_LABELS[target]}"]


def prepare_mining_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    """Encode every source row as distinguishable items for rule mining."""

    validated = validate_training_frame(frame)
    return pd.DataFrame(
        [labelled_items(row.to_dict()) for _, row in validated.iterrows()],
        columns=[*FEATURE_COLUMNS, TARGET_COLUMN],
        index=validated.index,
    ).reset_index(drop=True)


def prepare_mining_csv(source_path: str | Path, output_path: str | Path) -> pd.DataFrame:
    """Validate, encode, and write the duplicate-preserving mining dataset."""

    encoded = prepare_mining_dataset(load_training_frame(source_path))
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded.to_csv(destination, index=False)
    return encoded
