import pandas as pd
import pytest

from app.ml.preprocessing import (
    FEATURE_COLUMNS,
    PreprocessingError,
    feature_items,
    labelled_items,
    prepare_mining_dataset,
    validate_training_frame,
)


def make_frame():
    return pd.DataFrame(
        [{**{column: -1 for column in FEATURE_COLUMNS}, "Result": 1}]
    )


def test_prepare_mining_dataset_preserves_duplicates_and_encodes_items():
    frame = pd.concat([make_frame(), make_frame()], ignore_index=True)

    encoded = prepare_mining_dataset(frame)

    assert len(encoded) == 2
    assert encoded.iloc[0]["SFH"] == "SFH=-1"
    assert encoded.iloc[0]["SSLfinal_State"] == "SSLfinal_State=-1"
    assert encoded.iloc[0]["Result"] == "Result=Legitimate"


def test_feature_items_matches_training_representation():
    record = {column: 0 for column in FEATURE_COLUMNS}

    items = feature_items(record)

    assert items == [f"{column}=0" for column in FEATURE_COLUMNS]


def test_labelled_items_encodes_phishy_class():
    record = {**{column: 1 for column in FEATURE_COLUMNS}, "Result": -1}

    assert labelled_items(record)[-1] == "Result=Phishy"


def test_missing_feature_is_rejected():
    record = {column: 0 for column in FEATURE_COLUMNS[:-1]}

    with pytest.raises(PreprocessingError, match="missing required features"):
        feature_items(record)


def test_invalid_domain_value_is_rejected():
    frame = make_frame()
    frame.loc[0, "SFH"] = 2

    with pytest.raises(PreprocessingError, match="invalid values"):
        validate_training_frame(frame)


def test_invalid_target_is_rejected():
    frame = make_frame()
    frame.loc[0, "Result"] = 2

    with pytest.raises(PreprocessingError, match="Target column"):
        validate_training_frame(frame)
