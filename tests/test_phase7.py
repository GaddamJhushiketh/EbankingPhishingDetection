"""Phase 7 reproducibility and artifact metadata tests."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from app.ml.preprocessing import FEATURE_COLUMNS, TARGET_COLUMN
from scripts.validate_dataset import validate_dataset


ROOT = Path(__file__).resolve().parents[1]


def test_real_dataset_validation_and_quality():
    frame, report = validate_dataset(ROOT / "dataset/raw/phishing_dataset.csv")
    assert len(frame) == 1353
    assert list(frame.columns) == [*FEATURE_COLUMNS, TARGET_COLUMN]
    assert report["duplicate_rows"] == 629
    assert report["unique_rows"] == 724
    assert report["conflicting_feature_group_count"] == 48


def test_invalid_dataset_structure_is_rejected(tmp_path):
    invalid = pd.DataFrame({"wrong": [1]})
    path = tmp_path / "invalid.csv"
    invalid.to_csv(path, index=False)
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_dataset(path)


@pytest.mark.parametrize(
    "column,value",
    [("age_of_domain", 0), ("having_IP_Address", -1), ("Result", 2)],
)
def test_dataset_value_domains_are_rejected(tmp_path, column, value):
    source = pd.read_csv(ROOT / "dataset/raw/phishing_dataset.csv")
    source.loc[0, column] = value
    path = tmp_path / "invalid-values.csv"
    source.to_csv(path, index=False)
    with pytest.raises(ValueError):
        validate_dataset(path)


def test_metadata_and_manifest_hashes_match_files():
    metadata = json.loads(
        (ROOT / "models/reproducibility_metadata.json").read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (ROOT / "models/model_manifest.json").read_text(encoding="utf-8")
    )
    for entry in metadata["artifacts"].values():
        path = ROOT / entry["path"]
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        assert actual == entry["sha256"]
    assert manifest["artifact_sha256"] == metadata["artifacts"]["model"]["sha256"]
    assert manifest["feature_names"] == list(FEATURE_COLUMNS)


def test_phase4_evaluation_values_remain_unchanged():
    evaluation = json.loads((ROOT / "models/evaluation.json").read_text(encoding="utf-8"))
    assert evaluation["accuracy"] == 0.7822878228782287
    assert evaluation["macro_avg"]["precision"] == 0.8522692601067887
    assert evaluation["macro_avg"]["recall"] == 0.6165223665223665
    assert evaluation["macro_avg"]["f1"] == 0.6420598680872653
    assert evaluation["classification_report"]["weighted avg"]["f1"] == 0.7669808533701776


def test_validation_script_runs_without_modifying_raw_dataset():
    result = subprocess.run(
        [sys.executable, "scripts/validate_dataset.py"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert '"row_count": 1353' in result.stdout
