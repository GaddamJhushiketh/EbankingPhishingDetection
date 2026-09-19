"""Phase 10 integration, hardening, and deployment-readiness tests."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from sqlalchemy import inspect, text

from app import create_app
from app.ml.preprocessing import FEATURE_COLUMNS
from app.models.db import PredictionHistory, db, ensure_history_schema
from app.services.feature_extraction import FeatureExtractionService
from app.services.http_client import SafeHTTPClient
from app.services.url_security import URLValidationError
from app.utils.model_integrity import ModelIntegrityError, load_verified_model


ROOT = Path(__file__).resolve().parents[1]
MODEL_HASH = "5149828ac3bf8aae0964a484d14db67228bfd9b3679487df7e521b1bb0fe78b4"


def valid_vector():
    values = {column: "0" for column in FEATURE_COLUMNS}
    values["age_of_domain"] = "1"
    values["having_IP_Address"] = "0"
    return values


def test_core_routes_and_static_asset_start_cleanly():
    client = create_app("testing").test_client()
    for path in ("/", "/health", "/status", "/dashboard", "/history", "/predict/features"):
        assert client.get(path).status_code == 200
    assert client.get("/static/css/main.css").status_code == 200


def test_feature_vector_end_to_end_persists_and_reaches_dashboard():
    application = create_app("testing")
    client = application.test_client()
    response = client.post("/predict/features", data=valid_vector())
    assert response.status_code == 200
    assert b"PREDICTION STATUS" in response.data
    assert b"MODEL-DERIVED EVIDENCE" in response.data
    assert b"prediction probability" in response.data
    with application.app_context():
        row = PredictionHistory.query.one()
        assert row.prediction_status == "predicted"
        assert row.prediction in {-1, 0, 1}
        assert row.label in {"Phishy", "Suspicious", "Legitimate"}
        assert set(row.features) == set(FEATURE_COLUMNS)
    dashboard = client.get("/dashboard")
    assert dashboard.status_code == 200
    assert b"Total analyses" in dashboard.data
    assert b"Dataset Feature Vector Test" in dashboard.data


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"SFH": "1"},
        {**valid_vector(), "SFH": "not-an-integer"},
        {**valid_vector(), "SFH": "9"},
    ],
)
def test_invalid_feature_requests_fail_without_persistence(payload):
    application = create_app("testing")
    response = application.test_client().post("/predict/features", data=payload)
    assert response.status_code == 400
    assert b"Model input unavailable" in response.data
    with application.app_context():
        assert PredictionHistory.query.count() == 0


def test_malformed_json_request_is_handled_without_stack_trace():
    application = create_app("testing")
    response = application.test_client().post(
        "/predict/features",
        data="{malformed",
        content_type="application/json",
    )
    assert response.status_code == 400
    assert b"Traceback" not in response.data
    assert b"Model input unavailable" in response.data


def test_live_analysis_remains_mapping_unavailable_and_redacts_error_url(monkeypatch):
    application = create_app("testing")
    service = application.extensions["prediction_service"]
    called = False

    def fail_prediction(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("classifier must not run for live mapping-unavailable flow")

    monkeypatch.setattr(
        service,
        "analyze_url",
        lambda url: {
            "url": url,
            "SFH": {"raw_value": None, "encoded_value": None, "status": "mapping_unverified"},
        },
    )
    monkeypatch.setattr(service, "predict_features", fail_prediction)
    sensitive_url = "https://" + "user" + ":" + "secret" + "@example.com/path?token=private"
    response = application.test_client().post(
        "/predict", data={"url": sensitive_url}
    )
    assert response.status_code == 200
    assert b"model prediction was not produced" in response.data
    assert b"secret" not in response.data
    assert b"token=private" not in response.data
    assert called is False
    with application.app_context():
        row = PredictionHistory.query.one()
        assert row.prediction_status == "feature_mapping_unavailable"
        assert row.prediction is None
        assert row.label is None


@pytest.mark.parametrize(
    "url",
    [
        "",
        "not-a-url",
        "ftp://example.com",
        "http://localhost",
        "http://127.0.0.1",
        "http://10.0.0.1",
        "http://169.254.169.254",
        "http://[::1]",
        "https://user:password@example.com",
        "https://" + ("a" * 2050) + ".com",
    ],
)
def test_unsafe_or_malformed_live_urls_fail_safely(url):
    application = create_app("testing")
    response = application.test_client().post("/predict", data={"url": url})
    assert response.status_code == 400
    assert b"Traceback" not in response.data
    assert b"SECRET_KEY" not in response.data
    assert b"MODEL_SHA256" not in response.data


def test_http_client_bounds_redirects_and_response_size(monkeypatch):
    client = SafeHTTPClient(max_bytes=4, max_redirects=0)

    class Response:
        is_redirect = False
        headers = {}

        def iter_content(self, chunk_size):
            yield b"12345"

        def close(self):
            return None

    monkeypatch.setattr("requests.Session.get", lambda *args, **kwargs: Response())
    with pytest.raises(ValueError, match="too large"):
        client.get("https://example.com")


def test_model_integrity_and_application_controlled_artifact():
    model_path = ROOT / "models" / "associative_classifier.pkl"
    assert hashlib.sha256(model_path.read_bytes()).hexdigest() == MODEL_HASH
    with pytest.raises(ModelIntegrityError):
        load_verified_model(model_path, lambda path: object(), expected_hash="0" * 64)
    application = create_app("testing")
    assert application.config["MODEL_FILE"] == model_path


def test_reproducibility_metadata_matches_protected_files():
    metadata = json.loads(
        (ROOT / "models" / "reproducibility_metadata.json").read_text(encoding="utf-8")
    )
    expected = {
        "dataset": "ba1b5a221262c2c5d6da04ecbbea93cba879cdf56e38b385952e87a897c45c69",
        "evaluation": "40812e719bd3f7ea8d046e6269a8b1aeda9f5d659e5c568cd6a4b14ec3dac07f",
        "association_rules": "6fb53a64e998588e773293c374a5c75ece1799f78679d69a913038463bf3dba6",
        "model": MODEL_HASH,
    }
    actual = {
        "dataset": hashlib.sha256((ROOT / "dataset/raw/phishing_dataset.csv").read_bytes()).hexdigest(),
        "evaluation": hashlib.sha256((ROOT / "models/evaluation.json").read_bytes()).hexdigest(),
        "association_rules": hashlib.sha256((ROOT / "models/association_rules.csv").read_bytes()).hexdigest(),
        "model": hashlib.sha256((ROOT / "models/associative_classifier.pkl").read_bytes()).hexdigest(),
    }
    assert actual == expected
    assert metadata["artifacts"]["model"]["sha256"] == MODEL_HASH


def test_production_configuration_still_fails_without_required_secrets(monkeypatch):
    monkeypatch.delenv("SECRET_KEY", raising=False)
    monkeypatch.delenv("MODEL_SHA256", raising=False)
    with pytest.raises(RuntimeError):
        create_app("production")


def test_existing_history_schema_gets_only_additive_status_column():
    application = create_app("testing")
    with application.app_context():
        db.session.execute(
            text("DROP TABLE prediction_history")
        )
        db.session.execute(
            text(
                "CREATE TABLE prediction_history ("
                "id INTEGER PRIMARY KEY, url VARCHAR(2048) NOT NULL, "
                "prediction INTEGER, label VARCHAR(32), features JSON NOT NULL, "
                "created_at DATETIME DEFAULT CURRENT_TIMESTAMP)"
            )
        )
        db.session.commit()
        ensure_history_schema()
        columns = {column["name"] for column in inspect(db.engine).get_columns("prediction_history")}
        assert "prediction_status" in columns
        assert "id" in columns
