"""Deterministic Phase 5 security, configuration, and service tests."""
from __future__ import annotations

import hashlib
import os

import pytest

from app import create_app
from config import Config
from app.services.feature_extraction import FeatureExtractionError, FeatureExtractionService
from app.ml.preprocessing import FEATURE_COLUMNS
from app.services.url_security import URLValidationError, redact_url
from app.utils.model_integrity import ModelIntegrityError, load_verified_model


def test_model_integrity_hash_and_constant_time_failure(tmp_path):
    artifact = tmp_path / "artifact"
    artifact.write_bytes(b"known artifact")
    digest = hashlib.sha256(b"known artifact").hexdigest()
    assert load_verified_model(artifact, lambda path: path.read_bytes(), expected_hash=digest) == b"known artifact"
    with pytest.raises(ModelIntegrityError):
        load_verified_model(artifact, lambda path: object(), expected_hash="0" * 64)


def test_model_integrity_missing_artifact_fails(tmp_path):
    with pytest.raises(ModelIntegrityError):
        load_verified_model(tmp_path / "missing.pkl", lambda path: object())


def test_production_requires_secret_and_model_hash(monkeypatch):
    monkeypatch.delenv("SECRET_KEY", raising=False)
    monkeypatch.delenv("MODEL_SHA256", raising=False)
    with pytest.raises(RuntimeError):
        create_app("production")


def test_development_secret_has_no_predictable_fallback():
    assert Config.SECRET_KEY
    assert Config.SECRET_KEY != "development-only-secret-key"


def test_url_only_rejects_unavailable_features():
    with pytest.raises(FeatureExtractionError, match="mapping_unverified"):
        FeatureExtractionService().extract("https://example.com/")


def test_provider_must_supply_all_non_url_features():
    provider = lambda url: {"SFH": 1, "popUpWidnow": 0, "Request_URL": 1}
    with pytest.raises(FeatureExtractionError, match="URL_of_Anchor"):
        FeatureExtractionService(strategy="provider", provider=provider).extract("https://example.com/")


def test_provider_returns_exact_nine_features_without_fabrication():
    values = {
        "SFH": 1, "popUpWidnow": 0, "SSLfinal_State": 1,
        "Request_URL": 1, "URL_of_Anchor": 1, "web_traffic": 0,
        "URL_Length": 1, "age_of_domain": 1, "having_IP_Address": 0,
    }
    features = FeatureExtractionService(strategy="provider", provider=lambda url: values).extract("https://example.com/")
    assert len(features) == 9
    assert features["SSLfinal_State"] == 1
    assert features["URL_Length"] == 1
    assert features["having_IP_Address"] == 0


def test_provider_values_are_domain_validated():
    values = {
        "SFH": 2, "popUpWidnow": 0, "SSLfinal_State": 1,
        "Request_URL": 1, "URL_of_Anchor": 1, "web_traffic": 0,
        "URL_Length": 1, "age_of_domain": 1, "having_IP_Address": 0,
    }
    with pytest.raises(FeatureExtractionError):
        FeatureExtractionService(strategy="provider", provider=lambda url: values).extract("https://example.com")


@pytest.mark.parametrize("url", ["file:///etc/passwd", "http://localhost", "http://127.0.0.1"])
def test_url_validation_rejects_unsafe_urls(url):
    with pytest.raises(URLValidationError):
        FeatureExtractionService().extract(url)


def test_routes_and_history_are_deterministic():
    application = create_app("testing")
    client = application.test_client()
    assert client.get("/").status_code == 200
    assert client.get("/history").status_code == 200
    assert client.get("/health").status_code == 200
    assert client.get("/predict/features").status_code == 200


def test_feature_vector_route_reaches_classifier_and_is_separate():
    application = create_app("testing")
    client = application.test_client()
    vector = {column: "0" for column in FEATURE_COLUMNS}
    vector["age_of_domain"] = "1"
    vector["having_IP_Address"] = "0"
    response = client.post("/predict/features", data=vector)
    assert response.status_code == 200
    assert b"Dataset Feature Vector Test" in response.data


def test_missing_feature_vector_blocks_model():
    application = create_app("testing")
    response = application.test_client().post(
        "/predict/features", data={"SFH": "1"}
    )
    assert response.status_code == 400
    assert b"Model input unavailable" in response.data


class _FakeResponse:
    status_code = 200
    headers = {"Content-Type": "text/html"}
    text = "<html><form></form><a href='https://external.example/x'>x</a></html>"


class _FakeHTTPClient:
    def get(self, url):
        return _FakeResponse()


def test_live_analysis_returns_raw_observations_without_encoded_values():
    analysis = FeatureExtractionService(http_client=_FakeHTTPClient()).analyze(
        "https://example.com/"
    )
    assert analysis["URL_Length"]["raw_value"] > 0
    assert analysis["URL_Length"]["encoded_value"] is None
    assert all(
        analysis[feature]["status"] == "mapping_unverified"
        for feature in FEATURE_COLUMNS
    )


def test_live_url_does_not_call_classifier_when_mapping_is_unavailable():
    application = create_app("testing")
    response = application.test_client().post(
        "/predict", data={"url": "https://example.com/"}
    )
    assert response.status_code == 200
    assert b"model prediction was not produced" in response.data
    assert b"No predicted class or confidence was produced" in response.data


def test_history_url_redaction():
    assert redact_url("https://user:pass@example.com/path?token=secret#fragment") == (
        "https://example.com/path"
    )
