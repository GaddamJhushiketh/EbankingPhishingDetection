"""Deterministic Phase 5 security, configuration, and service tests."""
from __future__ import annotations

import hashlib
import os

import pytest

from app import create_app
from config import Config
from app.services.feature_extraction import FeatureExtractionError, FeatureExtractionService
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


def test_history_url_redaction():
    assert redact_url("https://user:pass@example.com/path?token=secret#fragment") == (
        "https://example.com/path"
    )
