"""Phase 9 security analytics dashboard tests."""
from __future__ import annotations

from datetime import datetime

from app import create_app
from app.ml.preprocessing import FEATURE_COLUMNS
from app.models.db import PredictionHistory, db


def _vector(value=0):
    values = {column: value for column in FEATURE_COLUMNS}
    values["age_of_domain"] = 1
    values["having_IP_Address"] = 0
    return values


def _insert_history(application):
    with application.app_context():
        db.session.add_all(
            [
                PredictionHistory(
                    url="Dataset Feature Vector Test",
                    prediction=-1,
                    label="Phishy",
                    features=_vector(-1),
                    prediction_status="predicted",
                    created_at=datetime(2026, 9, 18, 10, 0),
                ),
                PredictionHistory(
                    url="Dataset Feature Vector Test",
                    prediction=1,
                    label="Legitimate",
                    features=_vector(1),
                    prediction_status="predicted",
                    created_at=datetime(2026, 9, 19, 10, 0),
                ),
                PredictionHistory(
                    url="https://example.com/",
                    prediction=None,
                    label=None,
                    features={"SFH": {"encoded_value": None}},
                    prediction_status="feature_mapping_unavailable",
                    created_at=datetime(2026, 9, 19, 11, 0),
                ),
            ]
        )
        db.session.commit()


def test_dashboard_empty_state_and_print_metadata():
    application = create_app("testing")
    response = application.test_client().get("/dashboard")
    assert response.status_code == 200
    assert b"Security Analysis Dashboard" in response.data
    assert b"No prediction history available yet." in response.data
    assert b"Offline Dataset 379 Evaluation" in response.data
    css = application.test_client().get("/static/css/main.css")
    assert b"@media print" in css.data


def test_dashboard_counts_recent_activity_and_metadata():
    application = create_app("testing")
    _insert_history(application)
    response = application.test_client().get("/dashboard")
    assert response.status_code == 200
    assert b"Total analyses" in response.data
    assert b"Phishy detections" in response.data
    assert b"Mapping-unverified" in response.data
    assert b"Generated from application data" in response.data
    assert b"UCI Website Phishing Dataset 379" in response.data
    assert b"Dataset Feature Vector Test" in response.data


def test_dashboard_filters_are_server_side_and_safe():
    application = create_app("testing")
    _insert_history(application)
    client = application.test_client()
    filtered = client.get("/dashboard?classification=Phishy&status=predicted")
    assert filtered.status_code == 200
    assert b"Total analyses" in filtered.data
    assert b"Unsupported classification filter" not in filtered.data
    injected = client.get("/dashboard?classification=Phishy%27%20OR%201%3D1")
    assert injected.status_code == 200
    assert b"Unsupported classification filter" in injected.data
    bad_date = client.get("/dashboard?date_from=not-a-date")
    assert bad_date.status_code == 200
    assert b"Dates must use the YYYY-MM-DD format" in bad_date.data


def test_dashboard_feature_distribution_uses_encoded_values():
    application = create_app("testing")
    _insert_history(application)
    response = application.test_client().get("/dashboard?status=predicted")
    assert response.status_code == 200
    assert b"Stored encoded feature values" in response.data
    assert b"SFH" in response.data
    assert b"Value -1" in response.data
    assert b"Value 1" in response.data
    assert b"SSL is unsafe" not in response.data


def test_dashboard_rule_evidence_and_mapping_unverified_handling():
    application = create_app("testing")
    _insert_history(application)
    response = application.test_client().get("/dashboard")
    assert response.status_code == 200
    assert b"Predictions with matching rules" in response.data
    assert b"Predictions without matching rules" in response.data
    assert b"Mapping unavailable" in response.data
    assert b"Not available" in response.data


def test_dashboard_status_does_not_expose_secrets_or_filesystem_paths():
    application = create_app("testing")
    response = application.test_client().get("/dashboard")
    assert b"SECRET_KEY" not in response.data
    assert b"DB_PASSWORD" not in response.data
    assert b"copilot-worktrees" not in response.data
    assert b"Development integrity verification not configured" in response.data
