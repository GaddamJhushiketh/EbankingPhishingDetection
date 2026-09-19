"""Phase 6 UI and route verification."""
from __future__ import annotations

from app import create_app
from app.ml.preprocessing import FEATURE_COLUMNS
from app.models.db import PredictionHistory, db


def _valid_vector():
    vector = {column: "0" for column in FEATURE_COLUMNS}
    vector["age_of_domain"] = "1"
    vector["having_IP_Address"] = "0"
    return vector


def test_dashboard_and_navigation_render():
    response = create_app("testing").test_client().get("/")
    assert response.status_code == 200
    for link in (
        b"Dashboard",
        b"Live Analysis",
        b"Feature Test",
        b"History",
        b"System Status",
        b"main.css",
    ):
        assert link in response.data


def test_phase6_pages_render():
    client = create_app("testing").test_client()
    assert client.get("/health").status_code == 200
    assert client.get("/history").status_code == 200
    assert client.get("/status").status_code == 200
    feature_page = client.get("/predict/features")
    assert feature_page.status_code == 200
    assert b"Dataset Feature Vector Test" in feature_page.data
    assert b"not extracted from a submitted website" in feature_page.data


def test_valid_feature_vector_renders_prediction_result():
    response = create_app("testing").test_client().post(
        "/predict/features", data=_valid_vector()
    )
    assert response.status_code == 200
    assert b"PREDICTION STATUS" in response.data
    assert b"How this result was produced" in response.data
    assert b"Confidence:" not in response.data
    assert b"Probability:" not in response.data


def test_missing_and_invalid_feature_vectors_render_controlled_errors():
    client = create_app("testing").test_client()
    missing = client.post("/predict/features", data={"SFH": "1"})
    assert missing.status_code == 400
    assert b"Model input unavailable" in missing.data

    invalid = _valid_vector()
    invalid["SFH"] = "9"
    response = client.post("/predict/features", data=invalid)
    assert response.status_code == 400
    assert b"Model input unavailable" in response.data


def test_live_mapping_unavailable_result_is_distinct_and_deterministic(monkeypatch):
    application = create_app("testing")
    service = application.extensions["prediction_service"]

    def fake_analysis(url):
        return {
            "url": url,
            "URL_Length": {"raw_value": len(url), "encoded_value": None},
            "SFH": {
                "raw_value": None,
                "encoded_value": None,
                "status": "mapping_unverified",
            },
        }

    monkeypatch.setattr(service, "analyze_url", fake_analysis)
    response = application.test_client().post(
        "/predict", data={"url": "https://example.com/"}
    )
    assert response.status_code == 200
    assert b"model prediction was not produced" in response.data
    assert b"No predicted class or confidence was produced" in response.data
    assert b"RAW OBSERVATION" in response.data
    assert b"PREDICTION STATUS" not in response.data


def test_history_and_status_pages_render_expected_states():
    application = create_app("testing")
    with application.app_context():
        db.session.add(PredictionHistory(
            url="Dataset Feature Vector Test",
            prediction=1,
            label="Legitimate",
            features=_valid_vector(),
            prediction_status="predicted",
        ))
        db.session.add(PredictionHistory(
            url="https://example.com/",
            prediction=None,
            label=None,
            features={"SFH": {"encoded_value": None}},
            prediction_status="feature_mapping_unavailable",
        ))
        db.session.commit()

    client = application.test_client()
    history = client.get("/history")
    assert b"Predicted" in history.data
    assert b"Feature Mapping Unavailable" in history.data
    assert b"Not available" in history.data

    status = client.get("/status")
    assert status.status_code == 200
    assert b"System status" in status.data
    assert b"mapping restricted" in status.data
