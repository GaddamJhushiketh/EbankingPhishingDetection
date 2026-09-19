"""Phase 8 model-derived explanation tests."""
from __future__ import annotations

from app import create_app
from app.ml.preprocessing import FEATURE_COLUMNS
from app.services.explanation import ExplanationService


def _valid_vector():
    vector = {column: 0 for column in FEATURE_COLUMNS}
    vector["age_of_domain"] = 1
    vector["having_IP_Address"] = 0
    return vector


class FakeModel:
    def __init__(self, rules):
        self.rules = rules

    def match_rules(self, record):
        items = {f"{key}={value}" for key, value in record.items()}
        return [
            rule for rule in self.rules
            if set(rule["antecedent"]).issubset(items)
        ]


def _rule(antecedent, confidence, lift, support, label=1):
    return {
        "antecedent": tuple(antecedent),
        "consequent": f"Result={'Legitimate' if label == 1 else 'Phishy'}",
        "class": label,
        "confidence": confidence,
        "lift": lift,
        "support": support,
        "antecedent_length": len(antecedent),
    }


def test_matching_rules_preserve_metrics_and_deterministic_order():
    rules = [
        _rule(["SFH=0"], 0.8, 2.0, 0.2),
        _rule(["SFH=0", "SSLfinal_State=0"], 0.8, 2.0, 0.2, label=-1),
        _rule(["SFH=0"], 0.9, 1.5, 0.1),
        _rule(["SFH=1"], 1.0, 4.0, 0.5),
    ]
    explanation = ExplanationService(FakeModel(rules)).explain(_valid_vector())
    assert explanation["matching_rule_count"] == 3
    assert [rule["confidence"] for rule in explanation["matching_rules"]] == [0.9, 0.8, 0.8]
    assert explanation["matching_rules"][1]["support"] == 0.2
    assert explanation["matching_rules"][1]["lift"] == 2.0
    assert explanation["matching_rules"][1]["class_label"] == "Phishy"
    assert explanation["feature_evidence"][0]["matching_rule_count"] == 3


def test_non_matching_rules_are_excluded_and_no_match_is_explicit():
    rules = [_rule(["SFH=1"], 1.0, 2.0, 0.5)]
    explanation = ExplanationService(FakeModel(rules)).explain(_valid_vector())
    assert explanation["matching_rules"] == []
    assert explanation["matching_rule_count"] == 0
    assert explanation["has_matching_rules"] is False


def test_explanation_has_no_probability_or_fabricated_semantics():
    explanation = ExplanationService(
        FakeModel([_rule(["SFH=0"], 0.9, 2.0, 0.2)])
    ).explain(_valid_vector())
    text = str(explanation)
    assert "probability_value" not in text.lower()
    assert "probability=" not in text.lower()
    assert "is phishing because" not in text.lower()
    assert explanation["feature_evidence"][0]["name"] == "SFH"
    assert explanation["feature_evidence"][0]["value"] == 0


def test_result_page_renders_model_derived_evidence_and_feature_analysis():
    response = create_app("testing").test_client().post(
        "/predict/features",
        data={column: str(value) for column, value in _valid_vector().items()},
    )
    assert response.status_code == 200
    assert b"MODEL-DERIVED EVIDENCE" in response.data
    assert b"Rule confidence" in response.data
    assert b"FEATURE ANALYSIS" in response.data
    assert b"prediction probability" in response.data
    assert b"URL_Length" in response.data
    assert b"Confidence:" not in response.data
    assert b"Probability:" not in response.data


def test_no_rule_match_state_renders_without_replacing_prediction(monkeypatch):
    application = create_app("testing")
    service = application.extensions["prediction_service"]
    explanation = {
        "matching_rule_count": 0,
        "matching_rules": [],
        "has_matching_rules": False,
        "feature_evidence": [
            {
                "name": column,
                "value": value,
                "allowed_values": [-1, 0, 1],
                "matching_rule_count": 0,
                "strongest_rule": None,
            }
            for column, value in _valid_vector().items()
        ],
        "ranking": "deterministic ranking",
        "limitations": "model-derived evidence only",
    }
    monkeypatch.setattr(
        service,
        "explain_features",
        lambda features: (1, _valid_vector(), explanation),
    )
    response = application.test_client().post(
        "/predict/features",
        data={column: str(value) for column, value in _valid_vector().items()},
    )
    assert response.status_code == 200
    assert b"No matching association rule was found" in response.data
    assert b"Legitimate" in response.data


def test_mapping_unverified_live_analysis_has_no_explanation(monkeypatch):
    application = create_app("testing")
    service = application.extensions["prediction_service"]
    monkeypatch.setattr(
        service,
        "analyze_url",
        lambda url: {
            "url": url,
            "SFH": {"raw_value": None, "encoded_value": None, "status": "mapping_unverified"},
        },
    )
    response = application.test_client().post(
        "/predict", data={"url": "https://example.com/"}
    )
    assert response.status_code == 200
    assert b"model prediction was not produced" in response.data
    assert b"MODEL-DERIVED EVIDENCE" not in response.data
    assert b"No predicted class or confidence was produced" in response.data


def test_history_exposes_rule_count_for_completed_predictions():
    application = create_app("testing")
    response = application.test_client().post(
        "/predict/features",
        data={column: str(value) for column, value in _valid_vector().items()},
    )
    assert response.status_code == 200
    history = application.test_client().get("/history")
    assert history.status_code == 200
    assert b"Rule evidence" in history.data
