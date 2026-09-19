from app.services.dataset379_encoder import Dataset379Encoder
from app import create_app
from app.ml.preprocessing import FEATURE_COLUMNS


def base_observations():
    return {
        "url": "https://example.com/login",
        "resource_urls": [],
        "anchor_hrefs": [],
        "form_actions": [],
        "popup_credential_fields": False,
        "domain_age_months": 6,
        "hostname_is_ip": False,
        "ssl_state": None,
        "web_traffic": None,
    }


def encode(**changes):
    observations = base_observations()
    observations.update(changes)
    return Dataset379Encoder().encode(observations)


def test_url_length_boundaries():
    encoder = Dataset379Encoder()
    for length, expected in ((53, 1), (54, 0), (75, 0), (76, -1)):
        result = encoder.encode({**base_observations(), "url": "https://" + "a" * (length - 8)})
        assert result["features"]["URL_Length"]["encoded_value"] == expected


def test_request_url_boundaries():
    for percentage, expected in ((21.9, 1), (22, 0), (61, 0), (61.1, -1)):
        total = 1000
        external = round(total * percentage / 100)
        resources = (
            [{"url": "https://cdn.example.net/a.js"} for _ in range(external)]
            + [{"url": "https://example.com/a.js"} for _ in range(total - external)]
        )
        result = encode(resource_urls=resources)
        assert result["features"]["Request_URL"]["encoded_value"] == expected


def test_anchor_boundaries():
    for percentage, expected in ((30.9, 1), (31, 0), (67, 0), (67.1, -1)):
        total = 1000
        external = round(total * percentage / 100)
        anchors = (
            ["https://other.example/a"] * external
            + ["/local"] * (total - external)
        )
        result = encode(anchor_hrefs=anchors)
        assert result["features"]["URL_of_Anchor"]["encoded_value"] == expected


def test_sfh_mappings():
    assert encode(form_actions=[""])["features"]["SFH"]["encoded_value"] == -1
    assert encode(form_actions=["about:blank"])["features"]["SFH"]["encoded_value"] == -1
    assert encode(form_actions=[])["features"]["SFH"]["encoded_value"] == -1
    assert encode(form_actions=["/submit"])["features"]["SFH"]["encoded_value"] == 1
    assert encode(form_actions=["https://other.example/submit"])["features"]["SFH"]["encoded_value"] == 0


def test_popup_mappings_only_credential_popups_are_phishy():
    assert encode(popup_credential_fields=True)["features"]["popUpWidnow"]["encoded_value"] == -1
    assert encode(popup_credential_fields=False)["features"]["popUpWidnow"]["encoded_value"] == 1


def test_domain_age_and_ip_mappings():
    assert encode(domain_age_months=5)["features"]["age_of_domain"]["encoded_value"] == -1
    assert encode(domain_age_months=6)["features"]["age_of_domain"]["encoded_value"] == 1
    assert encode(domain_age_months=None)["features"]["age_of_domain"]["mapping_status"] == "mapping_unverified"
    assert encode(hostname_is_ip=True)["features"]["having_IP_Address"]["encoded_value"] == -1
    assert encode(hostname_is_ip=False)["features"]["having_IP_Address"]["encoded_value"] == 1


def test_ssl_and_traffic_remain_unresolved():
    result = encode()
    assert result["features"]["SSLfinal_State"]["mapping_status"] == "mapping_unverified"
    assert result["features"]["web_traffic"]["mapping_status"] == "mapping_unverified"
    assert result["complete"] is False


def test_complete_live_encoding_reaches_existing_classifier_path(monkeypatch):
    application = create_app("testing")
    service = application.extensions["prediction_service"]
    calls = {"count": 0}
    encoded = {
        name: {
            "raw_value": 1,
            "encoded_value": 1 if name in {"age_of_domain", "having_IP_Address"} else 0,
            "mapping_status": "verified",
            "explanation": "test fixture",
        }
        for name in FEATURE_COLUMNS
    }

    monkeypatch.setattr(
        service,
        "analyze_url",
        lambda url: {"url": url, "fixture": True},
    )
    monkeypatch.setattr(
        service,
        "encode_live_observations",
        lambda observations: {
            "features": encoded,
            "complete": True,
            "unresolved_features": [],
        },
    )

    def explain(features):
        calls["count"] += 1
        return 1, features, {"matching_rules": [], "matching_rule_count": 0}

    monkeypatch.setattr(service, "explain_features", explain)
    response = application.test_client().post(
        "/predict", data={"url": "https://example.com/login"}
    )
    assert response.status_code == 200
    assert calls["count"] == 1
