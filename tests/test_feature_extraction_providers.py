from app.services.feature_extraction import FeatureExtractionService


class FakeResponse:
    status_code = 200
    headers = {"Content-Type": "text/html"}
    text = "<html><body><form action='/submit'></form></body></html>"


class FakeHTTPClient:
    def get(self, url):
        return FakeResponse()


class FailingHTTPClient:
    def get(self, url):
        raise TimeoutError("fetch timeout")


def available_domain(hostname):
    return {
        "domain": hostname,
        "creation_date": "2020-01-01T00:00:00+00:00",
        "age_days": 2000,
        "age_months": 65,
        "status": "available",
    }


def unavailable_traffic(domain):
    return {
        "domain": domain,
        "rank": None,
        "provider": None,
        "status": "provider_unavailable",
    }


def test_https_records_tls_observation_without_encoding_ssl():
    observations = FeatureExtractionService(
        http_client=FakeHTTPClient(),
        domain_information=type("Domain", (), {"lookup": staticmethod(available_domain)})(),
        traffic_provider=type("Traffic", (), {"get_rank": staticmethod(unavailable_traffic)})(),
    ).analyze("https://example.com/")
    assert observations["ssl_state"]["tls_verified"] is True
    assert observations["ssl_state"]["observation_status"] == "observation_available"
    assert observations["domain_age_months"] == 65
    assert observations["web_traffic"]["status"] == "provider_unavailable"


def test_http_marks_tls_not_applicable():
    observations = FeatureExtractionService(
        http_client=FakeHTTPClient(),
        domain_information=type("Domain", (), {"lookup": staticmethod(available_domain)})(),
    ).analyze("http://example.com/")
    assert observations["ssl_state"]["observation_status"] == "not_applicable"


def test_tls_fetch_failure_is_an_extraction_error_not_a_prediction():
    observations = FeatureExtractionService(
        http_client=FailingHTTPClient(),
        domain_information=type("Domain", (), {"lookup": staticmethod(available_domain)})(),
    ).analyze("https://example.com/")
    assert observations["ssl_state"]["tls_verified"] is False
    assert observations["ssl_state"]["observation_status"] == "extraction_error"
