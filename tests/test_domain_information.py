from datetime import datetime, timezone

from app.services.domain_information import DomainInformationService


def fixed_now():
    return datetime(2026, 9, 19, tzinfo=timezone.utc)


def test_registrable_domain_is_extracted_without_network():
    service = DomainInformationService(client=lambda domain: {}, now=fixed_now)
    assert service.registrable_domain("login.example.co.uk") == "example.co.uk"


def test_creation_date_datetime_calculates_age():
    service = DomainInformationService(
        client=lambda domain: {"creation_date": datetime(2026, 3, 19, tzinfo=timezone.utc)},
        now=fixed_now,
    )
    result = service.lookup("example.com")
    assert result["status"] == "available"
    assert result["age_months"] >= 6


def test_creation_date_list_uses_earliest_valid_date_deterministically():
    service = DomainInformationService(
        client=lambda domain: {
            "creation_date": [
                "2026-06-19T00:00:00+00:00",
                datetime(2026, 3, 19, tzinfo=timezone.utc),
            ]
        },
        now=fixed_now,
    )
    result = service.lookup("example.com")
    assert result["creation_date"].startswith("2026-03-19")
    assert result["age_months"] >= 6


def test_missing_or_malformed_creation_date_is_unavailable():
    for record in ({}, {"creation_date": "not-a-date"}):
        service = DomainInformationService(client=lambda domain, record=record: record, now=fixed_now)
        result = service.lookup("example.com")
        assert result["status"] == "creation_date_unavailable"
        assert result["age_months"] is None


def test_whois_failure_is_provider_unavailable():
    def fail(domain):
        raise TimeoutError("timeout")

    result = DomainInformationService(client=fail, now=fixed_now).lookup("example.com")
    assert result["status"] == "provider_unavailable"


def test_missing_whois_client_is_provider_unavailable():
    service = DomainInformationService(client=lambda domain: {}, now=fixed_now)
    service.client = None
    result = service.lookup("example.com")
    assert result["status"] == "provider_unavailable"
