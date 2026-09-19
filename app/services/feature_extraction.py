"""Safe raw observations and explicitly verified feature-vector handling."""
from __future__ import annotations

import ipaddress
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

from app.ml.preprocessing import FEATURE_COLUMNS, validate_feature_vector
from .http_client import SafeHTTPClient
from .domain_information import DomainInformationService
from .traffic_provider import TrafficProvider
from .url_security import validate_url


class FeatureExtractionError(RuntimeError):
    pass


class FeatureExtractionService:
    """Collect observations without inventing Dataset 379 encodings.

    ``provider`` is reserved for an explicitly verified provider, such as a
    deterministic test fixture. The default path never converts observations
    into model feature values.
    """
    def __init__(
        self,
        *,
        strategy: str = "url_only",
        provider=None,
        http_client=None,
        domain_information=None,
        traffic_provider=None,
    ):
        if strategy not in {"url_only", "provider"}:
            raise ValueError("Unsupported extraction strategy")
        self.strategy, self.provider = strategy, provider
        self.http_client = http_client or SafeHTTPClient()
        self.domain_information = domain_information or DomainInformationService()
        self.traffic_provider = traffic_provider or TrafficProvider()

    def analyze(self, url: str) -> dict[str, object]:
        value = validate_url(url)
        parsed = urlparse(value)
        host = parsed.hostname or ""
        try:
            ipaddress.ip_address(host)
            hostname_is_ip = True
        except ValueError:
            hostname_is_ip = False
        observations: dict[str, object] = {
            "url": value,
            "URL_Length": {"raw_value": len(value)},
            "SSLfinal_State": {"raw_value": {"scheme": parsed.scheme}},
            "having_IP_Address": {"raw_value": hostname_is_ip},
            "hostname_is_ip": hostname_is_ip,
            "resource_urls": [],
            "anchor_hrefs": [],
            "form_actions": [],
            "popup_credential_fields": None,
            "domain_age_months": None,
            "ssl_state": None,
            "web_traffic": None,
        }
        try:
            domain_info = self.domain_information.lookup(host)
        except Exception:
            domain_info = {
                "domain": host,
                "creation_date": None,
                "age_days": None,
                "age_months": None,
                "status": "extraction_error",
            }
        observations["domain_information"] = domain_info
        if domain_info.get("status") == "available":
            observations["domain_age_months"] = domain_info["age_months"]
        else:
            observations["domain_age_status"] = domain_info.get("status")
        try:
            observations["web_traffic"] = self.traffic_provider.get_rank(host)
        except Exception:
            observations["web_traffic"] = {
                "domain": host,
                "rank": None,
                "provider": None,
                "status": "provider_unavailable",
            }
        observations["ssl_state"] = {
            "scheme": parsed.scheme,
            "tls_verified": None,
            "certificate_valid": None,
            "hostname_match": None,
            "expires_at": None,
            "mapping_status": "mapping_unverified",
        }
        try:
            response = self.http_client.get(value)
            html = response.text
            soup = BeautifulSoup(html, "html.parser")
            resources = []
            for element in soup.find_all(["img", "script", "link", "video", "audio"]):
                attribute = "href" if element.name == "link" else "src"
                resource = element.get(attribute)
                if resource:
                    resources.append({"url": urljoin(value, resource), "kind": element.name})
            observations["resource_urls"] = resources
            observations["anchor_hrefs"] = [
                str(anchor.get("href") or "") for anchor in soup.find_all("a")
            ]
            observations["form_actions"] = [
                str(form.get("action") or "") for form in soup.find_all("form")
            ]
            scripts = " ".join(script.get_text(" ", strip=True) for script in soup.find_all("script"))
            has_popup = "window.open" in scripts.lower()
            credential_fields = soup.find_all(
                ["input", "textarea"],
                attrs={"type": lambda value: value and value.lower() in {"password", "email"}},
            )
            observations["popup_credential_fields"] = (
                bool(has_popup and credential_fields) if has_popup else False
            )
            observations["page"] = {
                "status_code": response.status_code,
                "content_type": response.headers.get("Content-Type", ""),
                "forms": len(soup.find_all("form")),
                "anchors": len(soup.find_all("a")),
                "external_reference_candidates": len(resources),
            }
            if parsed.scheme == "https":
                observations["ssl_state"]["tls_verified"] = True
                observations["ssl_state"]["observation_status"] = "observation_available"
            else:
                observations["ssl_state"]["observation_status"] = "not_applicable"
        except Exception:
            observations["page"] = {"status": "unavailable", "reason": "safe fetch failed"}
            if parsed.scheme == "https":
                observations["ssl_state"]["tls_verified"] = False
                observations["ssl_state"]["observation_status"] = "extraction_error"
        for feature in FEATURE_COLUMNS:
            observations.setdefault(feature, {"raw_value": None})
        return observations

    def extract(self, url: str) -> dict[str, int]:
        """Return values only from an explicitly verified provider."""

        value = validate_url(url)
        if self.strategy != "provider" or self.provider is None:
            raise FeatureExtractionError(
                "Dataset 379 feature encoding is unavailable for SFH and other "
                "required features: mapping_unverified"
            )
        try:
            supplied = dict(self.provider(value))
        except Exception as exc:
            raise FeatureExtractionError("Trusted feature provider failed") from exc
        try:
            return validate_feature_vector(supplied)
        except ValueError as exc:
            raise FeatureExtractionError(str(exc)) from exc
