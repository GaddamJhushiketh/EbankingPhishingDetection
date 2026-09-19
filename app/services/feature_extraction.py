"""Safe raw observations and explicitly verified feature-vector handling."""
from __future__ import annotations

import ipaddress
from urllib.parse import urlparse

from bs4 import BeautifulSoup

from app.ml.preprocessing import FEATURE_COLUMNS, validate_feature_vector
from .http_client import SafeHTTPClient
from .url_security import validate_url


class FeatureExtractionError(RuntimeError):
    pass


class FeatureExtractionService:
    """Collect observations without inventing Dataset 379 encodings.

    ``provider`` is reserved for an explicitly verified provider, such as a
    deterministic test fixture. The default path never converts observations
    into model feature values.
    """
    def __init__(self, *, strategy: str = "url_only", provider=None, http_client=None):
        if strategy not in {"url_only", "provider"}:
            raise ValueError("Unsupported extraction strategy")
        self.strategy, self.provider = strategy, provider
        self.http_client = http_client or SafeHTTPClient()

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
            "URL_Length": {"raw_value": len(value), "encoded_value": None},
            "SSLfinal_State": {
                "raw_value": {"scheme": parsed.scheme},
                "encoded_value": None,
            },
            "having_IP_Address": {
                "raw_value": hostname_is_ip,
                "encoded_value": None,
            },
        }
        try:
            response = self.http_client.get(value)
            html = response.text
            soup = BeautifulSoup(html, "html.parser")
            observations["page"] = {
                "status_code": response.status_code,
                "content_type": response.headers.get("Content-Type", ""),
                "forms": len(soup.find_all("form")),
                "anchors": len(soup.find_all("a")),
                "external_reference_candidates": len(
                    soup.find_all(["img", "script", "link", "video", "audio"])
                ),
            }
        except Exception:
            observations["page"] = {"status": "unavailable", "reason": "safe fetch failed"}
        for feature in FEATURE_COLUMNS:
            observations.setdefault(
                feature,
                {"raw_value": None, "encoded_value": None},
            )
            observations[feature].setdefault("status", "mapping_unverified")
            observations[feature].setdefault(
                "reason",
                "Dataset 379 feature encoding is not verified by available primary evidence",
            )
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
