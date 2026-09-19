"""Evidence-backed, network-free encoding for live Dataset 379 observations."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from urllib.parse import urljoin, urlparse

from app.ml.preprocessing import FEATURE_COLUMNS, FEATURE_ALLOWED_VALUES


class Dataset379Encoder:
    """Encode only observations with explicitly supported published rules."""

    def encode(self, observations: Mapping[str, object]) -> dict[str, object]:
        url = str(observations.get("url") or "")
        parsed = urlparse(url)
        page_host = (parsed.hostname or "").lower().rstrip(".")
        features = {
            "URL_Length": self._url_length(url),
            "Request_URL": self._request_url(observations, page_host),
            "URL_of_Anchor": self._url_of_anchor(observations, page_host),
            "SFH": self._sfh(observations, page_host),
            "popUpWidnow": self._popup(observations),
            "age_of_domain": self._age(observations),
            "having_IP_Address": self._ip_address(observations),
            "SSLfinal_State": self._unresolved(
                observations.get("ssl_state"),
                "Historical SSL state mapping is not established by the available evidence.",
            ),
            "web_traffic": self._unresolved(
                observations.get("web_traffic"),
                "The historical traffic provider and cutoff are not established.",
            ),
        }
        complete = all(
            item["mapping_status"] == "verified"
            and item["encoded_value"] in FEATURE_ALLOWED_VALUES[name]
            for name, item in features.items()
        )
        return {
            "features": {name: features[name] for name in FEATURE_COLUMNS},
            "complete": complete,
            "unresolved_features": [
                name for name, item in features.items()
                if item["mapping_status"] != "verified"
            ],
        }

    @staticmethod
    def _verified(raw_value: object, value: int, explanation: str) -> dict[str, object]:
        return {
            "raw_value": raw_value,
            "encoded_value": value,
            "mapping_status": "verified",
            "explanation": explanation,
        }

    @staticmethod
    def _unresolved(raw_value: object, explanation: str) -> dict[str, object]:
        return {
            "raw_value": raw_value,
            "encoded_value": None,
            "mapping_status": "mapping_unverified",
            "explanation": explanation,
        }

    def _url_length(self, url: str) -> dict[str, object]:
        length = len(url)
        value = 1 if length < 54 else 0 if length <= 75 else -1
        return self._verified(
            length,
            value,
            "URL length: <54 is 1, 54-75 is 0, and >75 is -1.",
        )

    def _request_url(self, observations: Mapping[str, object], page_host: str) -> dict[str, object]:
        resources = observations.get("resource_urls")
        if not isinstance(resources, Sequence) or isinstance(resources, (str, bytes)):
            return self._unresolved(resources, "Resource URLs were not collected.")
        relevant = [item for item in resources if isinstance(item, Mapping)]
        if not relevant:
            percentage = 0.0
        else:
            external = sum(
                self._is_external(str(item.get("url") or ""), page_host)
                for item in relevant
            )
            percentage = external * 100 / len(relevant)
        value = 1 if percentage < 22 else 0 if percentage <= 61 else -1
        return self._verified(
            percentage,
            value,
            "External image, script, stylesheet, video, and audio URLs divided "
            "by all collected resource URLs; <22 is 1, 22-61 is 0, and >61 is -1.",
        )

    def _url_of_anchor(self, observations: Mapping[str, object], page_host: str) -> dict[str, object]:
        anchors = observations.get("anchor_hrefs")
        if not isinstance(anchors, Sequence) or isinstance(anchors, (str, bytes)):
            return self._unresolved(anchors, "Anchor URLs were not collected.")
        targets = [str(item) for item in anchors]
        if not targets:
            percentage = 0.0
        else:
            external = sum(
                self._is_external(urljoin(str(observations.get("url") or ""), target), page_host)
                for target in targets
                if target.strip() and not target.strip().lower().startswith("javascript:")
            )
            percentage = external * 100 / len(targets)
        value = 1 if percentage < 31 else 0 if percentage <= 67 else -1
        return self._verified(
            percentage,
            value,
            "External anchor targets divided by all anchors; empty and "
            "javascript targets are retained in the denominator and are not external.",
        )

    def _sfh(self, observations: Mapping[str, object], page_host: str) -> dict[str, object]:
        actions = observations.get("form_actions")
        if not isinstance(actions, Sequence) or isinstance(actions, (str, bytes)):
            return self._unresolved(actions, "Form actions were not collected.")
        if not actions:
            return self._verified(
                [],
                -1,
                "No form handler was observed and is treated as an empty handler (-1).",
            )
        values = [str(action or "").strip() for action in actions]
        if any(not value or value.lower() == "about:blank" for value in values):
            return self._verified(values, -1, "Empty or about:blank form handlers map to -1.")
        resolved = [urljoin(str(observations.get("url") or ""), value) for value in values]
        if any(self._is_external(value, page_host) for value in resolved):
            return self._verified(values, 0, "A form handler on a different host maps to 0.")
        return self._verified(values, 1, "Form handlers remaining on the page host map to 1.")

    def _popup(self, observations: Mapping[str, object]) -> dict[str, object]:
        raw = observations.get("popup_credential_fields")
        if raw is None:
            return self._unresolved(raw, "Popup credential-field evidence was not collected.")
        value = -1 if bool(raw) else 1
        return self._verified(
            raw,
            value,
            "Popup behavior maps to -1 only when credential/input fields are present; "
            "otherwise it maps to 1.",
        )

    def _age(self, observations: Mapping[str, object]) -> dict[str, object]:
        months = observations.get("domain_age_months")
        if not isinstance(months, (int, float)) or isinstance(months, bool):
            return self._unresolved(
                months,
                "Reliable domain-registration age was not supplied by a trusted source.",
            )
        value = 1 if months >= 6 else -1
        return self._verified(
            months,
            value,
            "Trusted registration age >=6 months maps to 1; less than 6 months maps to -1.",
        )

    def _ip_address(self, observations: Mapping[str, object]) -> dict[str, object]:
        raw = observations.get("hostname_is_ip")
        if not isinstance(raw, bool):
            return self._unresolved(raw, "Hostname IP status was not collected.")
        return self._verified(
            raw,
            -1 if raw else 1,
            "An IP-literal hostname maps to -1; a normal hostname maps to 1.",
        )

    @staticmethod
    def _is_external(value: str, page_host: str) -> bool:
        parsed = urlparse(value)
        host = (parsed.hostname or "").lower().rstrip(".")
        return bool(host and host != page_host and not host.endswith("." + page_host))
