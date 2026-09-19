"""Optional, timeout-bounded domain registration information."""
from __future__ import annotations

import logging
import socket
from datetime import date, datetime, timezone
from typing import Any

try:
    import whois
except ImportError:  # pragma: no cover - exercised when the optional package is absent
    whois = None

import tldextract

log = logging.getLogger(__name__)
_DEFAULT_CLIENT = object()


class DomainInformationService:
    """Retrieve registration age without turning provider failure into a guess."""

    def __init__(self, *, timeout: float = 3.0, now=None, client=_DEFAULT_CLIENT):
        self.timeout = timeout
        self.now = now or (lambda: datetime.now(timezone.utc))
        self.client = (whois.whois if whois else None) if client is _DEFAULT_CLIENT else client
        self._extract = tldextract.TLDExtract(suffix_list_urls=())

    def lookup(self, hostname: str) -> dict[str, Any]:
        domain = self.registrable_domain(hostname)
        if not domain:
            return self._unavailable(hostname, "registrable_domain_unavailable")
        if self.client is None:
            return self._unavailable(domain, "provider_unavailable")

        previous_timeout = socket.getdefaulttimeout()
        try:
            socket.setdefaulttimeout(self.timeout)
            record = self.client(domain)
            creation_date = self._creation_date(record)
            if creation_date is None:
                return self._unavailable(domain, "creation_date_unavailable")
            now = self._as_utc(self.now())
            age_days = max(0, (now - creation_date).total_seconds() / 86400)
            return {
                "domain": domain,
                "creation_date": creation_date.isoformat(),
                "age_days": age_days,
                "age_months": age_days / (365.2425 / 12),
                "status": "available",
            }
        except Exception as exc:
            log.info("Domain registration lookup unavailable for %s: %s", domain, type(exc).__name__)
            return self._unavailable(domain, "provider_unavailable")
        finally:
            socket.setdefaulttimeout(previous_timeout)

    def registrable_domain(self, hostname: str) -> str | None:
        value = (hostname or "").strip().rstrip(".").lower()
        extracted = self._extract(value)
        if not extracted.domain or not extracted.suffix:
            return None
        return f"{extracted.domain}.{extracted.suffix}"

    @staticmethod
    def _creation_date(record: Any) -> datetime | None:
        if isinstance(record, dict):
            values = record.get("creation_date")
        else:
            values = record if isinstance(record, (list, tuple)) else getattr(record, "creation_date", None)
        if values is None:
            return None
        if not isinstance(values, (list, tuple)):
            values = [values]
        parsed = []
        for value in values:
            if isinstance(value, datetime):
                parsed.append(value)
            elif isinstance(value, date):
                parsed.append(datetime.combine(value, datetime.min.time()))
            elif isinstance(value, str):
                try:
                    parsed.append(datetime.fromisoformat(value.replace("Z", "+00:00")))
                except ValueError:
                    continue
        if not parsed:
            return None
        return min(DomainInformationService._as_utc(value) for value in parsed)

    @staticmethod
    def _as_utc(value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    @staticmethod
    def _unavailable(domain: str | None, reason: str) -> dict[str, Any]:
        return {
            "domain": domain,
            "creation_date": None,
            "age_days": None,
            "age_months": None,
            "status": reason,
        }
