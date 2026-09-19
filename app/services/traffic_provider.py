"""Optional provider boundary for the historical Dataset 379 traffic signal."""
from __future__ import annotations


class TrafficProvider:
    """Provider interface; no modern ranking service is treated as Dataset 379."""

    def get_rank(self, domain: str) -> dict[str, object]:
        return {
            "domain": domain,
            "rank": None,
            "provider": None,
            "status": "provider_unavailable",
        }
