"""Strict URL validation and bounded SSRF-safe HTTP access."""
from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urljoin, urlparse, urlunparse


class URLValidationError(ValueError):
    pass


def _blocked_ip(address: str) -> bool:
    ip = ipaddress.ip_address(address)
    return ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast or ip.is_unspecified


def validate_url(url: str) -> str:
    if not isinstance(url, str) or len(url) > 2048:
        raise URLValidationError("Enter a valid URL")
    value = url.strip()
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
        raise URLValidationError("Only public HTTP(S) URLs are accepted")
    hostname = parsed.hostname.rstrip(".").lower()
    if hostname in {"localhost", "localhost.localdomain"} or hostname.endswith(".local") or hostname.endswith(".internal"):
        raise URLValidationError("Internal hostnames are not allowed")
    try:
        literal = ipaddress.ip_address(hostname)
    except ValueError:
        literal = None
    if literal and _blocked_ip(str(literal)):
        raise URLValidationError("Private or internal addresses are not allowed")
    try:
        addresses = {item[4][0] for item in socket.getaddrinfo(hostname, parsed.port or (443 if parsed.scheme == "https" else 80), type=socket.SOCK_STREAM)}
    except (OSError, ValueError):
        raise URLValidationError("Unable to resolve hostname")
    if not addresses or any(_blocked_ip(address) for address in addresses):
        raise URLValidationError("Host resolves to a private or internal address")
    return value


def validate_redirect(original: str, location: str) -> str:
    target = urljoin(original, location)
    return validate_url(target)


def redact_url(url: str) -> str:
    """Return a history-safe URL without credentials, query, or fragment."""

    parsed = urlparse(url)
    return urlunparse(
        (
            parsed.scheme,
            parsed.hostname or "",
            parsed.path or "/",
            "",
            "",
            "",
        )
    )
