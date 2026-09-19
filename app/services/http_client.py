"""Bounded HTTP retrieval with DNS revalidation and redirect checks."""
from __future__ import annotations
import requests
from .url_security import validate_redirect, validate_url


class SafeHTTPClient:
    def __init__(self, *, timeout=(3.05, 8), max_bytes=1_048_576, max_redirects=3, verify_tls=True):
        self.timeout, self.max_bytes, self.max_redirects, self.verify_tls = timeout, max_bytes, max_redirects, verify_tls

    def get(self, url: str):
        current = validate_url(url)
        session = requests.Session()
        for _ in range(self.max_redirects + 1):
            response = session.get(current, timeout=self.timeout, allow_redirects=False, stream=True, verify=self.verify_tls)
            if response.is_redirect:
                location = response.headers.get("Location")
                response.close()
                if not location:
                    break
                current = validate_redirect(current, location)
                continue
            data = bytearray()
            for chunk in response.iter_content(8192):
                data.extend(chunk)
                if len(data) > self.max_bytes:
                    response.close()
                    raise ValueError("Remote response is too large")
            response._content = bytes(data)
            response.close()
            return response
        raise ValueError("Too many redirects")
