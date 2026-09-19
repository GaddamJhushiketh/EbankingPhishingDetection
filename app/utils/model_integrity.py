"""Integrity checked loading for the persisted classifier artifact."""
from __future__ import annotations

import hashlib
import hmac
import os
import warnings
from pathlib import Path
from typing import Callable, TypeVar

T = TypeVar("T")


class ModelIntegrityError(RuntimeError):
    """Raised when the configured model cannot be trusted."""


def calculate_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


sha256_file = calculate_sha256


def verify_model_hash(path: str | Path, expected_hash: str | None) -> bool:
    """Verify a model hash using constant-time comparison."""
    if not expected_hash:
        return False
    actual = calculate_sha256(path)
    return hmac.compare_digest(actual.lower(), expected_hash.strip().lower())


verify_expected_hash = verify_model_hash


def load_verified_model(
    path: str | Path,
    loader: Callable[[str | Path], T],
    *,
    expected_hash: str | None = None,
    production: bool = False,
) -> T:
    """Load only an application-controlled artifact with an accepted hash.

    Development may explicitly omit MODEL_SHA256 (a warning is emitted by the
    service), but production always requires it.
    """
    artifact = Path(path).resolve()
    if not artifact.is_file():
        raise ModelIntegrityError("Model artifact is missing")
    configured = expected_hash if expected_hash is not None else os.getenv("MODEL_SHA256")
    if not configured and production:
        raise ModelIntegrityError("MODEL_SHA256 is required in production")
    if not configured:
        warnings.warn(
            "MODEL_SHA256 is not configured; development model loading is unverified",
            RuntimeWarning,
            stacklevel=2,
        )
    if configured and not verify_model_hash(artifact, configured):
        raise ModelIntegrityError("Model artifact integrity check failed")
    return loader(artifact)
