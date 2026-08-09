"""Pure upload and session-state helpers for the Streamlit adapter.

The functions in this module neither import Streamlit nor retain document bytes or
credentials.  That makes document replacement and rerun reuse deterministic and
unit-testable.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import PurePath
from typing import Any, Iterable, Mapping


MAX_UPLOAD_BYTES = 50 * 1024 * 1024


@dataclass(frozen=True)
class UploadValidation:
    accepted: bool
    error: str | None = None


@dataclass(frozen=True)
class UploadIdentity:
    document_id: str
    display_name: str
    size_bytes: int
    file_type: str


@dataclass(frozen=True)
class StateTransition:
    active_document: UploadIdentity | None
    document_changed: bool
    extraction_invalidated: bool
    generated_results_invalidated: bool


@dataclass(frozen=True)
class SettingsTransition:
    settings_changed: bool
    generated_results_invalidated: bool


def _display_name(name: str | None) -> str:
    """Return a display-only basename; never use names as filesystem paths."""
    value = str(name or "document").replace("\\", "/")
    return PurePath(value).name or "document"


def validate_upload(name: str | None, size_bytes: int, max_bytes: int = MAX_UPLOAD_BYTES) -> UploadValidation:
    """Validate file metadata before parsing or model work."""
    if not isinstance(size_bytes, int) or isinstance(size_bytes, bool) or size_bytes < 0:
        return UploadValidation(False, "The uploaded file has an invalid size.")
    if size_bytes > max_bytes:
        return UploadValidation(False, f"This file is larger than the {max_bytes // (1024 * 1024)} MB maximum.")
    suffix = PurePath(_display_name(name)).suffix.lower()
    if suffix not in {".pdf", ".pptx"}:
        return UploadValidation(False, "Please upload a PDF or PPTX file.")
    return UploadValidation(True)


def upload_identity(data: bytes | bytearray | memoryview, display_name: str | None) -> UploadIdentity:
    """Create a content identity; names are only retained for display."""
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError("Upload data must be bytes-like.")
    raw = bytes(data)
    name = _display_name(display_name)
    suffix = PurePath(name).suffix.lower().lstrip(".")
    return UploadIdentity(sha256(raw).hexdigest(), name, len(raw), suffix)


def _previous_document_id(previous: Any) -> str | None:
    if previous is None:
        return None
    if isinstance(previous, UploadIdentity):
        return previous.document_id
    if isinstance(previous, Mapping):
        active = previous.get("active_document", previous.get("document"))
        if isinstance(active, UploadIdentity):
            return active.document_id
        if isinstance(active, Mapping):
            value = active.get("document_id")
            return value if isinstance(value, str) else None
        value = previous.get("document_id")
        return value if isinstance(value, str) else None
    active = getattr(previous, "active_document", previous)
    value = getattr(active, "document_id", None)
    return value if isinstance(value, str) else None


def transition_document_state(previous: Any, upload: UploadIdentity) -> StateTransition:
    """Describe how an incoming upload affects local and generated session data."""
    changed = _previous_document_id(previous) != upload.document_id
    return StateTransition(
        active_document=upload,
        document_changed=changed,
        extraction_invalidated=changed,
        generated_results_invalidated=changed,
    )


def settings_fingerprint(provider_settings: Mapping[str, Any]) -> str:
    """Fingerprint provider/budget settings while excluding credential-like values."""
    def clean(value: Any, key: str = "") -> Any:
        lowered = key.lower()
        # Token *budgets* are safe settings, unlike credentials.  Match only
        # credential-shaped field names so cache fingerprints retain budget changes.
        sensitive_names = {"api_key", "secret", "password", "credential", "authorization", "access_token", "refresh_token"}
        if lowered in sensitive_names or lowered.endswith(("_api_key", "_secret", "_password", "_credential")):
            return "[redacted]"
        if isinstance(value, Mapping):
            return {str(child_key): clean(child_value, str(child_key)) for child_key, child_value in sorted(value.items(), key=lambda item: str(item[0]))}
        if isinstance(value, (list, tuple, frozenset, set)):
            return [clean(item) for item in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return repr(type(value).__name__)

    encoded = json.dumps(clean(provider_settings), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return sha256(encoded.encode("utf-8")).hexdigest()


def transition_generation_settings(previous_fingerprint: str | None, provider_settings: Mapping[str, Any]) -> SettingsTransition:
    current = settings_fingerprint(provider_settings)
    changed = previous_fingerprint != current
    return SettingsTransition(changed, changed)


def generation_fingerprint(
    document_id: str,
    operation: str,
    input_text: str,
    provider_settings: Mapping[str, Any],
    prompt_version: str,
    regeneration_nonce: int = 0,
) -> str:
    """Create a key-free cache key for a summary, answer, or PPTX operation."""
    payload = {
        "document_id": document_id,
        "operation": operation,
        "input_text": input_text,
        "settings": settings_fingerprint(provider_settings),
        "prompt_version": prompt_version,
        "regeneration_nonce": regeneration_nonce,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return sha256(encoded.encode("utf-8")).hexdigest()


def next_regeneration_nonce(current: int | None) -> int:
    """Return a monotonically increasing nonce for an explicit regenerate action."""
    return (current or 0) + 1


def format_page_ranges(pages: Iterable[int]) -> str:
    """Compact a physical-page set for display without losing the exact set in state."""
    ordered = sorted({page for page in pages if isinstance(page, int) and not isinstance(page, bool) and page > 0})
    if not ordered:
        return "None"
    ranges: list[str] = []
    start = previous = ordered[0]
    for page in ordered[1:]:
        if page == previous + 1:
            previous = page
            continue
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = page
    ranges.append(str(start) if start == previous else f"{start}-{previous}")
    return ", ".join(ranges)


def parse_page_ranges(text: str) -> frozenset[int]:
    """Parse the format emitted by :func:`format_page_ranges` for round-trip tests."""
    if not text or text.strip().lower() == "none":
        return frozenset()
    pages: set[int] = set()
    for part in text.split(","):
        value = part.strip()
        if not value:
            continue
        bounds = value.split("-", 1)
        try:
            start = int(bounds[0].strip())
            end = int(bounds[-1].strip())
        except ValueError as exc:
            raise ValueError("Page ranges must contain positive whole numbers.") from exc
        if start <= 0 or end < start:
            raise ValueError("Page ranges must be ascending positive values.")
        pages.update(range(start, end + 1))
    return frozenset(pages)
