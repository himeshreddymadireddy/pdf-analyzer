from __future__ import annotations

from app_state import (
    MAX_UPLOAD_BYTES,
    format_page_ranges,
    generation_fingerprint,
    next_regeneration_nonce,
    parse_page_ranges,
    settings_fingerprint,
    transition_document_state,
    transition_generation_settings,
    upload_identity,
    validate_upload,
)


def test_upload_limit_accepts_exact_boundary_and_rejects_one_byte_over():
    assert validate_upload("report.pdf", MAX_UPLOAD_BYTES).accepted
    rejected = validate_upload("report.pdf", MAX_UPLOAD_BYTES + 1)
    assert not rejected.accepted
    assert "50 MB" in (rejected.error or "")


def test_upload_identity_is_digest_based_and_filename_is_display_only():
    first = upload_identity(b"same bytes", r"../../private/report.pdf")
    duplicate_name_different_bytes = upload_identity(b"different bytes", "report.pdf")
    renamed_same_bytes = upload_identity(b"same bytes", "renamed.pdf")

    assert first.display_name == "report.pdf"
    assert first.document_id != duplicate_name_different_bytes.document_id
    assert first.document_id == renamed_same_bytes.document_id


def test_document_transition_replaces_only_when_content_changes():
    first = upload_identity(b"one", "one.pdf")
    renamed = upload_identity(b"one", "renamed.pdf")
    second = upload_identity(b"two", "two.pdf")

    initial = transition_document_state(None, first)
    unchanged = transition_document_state({"active_document": first}, renamed)
    replacement = transition_document_state({"active_document": first}, second)

    assert initial.document_changed and initial.extraction_invalidated
    assert not unchanged.document_changed
    assert replacement.document_changed and replacement.generated_results_invalidated


def test_settings_change_invalidates_generation_but_never_uses_key_material():
    settings = {"primary_model": "GPT-5.4 Mini", "api_key": "do-not-leak", "fallback_enabled": False}
    first = settings_fingerprint(settings)
    changed = transition_generation_settings(first, {**settings, "primary_model": "DeepSeek V4 Flash"})
    unchanged = transition_generation_settings(first, {**settings, "api_key": "another-secret"})

    assert "do-not-leak" not in first
    assert changed.settings_changed and changed.generated_results_invalidated
    assert not unchanged.settings_changed


def test_generation_fingerprint_reuses_unchanged_work_and_regenerate_nonce_changes_it():
    settings = {"primary_model": "GPT-5.4 Mini", "api_key": "not-in-fingerprint"}
    first = generation_fingerprint("doc", "summary", "", settings, "v2")
    again = generation_fingerprint("doc", "summary", "", {**settings, "api_key": "other"}, "v2")
    regenerated = generation_fingerprint("doc", "summary", "", settings, "v2", next_regeneration_nonce(0))

    assert first == again
    assert first != regenerated


def test_page_ranges_round_trip_exact_page_sets():
    pages = frozenset({1, 2, 3, 7, 9, 10, 11})
    display = format_page_ranges(pages)
    assert display == "1-3, 7, 9-11"
    assert parse_page_ranges(display) == pages
    assert parse_page_ranges("None") == frozenset()
