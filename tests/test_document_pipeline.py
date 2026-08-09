from __future__ import annotations

import pytest

import utils
from conftest import make_pdf_bytes, make_pptx_bytes
from utils import (
    CHUNK_OVERLAP_TOKENS,
    MAX_CHUNK_TOKENS,
    MAX_SUMMARY_MAP_CHUNKS,
    MAX_SUMMARY_PAGES,
    MIN_QUERY_TERM_COVERAGE,
    MIN_RELATIVE_BM25_SCORE_RATIO,
    OCR_DPI,
    OCR_MIN_USABLE_CHAR_GAIN,
    OCR_MIN_USABLE_CHAR_RATIO,
    MAX_OCR_PAGES,
    MAX_PPTX_SLIDES,
    MAX_PPTX_TOTAL_TOKENS,
    DocumentProcessingError,
    PptxPreflightError,
    OCRAvailability,
    PageContent,
    TextChunk,
    TokenBudgets,
    build_bm25_index,
    chunk_pages,
    chunk_text,
    estimate_tokens,
    is_ocr_candidate,
    retrieve_evidence,
    select_summary_scope,
    preflight_pptx,
    process_pdf,
    process_ppt,
)


def budgets(**changes):
    values = {"context_tokens": 20_000, "output_reserve": 1_000, "prompt_overhead": 100, "chunk_tokens": 20, "overlap_tokens": 4, "answer_tokens": 500, "safety_margin": 0}
    values.update(changes)
    return TokenBudgets(**values)


def chunk(page: int, index: int, text: str = "evidence") -> TextChunk:
    return TextChunk(f"p{page:04d}-c{index:03d}", page, index, text, "native", len(text.split()))


def test_named_contract_limits_are_stable():
    assert (MAX_CHUNK_TOKENS, CHUNK_OVERLAP_TOKENS) == (1200, 120)
    assert (MAX_SUMMARY_PAGES, MAX_SUMMARY_MAP_CHUNKS) == (40, 80)
    assert (MIN_RELATIVE_BM25_SCORE_RATIO, MIN_QUERY_TERM_COVERAGE, OCR_DPI) == (0.35, 0.50, 300)
    assert (OCR_MIN_USABLE_CHAR_GAIN, OCR_MIN_USABLE_CHAR_RATIO) == (50, 1.25)


def test_chunk_pages_are_token_bounded_deterministic_and_page_scoped():
    local_budgets = budgets(chunk_tokens=3, overlap_tokens=1)
    pages = [PageContent(3, "a b c d e f", "native", 6)]

    chunks = chunk_pages(pages, local_budgets)

    assert [value.chunk_id for value in chunks] == ["p0003-c001", "p0003-c002", "p0003-c003"]
    assert [value.page_number for value in chunks] == [3, 3, 3]
    assert all(value.estimated_tokens <= 3 for value in chunks)
    assert chunks[0].text.split()[-1] == chunks[1].text.split()[0]


def test_chunking_rejects_invalid_overlap_and_uses_conservative_encoding_estimate():
    with pytest.raises(ValueError, match="smaller"):
        TokenBudgets(chunk_tokens=10, overlap_tokens=10)

    value = estimate_tokens("one two three", budgets(estimator_encodings=("missing-tokenizer", "also-missing")))
    assert value >= 3
    assert chunk_text("", budgets=budgets()) == []


def test_scope_reports_exact_full_partial_and_omitted_pages_at_chunk_cap():
    chunks = tuple(chunk(1, index) for index in range(1, 3)) + tuple(chunk(2, index) for index in range(1, 4)) + (chunk(3, 1),)

    scope = select_summary_scope(chunks, max_pages=40, max_chunks=4)

    assert [value.chunk_id for value in scope.selected_chunks] == ["p0001-c001", "p0001-c002", "p0002-c001", "p0002-c002"]
    assert scope.fully_included_pages == frozenset({1})
    assert scope.partially_included_pages == frozenset({2})
    assert scope.omitted_pages == frozenset({3})


def test_scope_reports_pages_beyond_physical_page_cap_as_omitted():
    scope = select_summary_scope(tuple(chunk(page, 1) for page in range(1, 43)), max_pages=40, max_chunks=80)

    assert scope.fully_included_pages == frozenset(range(1, 41))
    assert scope.partially_included_pages == frozenset()
    assert scope.omitted_pages == frozenset({41, 42})


def test_ocr_candidate_requires_raster_signal_and_poor_native_text():
    assert is_ocr_candidate("", has_images=True)
    assert is_ocr_candidate("\ufffd" * 50, has_images=False, image_coverage=0.75)
    assert not is_ocr_candidate("", has_images=False)
    assert not is_ocr_candidate("A useful native text page with enough readable words." * 3, has_images=True)


def test_bm25_retrieves_direct_and_paraphrased_support_without_absolute_cutoff():
    chunks = (
        chunk(1, 1, "The project start and launch date is 14 October and the owner is Ada."),
        chunk(2, 1, "The budget has a contingency reserve for delivery risks."),
        chunk(3, 1, "Unrelated cafeteria menu and office parking information."),
    )
    index = build_bm25_index(chunks)

    direct = retrieve_evidence("What is the project launch date?", index, budgets())
    paraphrase = retrieve_evidence("Which date is the project scheduled to start?", index, budgets())

    assert direct.sufficient and direct.evidence_chunk_ids == ("p0001-c001",)
    assert paraphrase.sufficient and paraphrase.evidence_chunk_ids == ("p0001-c001",)


def test_bm25_refuses_irrelevant_or_contentless_questions_before_generation():
    index = build_bm25_index((chunk(1, 1, "Ada owns the October launch timeline."),))

    irrelevant = retrieve_evidence("What is the cafeteria menu?", index, budgets())
    contentless = retrieve_evidence("What is it?", index, budgets())

    assert not irrelevant.sufficient
    assert irrelevant.chunks == ()
    assert not contentless.sufficient
    assert contentless.refusal_reason == "Question has no document-searchable terms."


def test_process_pdf_preserves_physical_order_and_emits_extract_progress():
    events = []
    document = process_pdf(make_pdf_bytes(["First physical page", "Second physical page"]), "../unsafe name.pdf", budgets(), events.append)

    assert document.display_name == "../unsafe name.pdf"
    assert [page.page_number for page in document.pages] == [1, 2]
    assert [page.text for page in document.pages] == ["First physical page", "Second physical page"]
    assert [event.phase for event in events] == ["extract", "extract"]
    assert [event.page_number for event in events] == [1, 2]


def test_process_pdf_rejects_bad_bytes_without_treating_names_as_paths():
    with pytest.raises(DocumentProcessingError, match="valid, unprotected"):
        process_pdf(b"not a pdf", "../../not-opened.pdf", budgets())


def test_process_pdf_caps_ocr_candidates_in_physical_order(monkeypatch):
    # PDFs built by the fixture are digital.  Isolate cap mechanics by making each
    # extracted page an OCR candidate and recording the bounded local OCR calls.
    pages = [f"page {number}" for number in range(1, MAX_OCR_PAGES + 3)]
    attempted = []
    events = []
    monkeypatch.setattr(utils, "is_ocr_candidate", lambda *args: True)
    monkeypatch.setattr(utils, "detect_english_ocr", lambda *args: OCRAvailability(True, "/tmp/tessdata"))
    monkeypatch.setattr(utils, "_ocr_page_text", lambda page, path: attempted.append(page.number + 1) or "x" * 100)

    document = process_pdf(make_pdf_bytes(pages), "scans.pdf", budgets(), events.append)

    assert attempted == list(range(1, MAX_OCR_PAGES + 1))
    assert [event.page_number for event in events if event.phase == "ocr"] == attempted
    assert any("OCR cap reached: page 11" in warning for warning in document.warnings)
    assert all(page.extraction_method == "ocr" for page in document.pages[:MAX_OCR_PAGES])


def test_detect_english_ocr_and_ocr_page_failure_leave_native_content(monkeypatch):
    monkeypatch.setattr(utils, "is_ocr_candidate", lambda *args: True)
    monkeypatch.setattr(utils, "detect_english_ocr", lambda *args: OCRAvailability(False, None, "OCR unavailable for test."))

    unavailable = process_pdf(make_pdf_bytes(["native words"]), "mixed.pdf", budgets())
    assert unavailable.ocr_status == "unavailable"
    assert unavailable.pages[0].text == "native words"
    assert "OCR unavailable for test." in unavailable.pages[0].warnings

    monkeypatch.setattr(utils, "detect_english_ocr", lambda *args: OCRAvailability(True, "/tmp/tessdata"))
    monkeypatch.setattr(utils, "_ocr_page_text", lambda *args: (_ for _ in ()).throw(RuntimeError("local OCR error")))
    failed = process_pdf(make_pdf_bytes(["native words"]), "mixed.pdf", budgets())
    assert failed.pages[0].text == "native words"
    assert any("OCR failed for page 1" in warning for warning in failed.pages[0].warnings)


def test_process_ppt_and_preflight_enforce_slide_and_token_caps():
    accepted = make_pptx_bytes(["First slide", "Second slide"])
    assert process_ppt(accepted, budgets()) == ["First slide", "Second slide"]
    assert preflight_pptx(accepted, budgets()).slide_count == 2

    too_many_slides = make_pptx_bytes(["slide"] * (MAX_PPTX_SLIDES + 1))
    with pytest.raises(PptxPreflightError, match="split it"):
        preflight_pptx(too_many_slides, budgets())

    # A tiny conservative estimator lets this stay fast while exercising the exact
    # local preflight path instead of ever reaching a generation callable.
    token_heavy = make_pptx_bytes(["a" * (MAX_PPTX_TOTAL_TOKENS * 3 + 1)])
    with pytest.raises(PptxPreflightError, match="shorten or split"):
        preflight_pptx(token_heavy, budgets(estimator_encodings=("missing",)))


def test_tiktoken_network_or_cache_failure_uses_conservative_local_fallback(monkeypatch):
    class BrokenTikToken:
        @staticmethod
        def get_encoding(name):
            raise OSError("network/cache unavailable")

    monkeypatch.setitem(__import__("sys").modules, "tiktoken", BrokenTikToken())
    assert estimate_tokens("one two three", budgets()) >= 3


def test_bm25_term_coverage_is_based_on_unique_content_terms():
    index = build_bm25_index((
        chunk(1, 1, "The release schedule is October."),
        chunk(2, 1, "The facility closes in December."),
    ))

    result = retrieve_evidence("release release schedule facility", index, budgets(), relative_ratio=0.01)

    assert result.sufficient
    assert result.covered_terms == frozenset({"release", "schedule", "facility"})
    assert len(result.chunks) <= 6
