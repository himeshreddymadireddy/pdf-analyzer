from __future__ import annotations

import json

from utils import (
    REFUSAL_TEXT,
    AnswerResult,
    GroundedGenerationRequest,
    RetrievalResult,
    SummaryScope,
    TextChunk,
    TokenBudgets,
    answer_pdf_question,
    summarize_document,
)


def budgets(**changes):
    values = {
        "context_tokens": 20_000,
        "output_reserve": 500,
        "prompt_overhead": 100,
        "chunk_tokens": 100,
        "overlap_tokens": 10,
        "reduce_tokens": 500,
        "answer_tokens": 500,
        "safety_margin": 0,
    }
    values.update(changes)
    return TokenBudgets(**values)


def chunk(page: int, index: int = 1, *, method: str = "native") -> TextChunk:
    return TextChunk(f"p{page:04d}-c{index:03d}", page, index, f"Evidence from page {page}, passage {index}.", method, 8)


def scope_for(*chunks: TextChunk, fully: set[int] | None = None, partial: set[int] | None = None) -> SummaryScope:
    pages = {value.page_number for value in chunks}
    return SummaryScope(tuple(chunks), frozenset(pages if fully is None else fully), frozenset(partial or set()), frozenset())


def block(source_id: str, text: str = "Supported claim") -> str:
    return json.dumps({"blocks": [{"text": text, "source_ids": [source_id]}]})


def answer_block(source_id: str, text: str = "Supported answer") -> str:
    return json.dumps({"insufficient_evidence": False, "blocks": [{"text": text, "source_ids": [source_id]}]})


def test_summary_maps_then_reduces_strictly_sequentially_and_reports_progress():
    first, second = chunk(1), chunk(2)
    calls: list[GroundedGenerationRequest] = []
    events = []

    def generate(request):
        calls.append(request)
        if len(calls) == 1:
            return block(first.chunk_id, "Page one")
        if len(calls) == 2:
            return block(second.chunk_id, "Page two")
        return json.dumps({"blocks": [{"text": "Combined", "source_ids": [first.chunk_id, second.chunk_id]}]})

    result = summarize_document(scope_for(first, second), budgets(), generate, progress=events.append)

    assert [event.phase for event in events] == ["map", "map", "reduce"]
    assert len(calls) == 3
    assert all(request.grounded for request in calls)
    assert first.chunk_id in calls[0].user_text and second.chunk_id not in calls[0].user_text
    assert second.chunk_id in calls[1].user_text and first.chunk_id not in calls[1].user_text
    assert first.chunk_id in calls[2].user_text and second.chunk_id in calls[2].user_text
    assert result.succeeded
    assert result.blocks[0].source_ids == (first.chunk_id, second.chunk_id)
    assert result.successful_chunk_ratio == 1.0


def test_summary_fails_closed_below_eighty_percent_and_never_reduces():
    chunks = tuple(chunk(page) for page in range(1, 25))
    calls = []

    def generate(request):
        calls.append(request)
        source_id = f"p{len(calls):04d}-c001"
        if len(calls) <= 19:
            return block(source_id)
        return "not json"

    result = summarize_document(scope_for(*chunks), budgets(), generate)

    assert not result.succeeded
    assert result.blocks == ()
    assert result.successful_chunk_ratio == 19 / 24
    assert len(calls) == 24  # Maps only: reduce starts only after coverage passes.


def test_summary_accepts_exactly_eighty_percent_and_allows_partial_page_gap():
    first, second, third, fourth, partial = chunk(1, 1), chunk(1, 2), chunk(2), chunk(3), chunk(4)
    current = 0

    def generate(request):
        nonlocal current
        current += 1
        if current == 5:
            return "not json"
        if current <= 4:
            return block((first, second, third, fourth)[current - 1].chunk_id)
        return json.dumps({"blocks": [{"text": "Combined", "source_ids": [first.chunk_id, second.chunk_id, third.chunk_id, fourth.chunk_id]}]})

    result = summarize_document(
        scope_for(first, second, third, fourth, partial, fully={1, 2, 3}, partial={4}),
        budgets(),
        generate,
    )

    assert result.succeeded
    assert result.successful_chunk_ratio == 0.8
    assert result.uncovered_pages == frozenset()
    assert result.scope.partially_included_pages == frozenset({4})


def test_summary_rejects_fully_included_page_with_no_valid_map_output():
    first, second = chunk(1), chunk(2)

    def generate(request):
        if first.chunk_id in request.user_text:
            return "bad json"
        return block(second.chunk_id)

    result = summarize_document(scope_for(first, second), budgets(), generate)

    assert not result.succeeded
    assert result.uncovered_pages == frozenset({1})
    assert not any(event for event in result.warnings if "Reduce" in event)


def test_malformed_json_uses_one_same_target_repair_but_semantic_violation_does_not():
    source = chunk(1)
    repair_calls = []

    class Generated:
        text = "not json"
        target = object()
        attempts = ("initial",)

    def repair(malformed, schema, target):
        repair_calls.append((malformed, schema, target))
        return block(source.chunk_id)

    repaired = summarize_document(scope_for(source), budgets(), lambda _: Generated(), repair)

    assert repaired.succeeded
    assert len(repair_calls) == 1
    assert repair_calls[0][0] == "not json"
    assert source.text not in repair_calls[0][0]
    assert repair_calls[0][2] is Generated.target

    semantic_repairs = []
    invalid = summarize_document(
        scope_for(source), budgets(), lambda _: block("unknown-source"), lambda *args: semantic_repairs.append(args)
    )
    assert not invalid.succeeded
    assert semantic_repairs == []


def test_reduce_must_preserve_the_union_of_all_input_source_ids():
    first, second = chunk(1), chunk(2)
    calls = 0

    def generate(request):
        nonlocal calls
        calls += 1
        if calls == 1:
            return block(first.chunk_id)
        if calls == 2:
            return block(second.chunk_id)
        # It is a valid subset citation syntactically, but reduction must be
        # provenance-lossless and therefore rejects this dropped second source.
        return block(first.chunk_id, "Dropped page two")

    result = summarize_document(scope_for(first, second), budgets(), generate)

    assert not result.succeeded
    assert result.blocks == ()
    assert any("Reduce batch 1" in warning for warning in result.warnings)


def test_repair_type_error_is_not_retried_as_an_arity_fallback():
    source = chunk(1)
    calls = []

    def repair(malformed, schema, target):
        calls.append((malformed, schema, target))
        raise TypeError("repair implementation failed")

    result = summarize_document(scope_for(source), budgets(), lambda _: "not json", repair)

    assert not result.succeeded
    assert len(calls) == 1


def test_summary_final_citations_must_resolve_to_active_source_ids():
    source = chunk(1)

    def generate(request):
        if "<source" in request.user_text:
            return block(source.chunk_id)
        return block("not-active")

    # A single map does not require reduction and remains valid.
    result = summarize_document(scope_for(source), budgets(), generate)
    assert result.succeeded
    assert result.blocks[0].source_ids == (source.chunk_id,)


def test_answer_refuses_without_sufficient_retrieval_before_generation():
    retrieval = RetrievalResult((), (), 0.0, frozenset({"missing"}), frozenset(), False, "No relevant document passages were found.")
    calls = []

    result = answer_pdf_question("What is missing?", retrieval, budgets(), lambda request: calls.append(request))

    assert isinstance(result, AnswerResult)
    assert result.refusal == REFUSAL_TEXT
    assert calls == []


def test_answer_is_grounded_cites_only_evidence_and_marks_ocr_provenance():
    native, ocr = chunk(1), chunk(2, method="ocr")
    retrieval = RetrievalResult((native, ocr), (1.0, 0.8), 1.0, frozenset({"evidence"}), frozenset({"evidence"}), True)
    calls = []

    def generate(request):
        calls.append(request)
        return answer_block(ocr.chunk_id)

    result = answer_pdf_question("Which evidence is present?", retrieval, budgets(), generate)

    assert result.refusal is None
    assert result.evidence_chunk_ids == (ocr.chunk_id,)
    assert result.grounded_passage_count == 1
    assert result.cited_pages == (2,)
    assert result.ocr_page_numbers == (2,)
    assert len(calls) == 1
    assert native.chunk_id in calls[0].user_text and ocr.chunk_id in calls[0].user_text
    assert "previous answer" not in calls[0].user_text


def test_answer_refuses_malformed_insufficient_or_unknown_citations():
    source = chunk(1)
    retrieval = RetrievalResult((source,), (1.0,), 1.0, frozenset({"evidence"}), frozenset({"evidence"}), True)

    malformed = answer_pdf_question("Question", retrieval, budgets(), lambda _: "not json")
    insufficient = answer_pdf_question(
        "Question", retrieval, budgets(), lambda _: json.dumps({"insufficient_evidence": True, "blocks": []})
    )
    unknown = answer_pdf_question("Question", retrieval, budgets(), lambda _: answer_block("unknown"))

    assert malformed.refusal == REFUSAL_TEXT
    assert insufficient.refusal == REFUSAL_TEXT
    assert unknown.refusal == REFUSAL_TEXT
