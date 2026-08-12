"""Streamlit UI adapter for bounded, provider-neutral document analysis."""
from __future__ import annotations

import inspect
import os
from hashlib import sha256
from collections.abc import Mapping
from dataclasses import asdict
from typing import Any

import streamlit as st

from app_state import (
    format_page_ranges,
    generation_fingerprint,
    next_regeneration_nonce,
    settings_fingerprint,
    transition_document_state,
    transition_generation_settings,
    upload_identity,
    validate_upload,
)
from providers import (
    CATALOG_ALIASES,
    CredentialUnavailableError,
    GenerationRequest,
    GenerationResult,
    ProviderError,
    ResolvedGenerationConfig,
    generate_with_fallback,
    repair_malformed_json,
    resolve_api_key,
    resolve_generation_config,
)
from utils import (
    AnswerResult,
    DocumentContent,
    GroundedGenerationRequest,
    PptxPreflightError,
    SummaryResult,
    TokenBudgets,
    answer_pdf_question,
    build_bm25_index,
    chunk_document,
    process_pdf,
    process_ppt,
    retrieve_evidence,
    select_summary_scope,
    summarize_document,
)

PROMPT_VERSION = "v2"
PROVIDER_LABELS = {"anthropic": "Anthropic", "deepseek": "DeepSeek", "openai": "OpenAI"}


def read_streamlit_secrets(secrets: Any | None = None) -> Mapping[str, Any]:
    """Return secrets without letting a missing file render Streamlit's raw alert.

    Streamlit 1.33 exposes a private parser with ``print_exceptions=False``.  If a
    later Streamlit version changes that signature or removes the parser, safely
    treat secrets as unavailable rather than falling back to a public accessor that
    renders a raw configuration error in the application.
    """
    source = st.secrets if secrets is None else secrets
    parser = getattr(source, "_parse", None)
    if not callable(parser):
        return {}
    try:
        parameters = inspect.signature(parser).parameters
    except (TypeError, ValueError):
        return {}
    if "print_exceptions" not in parameters:
        return {}
    try:
        values = parser(print_exceptions=False)
    except FileNotFoundError:
        return {}
    except TypeError:
        # A changed private implementation may reject the former keyword.
        return {}
    return dict(values) if isinstance(values, Mapping) else {}


def _session_overrides() -> dict[str, str]:
    return dict(st.session_state.get("api_key_overrides", {}))


def _credential_label(alias: str, secrets: Mapping[str, Any]) -> str:
    from providers import MODEL_CATALOG

    resolution = resolve_api_key(MODEL_CATALOG[alias].provider_id, _session_overrides(), secrets, os.environ)
    return resolution.source


def _budgets_for_specs(specs: list[Any]) -> TokenBudgets:
    return TokenBudgets(
        context_tokens=min(spec.context_tokens for spec in specs),
        estimator_encodings=tuple(dict.fromkeys(spec.estimator_encoding for spec in specs)),
        output_reserve=min(spec.max_output_tokens for spec in specs),
    )


def token_budgets(config: ResolvedGenerationConfig) -> TokenBudgets:
    """Build conservative primitive budgets from the active primary/fallback targets."""
    specs = [config.primary.spec]
    if config.secondary is not None:
        specs.append(config.secondary.spec)
    return _budgets_for_specs(specs)


def selected_token_budgets(primary_alias: str, secondary_alias: str | None, fallback_enabled: bool, secrets: Mapping[str, Any]) -> TokenBudgets:
    """Budget local work before generation, excluding an unavailable fallback."""
    from providers import MODEL_CATALOG

    specs = [MODEL_CATALOG[primary_alias]]
    if fallback_enabled and secondary_alias and _credential_label(secondary_alias, secrets) != "missing":
        specs.append(MODEL_CATALOG[secondary_alias])
    return _budgets_for_specs(specs)


def _settings_payload(primary_alias: str, secondary_alias: str | None, fallback_enabled: bool, budgets: TokenBudgets | None = None) -> dict[str, Any]:
    overrides = _session_overrides()
    override_fingerprint = {k: sha256(v.encode("utf-8")).hexdigest()[:12] for k, v in sorted(overrides.items()) if v}
    return {
        "primary_model": primary_alias,
        "secondary_model": secondary_alias if fallback_enabled else None,
        "fallback_enabled": fallback_enabled,
        "api_key_overrides": override_fingerprint,
        "budgets": repr(budgets) if budgets else None,
        "prompt_version": PROMPT_VERSION,
    }


def _clear_generated_state() -> None:
    for key in ("summary_results", "answer_history", "pptx_slide_results", "pptx_answers"):
        st.session_state[key] = {}


def _ensure_state() -> None:
    st.session_state.setdefault("api_key_overrides", {})
    st.session_state.setdefault("summary_results", {})
    st.session_state.setdefault("answer_history", {})
    st.session_state.setdefault("pptx_slide_results", {})
    st.session_state.setdefault("pptx_answers", {})
    st.session_state.setdefault("summary_regeneration_nonce", 0)
    st.session_state.setdefault("pdf_pipeline_cache", {})
    st.session_state.setdefault("active_document", None)
    st.session_state.setdefault("document_state", None)


def _sync_session_override(key_name: str, widget_key: str) -> None:
    """Synchronize a password widget before Streamlit locks its state for a run."""
    value = st.session_state.get(widget_key, "")
    overrides = st.session_state.setdefault("api_key_overrides", {})
    if isinstance(value, str) and value.strip():
        overrides[key_name] = value.strip()
    else:
        overrides.pop(key_name, None)


def _clear_session_override(key_name: str, widget_key: str) -> None:
    """Clear override and widget value in an on-click callback, not post-render."""
    st.session_state.setdefault("api_key_overrides", {}).pop(key_name, None)
    st.session_state[widget_key] = ""


def _render_api_key_control(provider_id: str, secrets: Mapping[str, Any]) -> None:
    key_name = {"anthropic": "ANTHROPIC_API_KEY", "deepseek": "DEEPSEEK_API_KEY", "openai": "OPENAI_API_KEY"}[provider_id]
    widget_key = f"override_{key_name}"
    st.text_input(
        "API key",
        type="password",
        key=widget_key,
        placeholder="Session override (optional)",
        help="Stored only in this browser session.",
        on_change=_sync_session_override,
        args=(key_name, widget_key),
    )
    with st.columns((3, 2))[1]:
        st.button(
            "Clear override",
            key=f"clear_{key_name}",
            on_click=_clear_session_override,
            args=(key_name, widget_key),
        )
    resolution = resolve_api_key(provider_id, _session_overrides(), secrets, os.environ)
    st.caption(f"Credential status: **{resolution.source}**")


def _provider_controls(secrets: Mapping[str, Any]) -> tuple[str, str | None, bool]:
    from providers import MODEL_CATALOG

    with st.sidebar:
        st.header("Model settings")
        provider_ids = tuple(PROVIDER_LABELS)
        selected_provider = st.selectbox("Provider", provider_ids, format_func=lambda value: PROVIDER_LABELS[value])
        aliases = [alias for alias in CATALOG_ALIASES if MODEL_CATALOG[alias].provider_id == selected_provider]
        primary_alias = st.selectbox("Model", aliases, key="primary_model")
        _render_api_key_control(selected_provider, secrets)
        fallback_enabled = st.toggle("Use fallback model", value=False, help="Try one distinct model only after retryable primary failures.")
        secondary_alias: str | None = None
        if fallback_enabled:
            secondary_alias = st.selectbox("Fallback model", [alias for alias in CATALOG_ALIASES if alias != primary_alias], key="fallback_model")
            secondary_provider = MODEL_CATALOG[secondary_alias].provider_id
            if secondary_provider != selected_provider:
                _render_api_key_control(secondary_provider, secrets)
            source = _credential_label(secondary_alias, secrets)
            if source == "missing":
                st.warning("Fallback credential is missing; generation will use the primary only.")
        st.caption("Keys are session-only overrides. No key characters are displayed.")
    return primary_alias, secondary_alias, fallback_enabled


def _get_generation_config(primary_alias: str, secondary_alias: str | None, fallback_enabled: bool, secrets: Mapping[str, Any]) -> ResolvedGenerationConfig | None:
    try:
        return resolve_generation_config(
            primary_alias,
            secondary_alias,
            fallback_enabled=fallback_enabled,
            session_overrides=_session_overrides(),
            streamlit_values=secrets,
            environ=os.environ,
        )
    except CredentialUnavailableError:
        st.error("A primary credential is required before generation. Add a session override, Streamlit secret, or environment key.")
        return None
    except ProviderError as exc:
        st.error(f"Configuration error: {exc}")
        return None


def _provider_generate(config: ResolvedGenerationConfig, event_holder: Any):
    def generate(request: GroundedGenerationRequest) -> GenerationResult:
        event_holder.info("Generating sequentially…")
        return generate_with_fallback(
            GenerationRequest(request.system_text, request.user_text, request.max_output_tokens, request.grounded),
            config,
        )

    def repair(malformed_text: str, schema: str, target: object | None) -> str:
        if target is None:
            raise ValueError("A generated target is required for JSON repair.")
        event_holder.info("Repairing one malformed JSON response…")
        return repair_malformed_json(malformed_text, schema, target)

    return generate, repair


def _progress_renderer(holder: Any):
    def render(event: Any) -> None:
        holder.info(event.message)
    return render


def ingest_upload(name: str, data: bytes, *, budgets: TokenBudgets | None = None) -> tuple[object | None, str | None]:
    """Thin, injectable upload seam: validate, identify, and extract locally."""
    validation = validate_upload(name, len(data))
    if not validation.accepted:
        return None, validation.error
    identity = upload_identity(data, name)
    transition = transition_document_state(st.session_state.get("active_document"), identity)
    if not transition.document_changed:
        return st.session_state.get("document_state"), None
    try:
        if identity.file_type == "pdf":
            document = process_pdf(data, identity.display_name, budgets or TokenBudgets())
            prepared = {"identity": identity, "type": "pdf", "document": document}
        else:
            slides = process_ppt(data, budgets or TokenBudgets())
            prepared = {"identity": identity, "type": "pptx", "slides": slides}
    except (PptxPreflightError, ValueError) as exc:
        return None, str(exc)
    st.session_state.active_document = identity
    st.session_state.document_state = prepared
    st.session_state.pdf_pipeline_cache = {}
    _clear_generated_state()
    return prepared, None


def _render_scope(scope: Any) -> None:
    st.caption(
        "Summary scope — fully included pages: " + format_page_ranges(scope.fully_included_pages)
        + "; partially included pages: " + format_page_ranges(scope.partially_included_pages)
        + "; omitted pages: " + format_page_ranges(scope.omitted_pages)
    )


def _render_summary(result: SummaryResult, config: ResolvedGenerationConfig) -> None:
    if not result.succeeded:
        st.error("Summary coverage was insufficient. " + " ".join(result.warnings))
        return
    for block in result.blocks:
        pages = sorted({int(source[1:5]) for source in block.source_ids if source.startswith("p") and source[1:5].isdigit()})
        citation = f"  ·  Pages {format_page_ranges(pages)}" if pages else ""
        st.markdown(f"- {block.text}{citation}")
    status = f"Generated with {config.primary.spec.alias}"
    fallback_used = config.secondary is not None and any(
        getattr(attempt, "target", "") == config.secondary.spec.alias for attempt in result.attempts
    )
    if fallback_used:
        status += " (fallback used)"
    st.caption(status)


def _pdf_pipeline(document: DocumentContent, budgets: TokenBudgets) -> tuple[Any, Any, Any]:
    """Reuse local chunks, scope, and BM25 for unchanged document/budget inputs."""
    cache_key = f"{document.document_id}:{settings_fingerprint(asdict(budgets))}"
    cache = st.session_state.setdefault("pdf_pipeline_cache", {})
    cached = cache.get(cache_key)
    if cached is None:
        chunks = chunk_document(document, budgets)
        cached = (chunks, select_summary_scope(chunks), build_bm25_index(chunks))
        cache[cache_key] = cached
    return cached


def _render_pdf(prepared: Mapping[str, Any], primary: str, secondary: str | None, fallback_enabled: bool, secrets: Mapping[str, Any]) -> None:
    document: DocumentContent = prepared["document"]
    budgets = selected_token_budgets(primary, secondary, fallback_enabled, secrets)
    _chunks, scope, index = _pdf_pipeline(document, budgets)
    st.subheader(document.display_name)
    st.caption(f"{len(document.pages)} physical pages extracted · OCR status: {document.ocr_status}")
    affected = [page.page_number for page in document.pages if page.warnings or not page.text]
    if affected:
        st.warning("Some pages may have limited extraction. Review affected pages in diagnostics below.")
        st.markdown("[Review affected pages](#diagnostics)")
    _render_scope(scope)

    settings = _settings_payload(primary, secondary, fallback_enabled, budgets)
    summary_nonce = st.session_state.summary_regeneration_nonce
    key = generation_fingerprint(prepared["identity"].document_id, "pdf-summary", "", settings, PROMPT_VERSION, summary_nonce)
    if st.button("Generate summary" if key not in st.session_state.summary_results else "Regenerate summary"):
        candidate_nonce = next_regeneration_nonce(summary_nonce) if key in st.session_state.summary_results else summary_nonce
        candidate_key = generation_fingerprint(prepared["identity"].document_id, "pdf-summary", "", settings, PROMPT_VERSION, candidate_nonce)
        config = _get_generation_config(primary, secondary, fallback_enabled, secrets)
        if config is not None:
            status = st.empty()
            generate, repair = _provider_generate(config, status)
            result = summarize_document(scope, token_budgets(config), generate, repair, _progress_renderer(status))
            if result.succeeded:
                st.session_state.summary_regeneration_nonce = candidate_nonce
                st.session_state.summary_results[candidate_key] = (result, config)
            else:
                st.error("Summary generation failed; the previous successful summary was kept.")
    key = generation_fingerprint(
        prepared["identity"].document_id,
        "pdf-summary",
        "",
        settings,
        PROMPT_VERSION,
        st.session_state.summary_regeneration_nonce,
    )
    cached = st.session_state.summary_results.get(key)
    if cached:
        st.subheader("Summary")
        _render_summary(*cached)

    st.subheader("Ask this document")
    question = st.text_input("Ask an independent question about this PDF", key="pdf_question")
    if st.button("Ask question") and question.strip():
        config = _get_generation_config(primary, secondary, fallback_enabled, secrets)
        if config is not None:
            answer_key = generation_fingerprint(prepared["identity"].document_id, "pdf-answer", question.strip(), settings, PROMPT_VERSION)
            if answer_key not in st.session_state.answer_history:
                status = st.empty()
                generate, repair = _provider_generate(config, status)
                retrieval = retrieve_evidence(question, index, token_budgets(config))
                st.session_state.answer_history[answer_key] = (question, answer_pdf_question(question, retrieval, token_budgets(config), generate, repair), config)
    for _, (asked, answer, config) in st.session_state.answer_history.items():
        st.markdown(f"**Question:** {asked}")
        _render_answer(answer, config)

    st.markdown("<a id='diagnostics'></a>", unsafe_allow_html=True)
    with st.expander("Diagnostics"):
        for page in document.pages:
            details = "; ".join(page.warnings) or page.extraction_method
            st.write(f"Page {page.page_number}: {details}")


def _render_answer(answer: AnswerResult, config: ResolvedGenerationConfig) -> None:
    if answer.refusal:
        st.info(answer.refusal)
        return
    for block in answer.blocks:
        st.write(block.text)
    st.caption("Cited pages: " + format_page_ranges(answer.cited_pages))
    if answer.ocr_page_numbers:
        st.caption("OCR provenance: pages " + format_page_ranges(answer.ocr_page_numbers))
    st.success(f"Grounded in {answer.grounded_passage_count} document passages")
    st.caption(f"Generated with {config.primary.spec.alias}")


def _render_pptx(prepared: Mapping[str, Any], primary: str, secondary: str | None, fallback_enabled: bool, secrets: Mapping[str, Any]) -> None:
    slides: list[str] = prepared["slides"]
    pptx_budgets = selected_token_budgets(primary, secondary, fallback_enabled, secrets)
    st.subheader(prepared["identity"].display_name)
    st.caption(f"{len(slides)} slides accepted for sequential analysis.")
    if st.button("Generate slide summaries"):
        config = _get_generation_config(primary, secondary, fallback_enabled, secrets)
        if config:
            for number, text in enumerate(slides, start=1):
                key = generation_fingerprint(prepared["identity"].document_id, "pptx-slide", str(number), _settings_payload(primary, secondary, fallback_enabled, pptx_budgets), PROMPT_VERSION)
                if key not in st.session_state.pptx_slide_results:
                    st.info(f"Summarizing slide {number} of {len(slides)}")
                    request = GenerationRequest("Summarize this untrusted slide text in concise bullet points.", text, 600, True)
                    try:
                        st.session_state.pptx_slide_results[key] = generate_with_fallback(request, config)
                    except ProviderError:
                        st.warning(f"Slide {number} could not be generated. Existing results were kept.")
                        break
    for number, _ in enumerate(slides, start=1):
        key = generation_fingerprint(prepared["identity"].document_id, "pptx-slide", str(number), _settings_payload(primary, secondary, fallback_enabled, pptx_budgets), PROMPT_VERSION)
        result = st.session_state.pptx_slide_results.get(key)
        if result:
            with st.expander(f"Slide {number}", expanded=False):
                st.write(result.text)
    question = st.text_input("Ask an independent question about this presentation", key="pptx_question")
    if st.button("Ask presentation") and question.strip():
        config = _get_generation_config(primary, secondary, fallback_enabled, secrets)
        if config:
            deck = "\n\n".join(f"Slide {number}: {text}" for number, text in enumerate(slides, 1))
            try:
                result = generate_with_fallback(GenerationRequest("Answer only from the supplied presentation text.", f"Presentation:\n{deck}\n\nQuestion: {question}", 900, True), config)
                st.session_state.pptx_answers[question] = result
            except ProviderError:
                st.warning("The presentation answer could not be generated. Existing results were kept.")
    for asked, result in st.session_state.pptx_answers.items():
        st.markdown(f"**Question:** {asked}")
        st.write(result.text)


def main() -> None:
    st.set_page_config(page_title="PDF Analyzer", layout="wide")
    _ensure_state()
    secrets = read_streamlit_secrets()
    st.title("PDF Analyzer")
    st.caption("Private, page-aware document analysis")
    primary, secondary, fallback_enabled = _provider_controls(secrets)
    st.subheader("Upload a document")
    st.caption("PDF or PPTX · 50 MB maximum")
    uploaded = st.file_uploader("Choose a PDF or PPTX", type=["pdf", "pptx"], label_visibility="collapsed")
    if uploaded is not None:
        prepared, error = ingest_upload(
            uploaded.name,
            uploaded.getvalue(),
            budgets=selected_token_budgets(primary, secondary, fallback_enabled, secrets),
        )
        if error:
            st.error(error)
        elif prepared is not None:
            st.success(f"Loaded {prepared['identity'].display_name}")
    current_settings = _settings_payload(
        primary,
        secondary,
        fallback_enabled,
        selected_token_budgets(primary, secondary, fallback_enabled, secrets),
    )
    settings_change = transition_generation_settings(st.session_state.get("generation_settings_fingerprint"), current_settings)
    if settings_change.generated_results_invalidated:
        _clear_generated_state()
        st.session_state.generation_settings_fingerprint = settings_fingerprint(current_settings)
    prepared = st.session_state.get("document_state")
    if not prepared:
        st.info("Extraction happens locally before any credential is required.")
        return
    if prepared["type"] == "pdf":
        _render_pdf(prepared, primary, secondary, fallback_enabled, secrets)
    else:
        _render_pptx(prepared, primary, secondary, fallback_enabled, secrets)


if __name__ == "__main__":
    main()
