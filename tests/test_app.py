from __future__ import annotations

import ast
from pathlib import Path

import pytest

from app_state import upload_identity
from utils import DocumentContent, PageContent, TokenBudgets


def _app_test():
    pytest.importorskip("streamlit")
    try:
        from streamlit.testing.v1 import AppTest
    except ImportError:
        pytest.skip("This Streamlit installation has no supported AppTest module.")
    return AppTest.from_file("app.py")


def _caption_values(app) -> list[str]:
    return [str(caption.value) for caption in app.caption]


def test_app_uses_the_pure_upload_seam_and_does_not_restore_legacy_dotenv_flow():
    source = Path("app.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    from_imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert "dotenv" not in imports
    assert "dotenv" not in from_imports
    assert "query_deepseek" not in source
    assert "ingest_upload" in source
    assert "PDF or PPTX · 50 MB maximum" in source


def test_app_has_no_key_suffix_or_copy_ui_claims():
    source = Path("app.py").read_text(encoding="utf-8").lower()
    assert "suffix" not in source
    assert 'st.button("copy' not in source


def test_apptest_renders_setup_controls_without_a_secrets_file():
    app = _app_test()
    app.run()
    assert not app.exception
    assert any("PDF or PPTX" in value for value in _caption_values(app))
    assert "Choose a PDF or PPTX" in Path("app.py").read_text(encoding="utf-8")
    assert any(select.label == "Provider" for select in app.selectbox)
    assert any(select.label == "Model" for select in app.selectbox)
    assert "Credential status: **missing**" in _caption_values(app)


def test_apptest_password_override_and_blank_input_update_credential_label():
    app = _app_test()
    app.run()

    api_key = app.text_input[0]
    api_key.set_value("temporary-key").run()
    assert not app.exception
    assert app.session_state["api_key_overrides"] == {"ANTHROPIC_API_KEY": "temporary-key"}
    assert "Credential status: **session override**" in _caption_values(app)

    api_key.set_value("").run()
    assert not app.exception
    assert app.session_state["api_key_overrides"] == {}
    assert "Credential status: **missing**" in _caption_values(app)


def test_apptest_clear_override_uses_callback_without_post_widget_mutation():
    app = _app_test()
    app.run()
    app.text_input[0].set_value("temporary-key").run()
    app.button[0].click().run()

    assert not app.exception
    assert app.text_input[0].value == ""
    assert app.session_state["api_key_overrides"] == {}
    assert "Credential status: **missing**" in _caption_values(app)


def test_missing_primary_blocks_generation_with_a_safe_error(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    app = _app_test()
    app.run()

    # Inject prepared state through the documented pure-helper seam; AppTest 1.33
    # cannot upload files directly.
    identity = upload_identity(b"fixture", "fixture.pdf")
    document = DocumentContent(
        identity.document_id,
        identity.display_name,
        (PageContent(1, "Revenue was 12 percent.", "native", 20),),
        "not_needed",
    )
    app.session_state["document_state"] = {"identity": identity, "type": "pdf", "document": document}
    app.session_state["active_document"] = identity
    app.run()

    generate = next(button for button in app.button if button.label == "Generate summary")
    generate.click().run()
    assert not app.exception
    assert any("primary credential is required" in str(error.value).lower() for error in app.error)


def test_safe_secrets_reader_handles_missing_and_changed_private_parser_signatures():
    import app as app_module

    class MissingFile:
        def _parse(self, *, print_exceptions):
            assert print_exceptions is False
            raise FileNotFoundError

    class ChangedSignature:
        def _parse(self):
            return {"ANTHROPIC_API_KEY": "must-not-be-read"}

    assert app_module.read_streamlit_secrets(MissingFile()) == {}
    assert app_module.read_streamlit_secrets(ChangedSignature()) == {}


def test_pdf_pipeline_memoizes_chunks_scope_and_bm25_for_document_and_budget(monkeypatch):
    import app as app_module

    class State(dict):
        def setdefault(self, key, default=None):
            return super().setdefault(key, default)

    state = State()
    monkeypatch.setattr(app_module.st, "session_state", state)
    document = DocumentContent(
        "document-id",
        "fixture.pdf",
        (PageContent(1, "revenue evidence", "native", 16),),
        "not_needed",
    )
    calls = {"chunks": 0, "scope": 0, "index": 0}
    original_chunks = app_module.chunk_document
    original_scope = app_module.select_summary_scope
    original_index = app_module.build_bm25_index

    def chunks(*args, **kwargs):
        calls["chunks"] += 1
        return original_chunks(*args, **kwargs)

    def scope(*args, **kwargs):
        calls["scope"] += 1
        return original_scope(*args, **kwargs)

    def index(*args, **kwargs):
        calls["index"] += 1
        return original_index(*args, **kwargs)

    monkeypatch.setattr(app_module, "chunk_document", chunks)
    monkeypatch.setattr(app_module, "select_summary_scope", scope)
    monkeypatch.setattr(app_module, "build_bm25_index", index)

    first = app_module._pdf_pipeline(document, TokenBudgets())
    second = app_module._pdf_pipeline(document, TokenBudgets())
    changed_budget = app_module._pdf_pipeline(document, TokenBudgets(chunk_tokens=1000))

    assert first is second
    assert changed_budget is not first
    assert calls == {"chunks": 2, "scope": 2, "index": 2}
