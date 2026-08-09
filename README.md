# PDF Analyzer

A Streamlit application for bounded, page-aware analysis of PDFs and PPTX files. It extracts document text locally, sends only the selected analysis prompt and document passages to the model provider you choose, and preserves source-page citations for accepted PDF answers and summaries.

## What it does

- Accepts **PDF** and **PPTX** uploads up to **50 MiB**.
- Extracts PDF text locally with PyMuPDF; eligible image-heavy or poorly extracted pages can use local Tesseract OCR.
- Summarizes PDFs with sequential map/reduce generation and page-level source citations.
- Answers each PDF question independently using local BM25 retrieval; unsupported questions are refused before a provider call.
- Summarizes PPTX slides sequentially and supports independent presentation questions.
- Lets you choose a primary model and, optionally, one distinct fallback model for retryable failures only.

## Requirements

- Python 3.10 or newer (the project is tested with Python 3.12).
- A key for at least one selected provider:
  - Anthropic: `ANTHROPIC_API_KEY`
  - DeepSeek: `DEEPSEEK_API_KEY`
  - OpenAI: `OPENAI_API_KEY`
- For PDF support: PyMuPDF, installed through `requirements.txt`.
- For PPTX support: `python-pptx`, installed through `requirements.txt`.
- For local OCR: Tesseract with English language data (`eng.traineddata`). OCR is optional; PDF text extraction still works without it.

Install the Python dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Configure credentials

The application **does not load `.env` files**. Do not put credentials in source code, command history, or a committed file.

Choose one of these credential sources:

1. **Session override (recommended for a one-off local session).** Select a provider in the sidebar and enter its API key in the password field. The value is held in Streamlit session state and is not displayed back to the user.
2. **Streamlit secrets (recommended for a persistent local or deployed setup).** Copy the example and add only the provider keys you intend to use:

   ```bash
   cp .streamlit/secrets.toml.example .streamlit/secrets.toml
   ```

   Edit `.streamlit/secrets.toml` with a real key locally. That file is gitignored; the committed example contains placeholders only.
3. **Process environment.** Export the relevant provider variable before starting Streamlit:

   ```bash
   export OPENAI_API_KEY="replace-with-your-key"
   # or ANTHROPIC_API_KEY / DEEPSEEK_API_KEY
   ```

Credential precedence is exact and provider-specific:

1. non-blank **session override**;
2. non-blank **Streamlit secret**;
3. non-blank **process environment variable**;
4. otherwise, the credential is unavailable.

Blank values are skipped. For example, a blank session override does not suppress a configured Streamlit secret. A key for one provider is never used for another provider.

## Run

```bash
streamlit run app.py
```

Open the local address Streamlit prints, normally `http://localhost:8501`. Select a provider and model, add a credential through one of the supported sources, then upload a PDF or PPTX.

## Providers and model selection

The model menu is a deliberately fixed reviewed catalog; it is not a general model-ID input. The current entries are:

| Provider | Menu label | Provider model ID | Transport |
| --- | --- | --- | --- |
| Anthropic | Claude Sonnet 5 | `claude-sonnet-5` | Messages API |
| Anthropic | Claude Haiku 4.5 | `claude-haiku-4-5-20251001` | Messages API |
| DeepSeek | DeepSeek V4 Flash | `deepseek-v4-flash` | Chat Completions API |
| DeepSeek | DeepSeek V4 Pro | `deepseek-v4-pro` | Chat Completions API |
| OpenAI | GPT-5.4 Mini | `gpt-5.4-mini` | Responses API |
| OpenAI | GPT-5.6 Sol | `gpt-5.6-sol` | Responses API |

DeepSeek uses the official DeepSeek endpoint, `https://api.deepseek.com`, through the OpenAI-compatible client. It no longer uses NVIDIA NIM, NVIDIA API keys, or `langchain-nvidia-ai-endpoints`.

### Fallback behavior

Enable **Use fallback model** only when you have configured a second provider/model credential. The fallback must be different from the primary. A request makes at most two primary attempts (one retry after a retryable failure) and then at most two attempts with the one fallback. Authentication, authorization, invalid-request, model, content, and context failures do not trigger fallback. If the fallback credential is absent, the app keeps the primary-only configuration and shows a warning.

## Document behavior and limits

### PDFs

PDF extraction, chunking, retrieval, and OCR run locally. PDF summaries are built sequentially: the app maps selected source passages and safely reduces valid maps only when enough source coverage is available. A summary fails closed instead of presenting a partial final result when coverage or citations are invalid.

| Limit or behavior | Value |
| --- | --- |
| Upload size | 50 MiB |
| OCR candidates processed | First 10 eligible pages |
| OCR resolution | 300 DPI |
| Summary physical pages considered | First 40 pages |
| Summary source chunks considered | First 80 chunks |
| Chunk size / overlap | 1,200 / 120 estimated tokens |
| Retrieved PDF answer passages | At most 6 |
| Generation concurrency | 1 (sequential) |

A page is an OCR candidate only when it has raster evidence and weak native extraction. OCR output replaces native text only when it materially improves usable text. The Diagnostics panel reports extraction/OCR warnings and pages omitted by a summary scope.

For PDF questions, each question is independent: the app retrieves against the original document chunks, not previous questions or answers. If retrieval does not sufficiently cover the question, the application returns:

> I couldn’t find enough support in this document to answer that.

without calling a generation provider. Valid PDF answers show cited pages, and identify any citation relying on OCR provenance. Generated JSON or citations that fail validation are not shown as supported answers.

### PPTX

PPTX processing extracts text locally and accepts at most 50 slides or 60,000 estimated extracted tokens. Slide summaries are generated sequentially. Presentation questions are independent from prior presentation answers. PPTX responses do not currently provide the PDF-style page citation validation, so treat them as a separate, text-only presentation workflow.

## Privacy and data routing

- Upload bytes are processed in memory by the application; the app does not save uploaded documents to a user-named temporary file.
- PDF parsing, PPTX parsing, OCR, token estimation, chunking, and BM25 retrieval are local.
- When generation is requested, the selected provider receives the system instructions and the text required for that request: bounded PDF passages/maps, one slide for a slide summary, or the presentation text for a PPTX question.
- The selected provider may be Anthropic, DeepSeek, or OpenAI. Their account, retention, regional-routing, and data-use terms govern data after it leaves this app. Review those terms before uploading confidential or regulated material.
- Provider credentials are excluded from prompts, normal errors, cache fingerprints, and displayed status. This application does not load `.env` files.

## Testing and checks

Run the test suite without provider credentials or network access:

```bash
pytest -q
```

The tests cover credential precedence, provider request mapping and fallback bounds, upload/state handling, PDF extraction and retrieval limits, citation/refusal behavior, malformed JSON repair bounds, PPTX preflight, and the Streamlit setup controls. If Streamlit is not installed, the Streamlit `AppTest` test is skipped; install the project requirements to run it.

Run a syntax/bytecode consistency check:

```bash
python -m compileall -q app.py app_state.py providers.py utils.py tests
```

## Troubleshooting

| Symptom | Check / resolution |
| --- | --- |
| “A primary credential is required” | Configure the key for the selected provider. Confirm precedence: session override, then Streamlit secrets, then the process environment. Remove a stale session override with **Clear override**. |
| “SDK is unavailable” | Reinstall with `python -m pip install -r requirements.txt` in the environment that runs Streamlit. |
| PDF processing unavailable | Ensure PyMuPDF installed successfully: `python -m pip show PyMuPDF`. |
| PPTX processing unavailable or unreadable | Ensure `python-pptx` is installed and upload a valid `.pptx` file, not legacy `.ppt`. |
| OCR unavailable | Install Tesseract plus English language data and ensure `eng.traineddata` is readable. On Debian/Ubuntu: `sudo apt-get install tesseract-ocr tesseract-ocr-eng`. On macOS: `brew install tesseract`. Restart Streamlit after changing the local Tesseract installation. |
| OCR did not run on every scanned page | OCR is intentionally capped at the first 10 eligible candidates; inspect Diagnostics and split the PDF if necessary. |
| Unsupported PDF answer | Rephrase with terms that occur in the document, or inspect the cited/diagnostic pages. The refusal is intentional when local evidence is insufficient. |
| Summary is unavailable | The selected pages may not have reached valid map coverage, may exceed the safe request budget, or may have invalid citations. Review the displayed scope and diagnostics; split the document if needed. |
| Provider failure or fallback was not used | Permanent credential, permission, model, input, and content failures never fall back. Correct the provider configuration or choose a valid model. Retryable failures get one retry and then the optional distinct fallback. |
| Model is not available | The catalog is curated. Select another documented menu entry rather than entering an arbitrary provider model ID. |

## Risks and rollback

**Operational risks.** Provider model availability, quotas, API parameters, and policies can change. OCR quality varies by scan quality and language (this implementation requests English data). Local retrieval can refuse a question even if a human could infer an answer, by design. PPTX questions send the assembled presentation text to the selected provider and lack the PDF citation contract. The 50 MiB upload limit does not guarantee a small extracted token count; split unusually dense files.

**Rollback.** If a provider integration is unreliable, select another configured catalog model or disable fallback for primary-only behavior. To roll back this V2 integration at the repository level, revert the V2 changes to `app.py`, `app_state.py`, `providers.py`, `utils.py`, tests, `requirements.txt`, `.gitignore`, `.streamlit/secrets.toml.example`, and this README as one reviewed unit. Do not reintroduce NVIDIA NIM dependencies, hard-coded keys, or `.env` loading as an ad-hoc workaround.

## Repository layout

```text
app.py                              Streamlit UI adapter
app_state.py                        Pure upload and session-state helpers
providers.py                        Provider catalog, credentials, adapters, retry/fallback logic
utils.py                            Local extraction, OCR, chunking, retrieval, grounding
.streamlit/config.toml              Streamlit upload-size configuration
.streamlit/secrets.toml.example     Safe credential template
requirements.txt                    Runtime and test dependencies
tests/                              Offline unit and Streamlit UI tests
```
