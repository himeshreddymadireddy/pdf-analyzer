import sys
import json
import base64
import traceback
from utils import process_pdf, process_ppt, TokenBudgets, chunk_document, select_summary_scope, build_bm25_index, summarize_document, answer_pdf_question, retrieve_evidence
from providers import resolve_generation_config, GenerationRequest, generate_with_fallback, repair_malformed_json, ProviderError

def handle_command():
    try:
        raw_input = sys.stdin.read()
        if not raw_input.strip():
            print(json.dumps({"success": False, "error": "Empty input"}))
            return
        payload = json.loads(raw_input)
        action = payload.get("action")
        
        if action == "process":
            file_b64 = payload.get("file_data")
            file_name = payload.get("file_name", "document.pdf")
            file_bytes = base64.b64decode(file_b64)
            
            if file_name.lower().endswith(".pdf"):
                doc = process_pdf(file_bytes, file_name, TokenBudgets())
                pages_out = [{"page_number": p.page_number, "text": p.text, "method": p.extraction_method, "warnings": list(p.warnings)} for p in doc.pages]
                print(json.dumps({
                    "success": True,
                    "type": "pdf",
                    "display_name": doc.display_name,
                    "document_id": doc.document_id,
                    "ocr_status": doc.ocr_status,
                    "pages": pages_out,
                    "warnings": list(doc.warnings)
                }))
            else:
                slides = process_ppt(file_bytes, TokenBudgets())
                print(json.dumps({
                    "success": True,
                    "type": "pptx",
                    "display_name": file_name,
                    "slides": slides
                }))
        elif action == "summarize":
            pages_data = payload.get("pages", [])
            doc_type = payload.get("doc_type", "pdf")
            provider_name = payload.get("provider", "Anthropic")
            model_name = payload.get("model", "claude-sonnet-4-6")
            api_key = payload.get("api_key")
            
            # Construct summary
            if doc_type == "pdf":
                text_content = "\n\n".join([f"--- Page {p.get('page_number')} ---\n{p.get('text')}" for p in pages_data])
            else:
                text_content = "\n\n".join([f"--- Slide {i+1} ---\n{s}" for i, s in enumerate(pages_data)])
                
            config = resolve_generation_config(provider_name, model_name, api_key)
            req = GenerationRequest(
                system_prompt="You are an expert document intelligence assistant. Provide a comprehensive summary with page-range or slide citations.",
                user_prompt=f"Please analyze and summarize the following document content:\n\n{text_content}",
                max_tokens=2048,
                temperature=0.2
            )
            resp = generate_with_fallback(config, req, None)
            summary_text = resp.content if resp and resp.content else "Failed to generate summary."
            print(json.dumps({"success": True, "summary": summary_text}))
            
        elif action == "qa":
            pages_data = payload.get("pages", [])
            doc_type = payload.get("doc_type", "pdf")
            question = payload.get("question", "")
            provider_name = payload.get("provider", "Anthropic")
            model_name = payload.get("model", "claude-sonnet-4-6")
            api_key = payload.get("api_key")
            
            if doc_type == "pdf":
                # Build BM25 index and retrieve evidence
                class PageObj:
                    def __init__(self, num, text):
                        self.page_number = num
                        self.text = text
                pdf_pages = [PageObj(p.get('page_number'), p.get('text')) for p in pages_data]
                bm25_idx = build_bm25_index(pdf_pages)
                evidence = retrieve_evidence(question, pdf_pages, bm25_idx, top_k=3)
                context_str = "\n\n".join([f"[Page {e.page_number}]: {e.text}" for e in evidence])
                citations = [{"page": e.page_number} for e in evidence]
            else:
                context_str = "\n\n".join([f"[Slide {i+1}]: {s}" for i, s in enumerate(pages_data)])
                citations = [{"slide": i+1} for i in range(len(pages_data))]
                
            config = resolve_generation_config(provider_name, model_name, api_key)
            req = GenerationRequest(
                system_prompt="You are an expert document QA assistant. Answer the user question strictly based on the provided document evidence and cite your sources.",
                user_prompt=f"Document Evidence:\n{context_str}\n\nQuestion: {question}",
                max_tokens=1500,
                temperature=0.2
            )
            resp = generate_with_fallback(config, req, None)
            answer_text = resp.content if resp and resp.content else "Could not generate answer."
            print(json.dumps({"success": True, "answer": answer_text, "citations": citations}))
        else:
            print(json.dumps({"success": False, "error": f"Unknown action: {action}"}))
    except Exception as e:
        print(json.dumps({"success": False, "error": str(e), "traceback": traceback.format_exc()}))

if __name__ == "__main__":
    handle_command()
