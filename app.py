# app.py
import os
import streamlit as st
from utils import process_pdf, process_ppt, chunk_text, query_deepseek
from dotenv import load_dotenv
load_dotenv()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")


# ========== Document Processing ==========
def generate_summary(content, is_slide=True):
    """Generate AI summary using DeepSeek R1."""
    prompt = f"Summarize this {'slide' if is_slide else 'document'} in 3-5 key points:\n{content}"
    return query_deepseek(prompt, DEEPSEEK_API_KEY)

def answer_question(question, context):
    """Answer questions using DeepSeek R1."""
    prompt = f"Answer this question based on the context below:\n\nContext: {context}\n\nQuestion: {question}"
    return query_deepseek(prompt, DEEPSEEK_API_KEY)

# ========== UI ==========
def main():
    st.title("Document Analyzer with DeepSeek R1 🚀")
    
    # File upload
    uploaded_file = st.file_uploader("Upload PDF/PPT", type=["pdf", "pptx"])
    doc_type = "ppt" if uploaded_file and uploaded_file.name.endswith(".pptx") else "pdf" if uploaded_file else None
    
    if uploaded_file:
        # Save temp file
        with open(uploaded_file.name, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Process document
        if doc_type == "pdf":
            text = process_pdf(uploaded_file.name)
        else:
            text = process_ppt(uploaded_file.name)
        
        # Show summaries
        st.sidebar.header("Document Insights")
        if doc_type == "ppt":
            for i, slide in enumerate(text):
                with st.sidebar.expander(f"Slide {i+1} Summary"):
                    st.write(generate_summary(slide))
        else:
            with st.sidebar.expander("Full Summary"):
                st.write(generate_summary(text, is_slide=False))
        
        # Chat interface
        st.divider()
        query = st.chat_input("Ask anything about the document...")
        if query:
            context = "\n".join(text) if doc_type == "ppt" else text
            response = answer_question(query, context)
            st.write(f"**Answer:** {response}")

if __name__ == "__main__":
    main()