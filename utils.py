# utils.py
from openai import OpenAI
from pdfminer.high_level import extract_text as extract_pdf_text
from pptx import Presentation
from langchain.text_splitter import RecursiveCharacterTextSplitter

def process_pdf(file_path):
    """Extract text from PDF"""
    return extract_pdf_text(file_path)

def process_ppt(file_path):
    """Extract text from PowerPoint"""
    prs = Presentation(file_path)
    text = []
    for slide in prs.slides:
        slide_text = []
        for shape in slide.shapes:
            if hasattr(shape, "text"):
                slide_text.append(shape.text.strip())
        text.append("\n".join(slide_text))
    return text

def chunk_text(text, chunk_size=1000, chunk_overlap=200):
    """Split text into chunks"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_text(text)

def query_deepseek(prompt, api_key):
    """Query DeepSeek R1 API for completions."""
    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=api_key
    )
    completion = client.chat.completions.create(
        model="deepseek-ai/deepseek-r1",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.6,
        top_p=0.7,
        max_tokens=4096,
        stream=False  # Set to True for streaming responses
    )
    return completion.choices[0].message.content