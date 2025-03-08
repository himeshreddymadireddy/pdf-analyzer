# Pdf Analyzer with DeepSeek R1

A Streamlit-based application that analyzes PDF and PowerPoint documents using NVIDIA's DeepSeek R1 API. It provides **summarization** and **Q&A** capabilities for uploaded documents.

---

## Features

- **Document Upload**: Upload PDF or PPT files.
- **Summarization**:
  - Generate **slide/page-level summaries** for PPTs.
  - Generate **full-document summaries** for PDFs.
- **Q&A**:
  - Ask questions about the document content.
  - Get answers using DeepSeek R1's advanced reasoning.
- **Streaming Support**: Real-time responses for Q&A (optional).

---

## Tech Stack

- **Backend**: Python
- **AI Models**: DeepSeek R1 via NVIDIA API
- **Document Processing**: `PyPDF2`, `pdfminer.six`, `python-pptx`
- **UI**: Streamlit
- **Vector DB**: ChromaDB (optional for advanced Q&A)

---

## Prerequisites

1. **Python 3.8+**
2. **NVIDIA API Key**: Sign up at [NVIDIA API Portal](https://build.nvidia.com/) and generate an API key.
3. **Required Libraries**: Install dependencies using `pip`.

---

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/pdf-analyzer.git
   cd document-analyzer
   
2. **Install dependencies:**:
   ```bash
   pip install -r requirements.txt
   
3. **Set up your NVIDIA API key**:
   - Replace `nvapi-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx` in `app.py` with your actual API key.
   - Alternatively, set it as an environment variable:
     ```bash
     export NVIDIA_API_KEY="nvapi-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
## Usage
1. Run application:
   ```bash
   streamlit run app.py
   
2. Open your browser and navigate to `http://localhost:8501`
   
3. Upload a PDF or PPT file.
   
4. Explore the summaries and ask questions about the document.

---

## Code Structure
```bash
  document-analyzer/
├── app.py                # Main Streamlit application
├── utils.py              # Document processing and AI utilities
├── requirements.txt      # List of dependencies
└── README.md             # Project documentation
```

## Api Integration
- The project uses NVIDIA's DeepSeek R1 API for AI-powered document analysis. Here's how the API is integrated:
```bash
from openai import OpenAI

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key="nvapi-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"  # Replace with your API key
)

completion = client.chat.completions.create(
    model="deepseek-ai/deepseek-r1",
    messages=[{"role": "user", "content": "Your question here"}],
    temperature=0.6,
    top_p=0.7,
    max_tokens=4096,
    stream=False  # Set to True for streaming responses
)
```
## Example Workflow
1. Upload a PDF or PPT file.
   
2. Get automatic summaries for each slide/page.
 
3. Ask questions like:
  - "What are the key points of slide 3?"
  - "Explain the conclusion of the document."

# Customization
1. Change AI Model:
  - Replace `deepseek-ai/deepseek-r1` with another model supported by NVIDIA.
    
2. Add Streaming:
  - Set `stream=True` in the API call for real-time responses.
    
3. Enhance Security:
  - Store the API key in environment variables or a `.env` file.
## Troubleshooting
1. API Key Issues:
  - Ensure the API key is correct and has access to DeepSeek R1.
  - Check NVIDIA's API usage limits.
    
2. Streaming Errors:
  - If `stream=True` causes issues, set it to `False`.

3. Rate Limits:
   - Add retry logic for rate-limited requests:
  ```bash
     import time
def query_deepseek(prompt, api_key, retries=3):
    for _ in range(retries):
        try:
            return query_deepseek(prompt, api_key)
        except Exception as e:
            print(f"Error: {e}. Retrying...")
            time.sleep(2)
    return "Failed to generate response."
