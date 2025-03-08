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
