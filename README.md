# AI Research Assistant

An AI-powered research assistant that helps users explore scientific literature through document question answering, research paper summarization, and comparative paper analysis.

Built with Python, Google's Gemini API, ChromaDB, and Streamlit.

## Features

* **Research Q&A:** Ask questions about indexed research papers using Retrieval-Augmented Generation (RAG).
* **PDF Summarization:** Generate structured summaries of uploaded research papers.
* **Semantic Search:** Retrieve relevant document chunks using Gemini embeddings and ChromaDB.
* **Source References:** Display source filenames, page numbers, and retrieved text-chunk information with answers.
* **Paper Comparison:** Compare research papers across objectives, methodologies, techniques, findings, limitations, and research gaps.
* **Downloadable Summaries:** Export paper summaries as Markdown files.

## Technology Stack

* Python
* Google Gemini API
* Gemini Embeddings
* ChromaDB
* LangChain text splitters
* PyPDF
* Streamlit

## Project Structure

```text
AI Research Assistant/
├── app.py
├── ingest_documents.py
├── requirements.txt
├── .env.example
├── data/
│   ├── documents/
│   └── uploads/
├── src/
│   ├── document_loader.py
│   ├── text_splitter.py
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── retriever.py
│   ├── llm.py
│   ├── rag_pipeline.py
│   ├── summarizer.py
│   └── agents/
└── vectorstore/
```

The `data/documents/`, `data/uploads/`, and `vectorstore/` directories contain local data and are not intended to be committed.

## Prerequisites

* Python 3.10 or a compatible version supported by the installed dependencies
* Git
* A Google Gemini API key

## Installation

Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd "AI Research Assistant"
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use the virtual environment's Python executable directly:

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Configuration

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

Replace the placeholder with your own API key. Never commit `.env` or publish your API key.

## Add Research Papers

Place research papers you are authorized to process in:

```text
data/documents/
```

Supported input format: text-extractable PDF.

## Index Documents

Run from the project root:

```bash
python ingest_documents.py
```

This extracts text, splits it into chunks, generates embeddings, and stores the resulting vectors in ChromaDB.

## Run the Application

```bash
python -m streamlit run app.py
```

Open the local URL displayed in the terminal.

## Paper Comparison

The project includes a command-line paper comparison module. After placing two PDFs in accessible locations, run:

```bash
python -m src.paper_comparator "path/to/paper1.pdf" "path/to/paper2.pdf" --output "comparison_report.md"
```

The comparison report is generated locally.

## Security and Research Integrity

* Keep API keys and secrets out of version control.
* Do not commit private or copyrighted research documents without permission.
* Verify generated summaries, numerical results, citations, and research claims against the original papers.
* AI-generated research comparisons should be treated as drafts requiring human review.

## Current Status

The core research Q&A, PDF summarization, document ingestion, semantic retrieval, and paper comparison workflows have been implemented and tested locally.

## Future Improvements

* Improve evidence attribution and confidence handling.
* Integrate paper comparison into the Streamlit interface.
* Add evaluation datasets and retrieval-quality metrics.
* Improve handling of scanned PDFs and large documents.
* Add export options for comparative literature reviews.
