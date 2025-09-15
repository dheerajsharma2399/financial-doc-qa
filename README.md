# Financial Document Q&A Assistant (Local: Streamlit + Ollama + FAISS)

Local-first app that ingests **PDF/Excel** financial docs and answers questions via a **RAG** pipeline using **LangChain**, **FAISS**, and **Ollama** small models — all on your machine.

## Features
- Streamlit UI: upload multiple PDFs/XLSX, chat interface
- PDF text extraction (PyMuPDF) + Excel rows (pandas)
- Chunking (RecursiveCharacterTextSplitter)
- Local embeddings: `nomic-embed-text` via Ollama
- Vector store: **FAISS** (in-memory)
-- Chat LLM: `mistral:7b` (configurable) via Ollama
- Citations table (file/page/sheet/row)

## Prereqs
- Python 3.10+
- [Ollama](https://ollama.com/download) running locally (default `http://localhost:11434`)
- Pull models:
  ```bash
  # Pull the Mistral 7B chat model and the embeddings model into Ollama
  ollama pull mistral:7b
  ollama pull nomic-embed-text
