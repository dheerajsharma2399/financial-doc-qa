# Financial Document Q&A Assistant

A local-first application that ingests PDF and Excel financial documents and answers questions using a RAG (Retrieval-Augmented Generation) pipeline with LangChain, FAISS, and Ollama — all running on your machine with complete privacy.

## 🎯 Features

- **📤 Streamlit UI**: Clean web interface for uploading multiple PDFs and XLSX files with an intuitive chat interface
- **📄 PDF Text Extraction**: Extract text content using PyMuPDF (fitz) with page-level granularity
- **📊 Excel Processing**: Parse Excel files with pandas, extracting rows from all sheets
- **✂️ Intelligent Chunking**: Uses LangChain’s RecursiveCharacterTextSplitter for optimal text segmentation
- **🔐 Local Embeddings**: Generate embeddings using `nomic-embed-text` via Ollama (no external API calls)
- **💾 Vector Store**: FAISS in-memory vector database for fast similarity search
- **🤖 Chat LLM**: Powered by `mistral:7b` (configurable) via Ollama for natural language responses
- **📌 Citations Table**: Track answers with precise citations including file name, page number, sheet name, and row number
- **🔒 Privacy-First**: Everything runs locally — your financial documents never leave your machine

## 📋 Prerequisites

- **Python**: 3.10 or higher
- **Ollama**: Running locally (default: `http://localhost:11434`)
  - Download from [ollama.com/download](https://ollama.com/download)

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/dheerajsharma2399/financial-doc-qa.git
cd financial-doc-qa
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install and Configure Ollama

Download and install Ollama from [ollama.com/download](https://ollama.com/download)

Pull the required models:

```bash
# Pull the Mistral 7B chat model
ollama pull mistral:7b

# Pull the embeddings model
ollama pull nomic-embed-text
```

Verify Ollama is running:

```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags
```

## 💻 Usage

### Start the Application

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

### Using the Application

1. **Upload Documents**
- Click the file uploader in the sidebar
- Select one or multiple PDF or Excel (.xlsx) files
- Wait for the documents to be processed and indexed
1. **Ask Questions**
- Type your question in the chat input box
- Press Enter or click Submit
- View the AI-generated answer with citations
1. **Review Citations**
- Check the citations table below each answer
- See exactly which file, page, or sheet/row the information came from
- Verify sources for accuracy and transparency

### Example Questions

- “What was the total revenue in Q4 2023?”
- “Summarize the key findings from the financial report”
- “What are the main expenses listed in the budget?”
- “Compare the revenue figures across different quarters”

## 📦 Dependencies

Main libraries used in this project:

- **streamlit**: Web UI framework
- **langchain**: RAG pipeline orchestration
- **langchain-community**: Community integrations
- **faiss-cpu**: Vector similarity search
- **pymupdf** (fitz): PDF text extraction
- **pandas**: Excel file processing
- **openpyxl**: Excel file reading
- **ollama**: Local LLM integration

See `requirements.txt` for complete dependency list.

## 🏗️ Architecture

```
┌─────────────────┐
│  Streamlit UI   │
└────────┬────────┘
         │
         v
┌─────────────────┐
│ Document Loader │  (PDF: PyMuPDF, Excel: Pandas)
└────────┬────────┘
         │
         v
┌─────────────────┐
│ Text Chunking   │  (RecursiveCharacterTextSplitter)
└────────┬────────┘
         │
         v
┌─────────────────┐
│   Embeddings    │  (nomic-embed-text via Ollama)
└────────┬────────┘
         │
         v
┌─────────────────┐
│  FAISS Vector   │  (In-memory vector store)
│     Store       │
└────────┬────────┘
         │
         v
┌─────────────────┐
│  RAG Pipeline   │  (LangChain)
└────────┬────────┘
         │
         v
┌─────────────────┐
│ Mistral LLM     │  (via Ollama)
└────────┬────────┘
         │
         v
┌─────────────────┐
│     Answer      │  (with citations)
└─────────────────┘
```

## ⚙️ Configuration

You can customize the following parameters in the code:

- **LLM Model**: Change from `mistral:7b` to any Ollama-supported model
- **Chunk Size**: Adjust text chunking parameters for different document types
- **Embedding Model**: Switch to alternative embedding models available in Ollama
- **Top K Results**: Modify the number of similar documents retrieved

## 🔧 Troubleshooting

### Ollama Connection Issues

```bash
# Check if Ollama is running
ollama list

# Restart Ollama service
# On macOS/Linux:
killall ollama
ollama serve

# On Windows, restart the Ollama desktop application
```

### Model Not Found

```bash
# Verify models are pulled
ollama list

# Re-pull if necessary
ollama pull mistral:7b
ollama pull nomic-embed-text
```

### Memory Issues

For large documents, consider:

- Reducing chunk size
- Processing documents in batches
- Using a smaller LLM model (e.g., `mistral:7b-instruct-q4_0`)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
1. Create your feature branch (`git checkout -b feature/AmazingFeature`)
1. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
1. Push to the branch (`git push origin feature/AmazingFeature`)
1. Open a Pull Request

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- [Ollama](https://ollama.com/) for local LLM inference
- [LangChain](https://www.langchain.com/) for RAG framework
- [FAISS](https://github.com/facebookresearch/faiss) for vector similarity search
- [Streamlit](https://streamlit.io/) for rapid UI development

## 📧 Contact

Dheeraj Sharma - [@dheerajsharma2399](https://github.com/dheerajsharma2399)

Project Link: [https://github.com/dheerajsharma2399/financial-doc-qa](https://github.com/dheerajsharma2399)

-----

⭐ If you find this project helpful, please consider giving it a star!
