import io
import os
import sys
import time
from typing import List, Tuple, Dict
import os

from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import pandas as pd

# --- LangChain core ---
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# --- Google Gemini integrations ---
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# --- Ollama integrations ---
try:
    from langchain_community.llms import Ollama
    from langchain_community.embeddings import OllamaEmbeddings
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

# --- PDF/Excel parsing ---
import fitz  # PyMuPDF
import pandas as pd

# ----------------------------
# Helpers: extraction
# ----------------------------

def extract_pdf_text(file_bytes: bytes) -> List[Tuple[str, Dict]]:
    """Return list of (chunk_text, metadata) from a PDF."""
    chunks = []
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        for page_num in range(len(doc)):
            page = doc[page_num]
            txt = page.get_text("text") or ""
            if txt.strip():
                chunks.append((txt, {"type": "pdf", "page": page_num + 1}))
    return chunks

def extract_excel_text(file_bytes: bytes, filename: str) -> List[Tuple[str, Dict]]:
    """Return list of (chunk_text, metadata) from ALL sheets of an Excel file."""
    chunks = []
    xls = pd.ExcelFile(io.BytesIO(file_bytes))
    for sheet in xls.sheet_names:
        df = xls.parse(sheet_name=sheet)
        # Turn each row into a single line for retrieval; keep columns and values
        # For big sheets, this is a simple but effective approach
        for i, row in df.fillna("").iterrows():
            kv = [f"{c}: {str(row[c]).strip()}" for c in df.columns if str(row[c]).strip()]
            if kv:
                text = " | ".join(kv)
                chunks.append((text, {"type": "excel", "sheet": sheet, "row": int(i)}))
    return chunks

def normalize_and_chunk(text_batches: List[Tuple[str, Dict]], base_metadata: Dict, chunk_size=1000, chunk_overlap=150) -> List[Document]:
    """Normalize whitespace and split into chunks; wrap in LangChain Document objects."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs: List[Document] = []
    for raw_text, meta in text_batches:
        cleaned = " ".join(str(raw_text).split())
        if not cleaned:
            continue
        for chunk in splitter.split_text(cleaned):
            md = dict(base_metadata)
            md.update(meta)
            docs.append(Document(page_content=chunk, metadata=md))
    return docs

# ----------------------------
# RAG Pipeline
# ----------------------------

@st.cache_resource(show_spinner=False)
def get_embeddings(provider="Gemini Cloud API"):
    if provider == "Gemini Cloud API":
        return GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=os.environ.get("GOOGLE_API_KEY")
        )
    elif provider == "Ollama (localhost)":
        if not OLLAMA_AVAILABLE:
            st.error("Ollama is not available. Please install `langchain-community` to use it.")
            st.stop()
        return OllamaEmbeddings(model="nomic-embed-text")

@st.cache_resource(show_spinner=False)
def get_chat_model(provider, temperature: float, max_tokens: int):
    if provider == "Gemini Cloud API":
        return ChatGoogleGenerativeAI(
            model="gemini-1.5-pro-latest",
            temperature=temperature,
            max_output_tokens=max_tokens,
            google_api_key=os.environ.get("GOOGLE_API_KEY")
        )
    elif provider == "Ollama (localhost)":
        if not OLLAMA_AVAILABLE:
            st.error("Ollama is not available. Please install `langchain-community` to use it.")
            st.stop()
        return Ollama(
            model="mistral",
            temperature=temperature,
        )

def build_vectorstore(all_docs: List[Document], provider: str):
    emb = get_embeddings(provider)
    return FAISS.from_documents(all_docs, emb)

def retrieve_context(vs: FAISS, query: str, k: int = 6) -> List[Document]:
    return vs.similarity_search(query, k=k)

def make_prompt(user_q: str, context_docs: List[Document]) -> str:
    context = "\n\n".join(
        [f"[Doc {i+1}] {d.metadata}:\n{d.page_content}" for i, d in enumerate(context_docs)]
    )
    system = (
        "You are a careful financial analyst. Answer ONLY using the provided context. "
        "If a figure is requested, quote it exactly as written (with currency/period). "
        "If the context is insufficient, say you cannot find it."
    )
    return f"{system}\n\nUser Question:\n{user_q}\n\nContext:\n{context}\n\nAnswer:"

# ----------------------------
# Streamlit UI
# ----------------------------

st.set_page_config(page_title="Financial Doc Q&A", page_icon="💬", layout="wide")
st.title("📄 Financial Document Q&A Assistant")

with st.sidebar:
    st.header("⚙️ Settings")
    model_provider = st.selectbox(
        "Choose your model provider",
        ("Gemini Cloud API", "Ollama (localhost)")
    )

    if model_provider == "Gemini Cloud API":
        st.caption("Models: `Gemini 1.5 Pro` and `embedding-001`.")
    else:
        st.caption("Using Ollama models from localhost.")

    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
    max_tokens = st.number_input("Max tokens", 128, 4096, 768, 64)
    top_k = st.slider("Top-K Retrieval", 2, 12, 6, 1)


# Session state
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chat_model" not in st.session_state:
    st.session_state.chat_model = None
if "messages" not in st.session_state:
    st.session_state.messages = []

st.subheader("Upload Documents (PDF / Excel)")
files = st.file_uploader(
    "Drop one or more files",
    type=["pdf", "xlsx", "xls"],
    accept_multiple_files=True
)

col_a, col_b = st.columns(2)
build_clicked = col_a.button("🔍 Process & Build Index", use_container_width=True)
clear_clicked = col_b.button("🧹 Clear Session", use_container_width=True)

if clear_clicked:
    st.session_state.vectorstore = None
    st.session_state.chat_model = None
    st.session_state.messages = []
    st.toast("Session cleared.", icon="🧹")

if build_clicked:
    if not files:
        st.error("Please upload at least one PDF or Excel file.")
    else:
        with st.status("Parsing and indexing...", expanded=True) as status:
            all_docs: List[Document] = []
            for f in files:
                st.write(f"Reading: **{f.name}**")
                try:
                    data = f.read()
                    base_md = {"file": f.name}
                    if f.type == "application/pdf" or f.name.lower().endswith(".pdf"):
                        raw = extract_pdf_text(data)
                    else:
                        raw = extract_excel_text(data, f.name)
                    docs = normalize_and_chunk(raw, base_md)
                    all_docs.extend(docs)
                    st.write(f" → {len(docs)} chunks.")
                except Exception as e:
                    st.exception(e)

            if not all_docs:
                st.error("No content extracted from the uploaded files.")
                st.stop()

            try:
                vs = build_vectorstore(all_docs, model_provider)
                st.session_state.vectorstore = vs
                st.session_state.chat_model = get_chat_model(model_provider, temperature, int(max_tokens))
                status.update(label="Index built successfully ✅", state="complete", expanded=False)
                st.toast("Index ready.", icon="✅")
            except Exception as e:
                st.exception(e)
                st.stop()

st.subheader("💬 Ask Questions")
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_q = st.chat_input("e.g., What is total revenue and net income for FY 2023?")

if user_q:
    if st.session_state.vectorstore is None:
        st.error("Please upload and build the index first.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": user_q})
    with st.chat_message("user"):
        st.markdown(user_q)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                ctx = retrieve_context(st.session_state.vectorstore, user_q, k=top_k)
                prompt = make_prompt(user_q, ctx)
                llm = st.session_state.chat_model
                # Newer LangChain chat models use `invoke`; fall back to `predict` if needed.
                try:
                    res = llm.invoke(prompt)
                    # res might be a string or an LLMResult-like object
                    if isinstance(res, str):
                        answer = res
                    else:
                        # Try to extract generated text from common result shapes
                        gens = getattr(res, "generations", None)
                        if gens and len(gens) and len(gens[0]):
                            answer = getattr(gens[0][0], "text", str(gens[0][0]))
                        else:
                            # Fallback to string conversion
                            answer = str(res)
                except Exception:
                    # Older versions used predict()
                    try:
                        answer = llm.predict(prompt)
                    except Exception as e:
                        raise
            except Exception as e:
                st.exception(e)
                answer = "Sorry, I ran into an error generating the answer."

            st.markdown(answer)

            with st.expander("Context & Citations"):
                if not ctx:
                    st.caption("_No context matched._")
                else:
                    rows = []
                    for i, d in enumerate(ctx, 1):
                        rows.append({
                            "#": i,
                            "file": d.metadata.get("file"),
                            "type": d.metadata.get("type"),
                            "page": d.metadata.get("page"),
                            "sheet": d.metadata.get("sheet"),
                            "row": d.metadata.get("row"),
                            "chars": len(d.page_content)
                        })
                    st.dataframe(pd.DataFrame(rows), width='stretch')

            st.session_state.messages.append({"role": "assistant", "content": answer})
