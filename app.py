import os
import tempfile
from pathlib import Path
from io import BytesIO

import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Load environment variables from the .env file beside this app.py.
APP_DIR = Path(__file__).resolve().parent
ENV_FILE = APP_DIR / ".env"
load_dotenv(dotenv_path=ENV_FILE, override=False)

# All application logic is in this one file.
FAISS_DIR = APP_DIR / "faiss_index"
FAISS_DIR.mkdir(parents=True, exist_ok=True)
INDEX_NAME = "community_knowledge"
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

st.set_page_config(page_title="Local RAG with FAISS", page_icon="🧠", layout="wide")
st.title("🧠 Local RAG App")
st.caption("Streamlit + LangChain + FAISS + OpenAI embeddings")

@st.cache_resource
def get_embeddings():
    """Create embeddings using the OpenAI Embeddings API."""
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY is not set. Add it to the .env file beside app.py, "
            "then restart Streamlit."
        )
    return OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=api_key)

def get_vector_store():
    """Load the saved FAISS index, or return None if it does not exist."""
    index_file = FAISS_DIR / f"{INDEX_NAME}.faiss"
    pkl_file = FAISS_DIR / f"{INDEX_NAME}.pkl"
    if index_file.exists() and pkl_file.exists():
        # Only load an index you created and trust locally. The metadata file uses pickle.
        return FAISS.load_local(
            str(FAISS_DIR),
            get_embeddings(),
            index_name=INDEX_NAME,
            allow_dangerous_deserialization=True,
        )
    return None

def extract_uploaded_file(uploaded_file):
    """Extract text from PDF, TXT, or Markdown uploaded in Streamlit."""
    filename = uploaded_file.name
    suffix = Path(filename).suffix.lower()
    raw = uploaded_file.getvalue()

    if suffix == ".pdf":
        reader = PdfReader(BytesIO(raw))
        pages = []
        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                pages.append(
                    Document(
                        page_content=text,
                        metadata={"source": filename, "page": page_number},
                    )
                )
        return pages
    if suffix in {".txt", ".md"}:
        text = raw.decode("utf-8", errors="replace")
        return [Document(page_content=text, metadata={"source": filename})]
    raise ValueError("Supported formats are PDF, TXT, and Markdown (.md).")

def add_documents(docs):
    """Split documents and add chunks to FAISS, then save locally."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
        add_start_index=True,
    )
    chunks = splitter.split_documents(docs)
    if not chunks:
        return 0

    store = get_vector_store()
    if store is None:
        store = FAISS.from_documents(chunks, get_embeddings())
    else:
        store.add_documents(chunks)

    store.save_local(str(FAISS_DIR), index_name=INDEX_NAME)
    return len(chunks)

def retrieve_and_answer(question, top_k):
    store = get_vector_store()
    if store is None:
        return "Your FAISS database is empty. Add documents in the **Add knowledge** tab first.", []

    results = store.similarity_search(question, k=top_k)
    if not results:
        return "No relevant content was found in your local database.", []

    context_parts = []
    for doc in results:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page")
        source_label = f"{source}, page {page}" if page else source
        context_parts.append(f"[Source: {source_label}]\n{doc.page_content}")
    context = "\n\n---\n\n".join(context_parts)

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return (
            "Relevant content was retrieved from FAISS. Set `OPENAI_API_KEY` in the `.env` file "
            "to generate a natural-language answer.\n\n**Retrieved context:**\n\n" + context,
            results,
        )

    llm = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        api_key=api_key,
        temperature=0.2,
    )
    prompt = f"""Answer the question using only the context below.
If the answer is not in the context, say you could not find it in the indexed documents.
Do not invent facts. Cite source filenames in square brackets where useful.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""
    answer = llm.invoke(prompt).content
    return answer, results

with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Number of chunks to retrieve", 1, 8, 4)
    st.markdown(f"**Embedding model:** `{EMBEDDING_MODEL}`")
    st.caption("OpenAI embeddings require an API key and send text to OpenAI.")
    st.caption(f"FAISS index folder: `{FAISS_DIR}`")
    if (APP_DIR / ".env").exists():
        st.caption("Environment file: `.env` loaded")
    else:
        st.warning("No .env file found beside app.py. Create one with OPENAI_API_KEY=...")
    if st.button("Check FAISS index"):
        if get_vector_store() is None:
            st.info("No saved index found yet.")
        else:
            st.success("Saved FAISS index found.")

tab_add, tab_ask = st.tabs(["1. Add knowledge", "2. Ask questions"])

with tab_add:
    st.subheader("Upload knowledge files")
    uploaded_files = st.file_uploader(
        "Upload PDF, TXT, or Markdown files",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
    )
    if st.button("Index uploaded files", type="primary", disabled=not uploaded_files):
        total_chunks = 0
        errors = []
        progress = st.progress(0)
        for i, uploaded_file in enumerate(uploaded_files):
            try:
                docs = extract_uploaded_file(uploaded_file)
                for doc in docs:
                    doc.metadata.setdefault("source", uploaded_file.name)
                total_chunks += add_documents(docs)
            except Exception as exc:
                errors.append(f"{uploaded_file.name}: {exc}")
            progress.progress((i + 1) / len(uploaded_files))
        if total_chunks:
            st.success(f"Indexed {total_chunks} chunks and saved them to your local FAISS database.")
        if not total_chunks and not errors:
            st.warning("No readable text was found in the uploaded files.")
        for error in errors:
            st.error(error)


with tab_ask:
    st.subheader("Ask your knowledge base")
    question = st.text_area(
        "Your question",
        placeholder="Ask a question about the documents you indexed…",
        height=100,
    )
    if st.button("Retrieve and answer", type="primary", disabled=not question.strip()):
        try:
            with st.spinner("Searching FAISS and preparing the answer…"):
                answer, sources = retrieve_and_answer(question.strip(), top_k)
            st.markdown("### Answer")
            st.write(answer)
            if sources:
                st.markdown("### Retrieved chunks")
                for number, doc in enumerate(sources, start=1):
                    source = doc.metadata.get("source", "unknown")
                    page = doc.metadata.get("page")
                    label = f"{number}. {source}" + (f" · page {page}" if page else "")
                    with st.expander(label):
                        st.write(doc.page_content)
                        st.caption(f"Metadata: {doc.metadata}")
        except Exception as exc:
            st.error(
                f"RAG failed: {exc}\n\n"
                "Check that OPENAI_API_KEY is set and valid. OpenAI embeddings and generated "
                "answers both require API access; confirm your account has API billing enabled."
            )

st.divider()
st.caption(
    "The FAISS index is stored locally in ./faiss_index. Source documents stay on your computer, "
    "but text is sent to OpenAI to create embeddings. If answer generation is enabled, the question "
    "and retrieved context are also sent to OpenAI."
)
