# Local LangChain RAG App

A beginner-friendly RAG app built with Streamlit, LangChain, Ollama embeddings, ChromaDB, and optional OpenAI answer generation.

## Requirements
- Windows 10/11
- Python 3.10+
 
- Optional OpenAI API key for generated answers

## Windows setup (PowerShell)
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
 
streamlit run app.py
```

If PowerShell blocks virtual-environment activation, use:
```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
ollama pull nomic-embed-text
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open the local URL shown by Streamlit, usually http://localhost:8501.

## Optional OpenAI answer generation
Set the key in the same PowerShell terminal before starting Streamlit:
```powershell
$env:OPENAI_API_KEY = "YOUR_API_KEY"
$env:OPENAI_MODEL = "gpt-4o-mini"
streamlit run app.py
```
Without a key, the app still retrieves local chunks and displays them, but it will not generate an LLM answer.

## Usage
1. Open **Add knowledge**.
2. Upload PDF, TXT or Markdown documents, or paste notes/community post text.
3. Click Index/Save.
4. Open **Ask questions** and submit a question.
5. Review the answer and retrieved source chunks.

## Local storage
- `chroma_db/` contains the persistent local vector database.
- `data/` is used temporarily while parsing uploaded PDFs.
- Uploaded files are not permanently copied into `data/`; parsed text chunks are stored in Chroma.

## Important notes
- The first embedding request downloads/uses the Ollama model locally. Ensure Ollama is running.
- The embedding model must stay the same for an existing database. If you change embedding models, use a fresh Chroma directory.
- If OpenAI is enabled, the retrieved text and question are sent to OpenAI for answer generation.
- This is a small demo, not a production multi-user deployment.
