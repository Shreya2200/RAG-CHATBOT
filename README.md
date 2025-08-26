# RAG Chatbot Project

A Retrieval-Augmented Generation (RAG) chatbot that indexes documents, retrieves relevant context, and generates answers using an LLM. This README documents setup, usage, architecture, data layout, evaluation utilities, and troubleshooting, based on the repository structure and accessible files. Some files are tracked with Git LFS and not materialized in this environment; those are documented from context where possible.

---

## Quick Start

```bash
# 1) Create and activate a virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) Set API keys and environment variables
# Create a .env file in the repo root with at least:
# OPENAI_API_KEY=your_openai_key

# 4) Ingest documents into the vector store
# (Requires LFS content of ingest_database.py)
python ingest_database.py

# 5) Run the chatbot UI/server
# (Requires LFS content of chatbot.py)
python chatbot.py

# 6) Evaluate retrieval and generation
python evaluation.py
```

---

## Requirements

From `requirements.txt`:
- openai
- langchain
- python-dotenv
- PyMuPDF
- Pillow
- requests

System prerequisites:
- Python 3.9+
- Git LFS if cloning from source with LFS-tracked files
- Build tools for PyMuPDF (if needed for your OS)

---

## Project Structure

```
.
├─ .git/
├─ .github/workflows/          # GitHub Actions (if configured)
├─ .gradio/                    # (empty or assets for a Gradio UI)
├─ .vscode/settings.json       # Editor settings (LFS pointer here)
├─ chroma_db/                  # Chroma vector DB files
├─ chroma_db_chats/            # Alternate/legacy chat DB (Chroma)
├─ chroma_db_secure/           # Secure/private DB (Chroma)
├─ chroma_chat_sessions/       # Chroma DB for chat sessions
├─ chroma_sessions/            # Session storage (variant)
├─ data/                       # Source documents (PDFs etc.)
├─ histories/                  # Chat histories (.jsonl) (LFS pointers here)
├─ logs/                       # Run logs (jsonl)
├─ chatbot.py                  # Main app entry (LFS pointer)
├─ ingest_database.py          # Ingestion pipeline (LFS pointer)
├─ evaluation.py               # Evaluation utilities (present)
├─ requirements.txt            # Python dependencies
├─ sessions.json               # Session store (LFS pointer)
├─ processing_progress.json    # Ingestion progress (LFS pointer)
├─ .deployment                 # Deployment descriptor (LFS pointer)
└─ README.md                   # This file
```

Note: Many files are tracked with Git LFS and may appear as pointers unless LFS objects are fetched. See Troubleshooting for help.

---

## Architecture Overview

- Data ingestion: `ingest_database.py` (LFS) likely:
  - Loads documents from `data/` (PDFs, images)
  - Splits text into chunks, computes embeddings (via `langchain` + OpenAI embeddings)
  - Persists vectors and metadata to Chroma (`chroma_db/`)
  - Writes progress to `processing_progress.json`

- Chatbot runtime: `chatbot.py` (LFS) likely:
  - Initializes LLM client via OpenAI (`openai`) and orchestration via `langchain`
  - Loads `chroma_db` as retriever
  - Manages sessions (`sessions.json`, `histories/*.jsonl`), possibly using additional `chroma_*` stores
  - Serves a UI (Gradio) or CLI for conversation
  - Exposes functions used by `evaluation.py`: `make_llm`, `embeddings_model`, `best_answer`, `session_mgr`

- Evaluation: `evaluation.py` (present):
  - Imports from `chatbot.py`: `make_llm`, `embeddings_model`, `best_answer`, `session_mgr`
  - Retrieval metrics: Precision@K, Recall@K, F1@K, MRR@K, Semantic Similarity
  - Generation metrics: Answer Relevance, Completeness (LLM-judged), and Answer Similarity
  - Plotting: Generates `metrics.png`
  - Saving: Writes results to JSON files

- Storage:
  - Vector store: Chroma files in `chroma_db/` (and variants)
  - Session and history: `sessions.json`, `histories/*.jsonl`
  - Logs: `logs/chat-*.jsonl`

---

## Environment Variables

Create a `.env` file (loaded by `python-dotenv`) with keys such as:
- `OPENAI_API_KEY` (required)
- Optional keys depending on your setup, e.g., model selection, ports, feature flags

Example `.env`:
```
OPENAI_API_KEY=sk-...
MODEL=gpt-4o-mini
UI_PORT=7860
```

---

## Data Ingestion

- Place source documents in `data/`.
- Run:
```bash
python ingest_database.py
```
- Expected behavior (based on context):
  - The script processes PDFs (via PyMuPDF) and images (via Pillow) if present
  - Embeddings are created and upserted into Chroma collections in `chroma_db/`
  - Progress is tracked in `processing_progress.json`
- If you see only LFS pointers, fetch LFS objects first (see Troubleshooting).

---

## Running the Chatbot

Start the application:
```bash
python chatbot.py
```
Likely features:
- Starts a Gradio web UI (directory `.gradio/` exists) or CLI
- Uses Chroma retriever for context, then generates answers with OpenAI via LangChain
- Maintains per-session memory/history in `histories/` and `sessions.json`

---

## Evaluation

Use `evaluation.py` to quantify retrieval and generation quality.

- Retrieval evaluation:
  - Metrics: Precision@K, Recall@K, F1@K, MRR@K, Semantic Similarity (via `embeddings_model` cosine similarity)
  - Requires an active session in `session_mgr`

- Generation evaluation:
  - LLM-judged Relevance and Completeness (scaled 1–5 to 0–100)
  - Semantic similarity between expected and generated answers

- Run end-to-end:
```bash
python evaluation.py
```
- Outputs:
  - `retrieval_results.json`
  - `generation_results.json`
  - `metrics.png`

---

## Logs, Sessions, and Histories

- `logs/chat-*.jsonl`: Append-only logs per run
- `histories/*.jsonl`: Conversation transcripts (LFS pointers here)
- `sessions.json`: Session registry/state (LFS pointer here)

If these appear as LFS pointers, fetch the LFS objects to inspect contents.

---

## Deployment

A `.deployment` file exists (LFS pointer). Typical deployment steps may include:
- Setting environment variables in the target environment
- Ensuring LFS files are available at build time
- Running `ingest_database.py` as a provisioning step
- Running `chatbot.py` as a service (e.g., systemd, Docker, PaaS)

---

## Troubleshooting

- LFS files show as text pointers:
```bash
git lfs install
git lfs fetch --all
git lfs checkout
```
If objects are missing privately, ensure you have access to the remote and that the LFS artifacts were pushed.

- Missing OpenAI key:
  - Set `OPENAI_API_KEY` in `.env` or environment

- PyMuPDF build issues:
  - Ensure system packages for your OS are installed

- Empty retrieval results:
  - Verify `ingest_database.py` ran successfully and `chroma_db/` contains populated collections

- Evaluation errors like "No active session found in session_mgr":
  - Start the chatbot first to create a session or adapt `evaluation.py` to initialize one

---

## Contributing

1. Fork the repo and create a feature branch
2. Ensure LFS is configured if you modify LFS-tracked files
3. Add/update tests or evaluation examples where applicable
4. Open a pull request with a clear description

---

## License

Add your license file and reference it here.