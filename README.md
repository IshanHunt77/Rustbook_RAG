# Rust Book RAG

RAG system over The Rust Programming Language book. Chunks markdown chapters, embeds via HuggingFace Inference API, stores in ChromaDB, and answers questions using Groq LLaMA 3.3 70B.

## Components

| File | Purpose |
|---|---|
| `ingest.py` | Chunk + embed markdown files → ChromaDB (`rust_book` collection) |
| `query.py` | Interactive Q&A over the indexed book |
| `src/vectorstore.py` | ChromaDB interface + HF embedding calls |
| `src/generator.py` | Groq LLM answer generation |

---

## Setup

### 1. Install dependencies

```cmd
pip install -r requirements.txt
```

### 2. Configure credentials

Copy `.env.example` to `.env` and fill in your tokens:

```
HF_TOKEN=hf_...
GROQ_API_KEY=gsk_...
ANONYMIZED_TELEMETRY=False
CHROMA_TELEMETRY=false
```

- HuggingFace token: <https://huggingface.co/settings/tokens> (read-only scope)
- Groq API key: <https://console.groq.com/keys> (free tier available)

### 3. Load env vars and run ingestion

```cmd
set HF_TOKEN=<your_token>
set GROQ_API_KEY=<your_key>
set ANONYMIZED_TELEMETRY=False
set CHROMA_TELEMETRY=false
python -u ingest.py
```

Ingestion is **resume-safe** — restart at any time and already-indexed chunks are skipped.

> **First run note:** ChromaDB takes 20–30 seconds to initialise on first run. This is normal — do not kill the process.

### 4. Query

```cmd
python query.py
```

Type any Rust question at the prompt. Type `quit` to exit.

---

## Configuration

| Env variable | Required | Description |
|---|---|---|
| `HF_TOKEN` | Yes | HuggingFace API token for embeddings |
| `GROQ_API_KEY` | Yes | Groq API key for answer generation |
| `ANONYMIZED_TELEMETRY` | Recommended | Set `False` to prevent ChromaDB telemetry hang |
| `CHROMA_TELEMETRY` | Recommended | Set `false` to prevent ChromaDB telemetry hang |
| `RUST_BOOK_DIR` | No | Override markdown source dir (default: `d:\rag_ml\data`) |
| `CHROMA_DIR` | No | Override ChromaDB path (default: `./chroma_db`) |

---

## Project Status

- **Phase 1** — Complete. 3821 chunks indexed, retrieval and generation working.
- **Phase 2** — Planned. Hybrid BM25 + vector retrieval with cross-encoder reranking.
- **Phase 3** — Planned. Evaluation dataset + faithfulness CI gate.

See `IMPLEMENTATION_PLAN.md` for full details.
