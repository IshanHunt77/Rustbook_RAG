# Rust Book RAG

Ask any question about The Rust Programming Language book and get a cited answer — the system finds the most relevant passages and uses an LLM to write the response.

---

## How it works

```
SETUP  (runs once)
──────────────────────────────────────────────────────────────────

  The Rust Book (111 chapters on GitHub)
            │
            ▼
  ┌─────────────────────┐
  │  Split into chunks  │  each chunk = ~1500 characters of text
  │  (1,278 pieces)     │
  └─────────────────────┘
            │
     ┌──────┴──────┐
     ▼             ▼
  ┌──────────┐  ┌──────────┐
  │ MEANING  │  │ KEYWORDS │
  │ DATABASE │  │ DATABASE │
  │ ChromaDB │  │  BM25    │
  │          │  │  Index   │
  │ "what is │  │          │
  │  the     │  │ finds    │
  │  intent?"│  │ exact    │
  │          │  │ words    │
  └──────────┘  └──────────┘


EVERY QUESTION
──────────────────────────────────────────────────────────────────

  You type:  "what does Box<T> do in Rust?"
                          │
            ┌─────────────┴─────────────┐
            ▼                           ▼
     ┌─────────────┐             ┌─────────────┐
     │  Search by  │             │  Search by  │
     │   MEANING   │             │  KEYWORDS   │
     │  (ChromaDB) │             │   (BM25)    │
     │  top 20     │             │  top 20     │
     └─────────────┘             └─────────────┘
            │                           │
            └─────────────┬─────────────┘
                          ▼
               ┌─────────────────────┐
               │   COMBINE & RANK    │
               │      (RRF)          │
               │  chunks in BOTH     │
               │  lists rank higher  │
               │  → best 20 chunks   │
               └─────────────────────┘
                          │
                          ▼
               ┌─────────────────────┐
               │     RERANKER        │
               │  (AI judge via HF)  │
               │  scores each chunk: │
               │  "how relevant is   │
               │  this to the exact  │
               │  question asked?"   │
               │  → best 5 chunks    │
               └─────────────────────┘
                          │
                          ▼
               ┌─────────────────────┐
               │      LLM            │
               │  (Llama 3.3 70B     │
               │   via Groq)         │
               │  reads 5 chunks +   │
               │  your question →    │
               │  writes answer with │
               │  [N] citation tags  │
               └─────────────────────┘
                          │
                          ▼
          ┌───────────────────────────────┐
          │  Answer:                      │
          │  Box<T> stores data on the    │
          │  heap [1]. Memory is freed    │
          │  automatically when the box   │
          │  goes out of scope [1][5].    │
          │                               │
          │  Sources:                     │
          │  [1] ch15-01-box.md           │
          │  [5] ch15-01-box.md           │
          └───────────────────────────────┘


WHY TWO DATABASES?
──────────────────────────────────────────────────────────────────

  Question: "how does a box work?"
    ChromaDB finds  → Box<T>, smart pointers, heap  (understands meaning)
    BM25 struggles  → "box" is a common English word (too generic)

  Question: "RefCell<T> borrow rules"
    ChromaDB misses → exact type name is too specific (meaning too vague)
    BM25 nails it   → exact string "RefCell<T>"      (keyword match)

  Together they cover what neither alone can handle.
```

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

### 3. Build the vector index

```cmd
python -u ingest.py
```

Resume-safe — restart anytime, already-indexed chunks are skipped.

> **First run:** ChromaDB takes 20–30 seconds to initialise. This is normal.

### 4. Build the BM25 index

The BM25 index is CPU-intensive and must be built on a machine with sufficient RAM.
A pre-built index can be generated via the GitHub Actions workflow (`.github/workflows/build_bm25.yml`) and downloaded as an artifact. Place the extracted files in `bm25_index/`.

### 5. Query

```cmd
python query.py
```

Type any Rust question at the prompt. Type `quit` to exit.

---

## Components

| File | Purpose |
|---|---|
| `ingest.py` | Chunk + embed markdown files → ChromaDB (`rust_book` collection) |
| `build_bm25.py` | Build BM25 keyword index → `bm25_index/` |
| `query.py` | Interactive Q&A — full hybrid pipeline |
| `src/config.py` | All constants and prompts |
| `src/vectorstore.py` | ChromaDB interface + HF embedding calls |
| `src/bm25_store.py` | BM25 index build + query |
| `src/hybrid_retriever.py` | RRF fusion of vector + BM25 results |
| `src/reranker.py` | Cross-encoder reranking via HF API |
| `src/generator.py` | Groq LLM answer generation with citations |

---

## Configuration

| Env variable | Required | Description |
|---|---|---|
| `HF_TOKEN` | Yes | HuggingFace API token for embeddings and reranking |
| `GROQ_API_KEY` | Yes | Groq API key for answer generation |
| `ANONYMIZED_TELEMETRY` | Recommended | Set `False` to prevent ChromaDB telemetry hang |
| `CHROMA_TELEMETRY` | Recommended | Set `false` to prevent ChromaDB telemetry hang |

---

## Evaluation (Phase 3)

Every push runs an automated quality gate via GitHub Actions:

1. The full RAG pipeline runs on 15 samples from `data/eval_dataset.json` (200 Q/A pairs generated from the corpus via Groq)
2. RAGAS scores each answer for **faithfulness** (are claims grounded in retrieved context?) and **context recall** (did retrieval find the reference chunk?)
3. CI fails if `mean_faithfulness < 0.75`

**Current scores:** faithfulness `1.000` · context recall `1.000`

| Script | Purpose |
|---|---|
| `scripts/generate_dataset.py` | One-time: generates 200 Q/A pairs from corpus chunks via Groq |
| `scripts/evaluate.py` | Runs pipeline on eval samples, scores with RAGAS |
| `scripts/ci_check.py` | Reads `results/eval_report.json`, exits 1 if below threshold |
| `scripts/test_apis.py` | Pre-flight check — verifies all external APIs before triggering CI |

---

## Project Status

- **Phase 1** — Complete. Vector retrieval, HF embeddings, Groq LLM generation.
- **Phase 2** — Complete. Hybrid BM25 + vector retrieval, RRF fusion, cross-encoder reranking, citations.
- **Phase 3** — Complete. RAGAS evaluation dataset + faithfulness/context-recall CI gate. Passing at 1.000/1.000.

See `IMPLEMENTATION_PLAN.md` for full technical details.
