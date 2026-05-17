# RAG on Rust Book — Implementation Plan

## Current State (Phase 1 — COMPLETE ✓)

| File | What it does |
|------|-------------|
| `ingest.py` | Reads 111 `.md` files from `data/`, chunks via langchain `MarkdownTextSplitter`, embeds via HF Inference API (batches of 32), stores in ChromaDB. Resume-safe by chunk ID. |
| `src/vectorstore.py` | ChromaDB `rust_book` collection (cosine), `all-MiniLM-L6-v2` via HF router API, `embedding_function=None` to prevent local model loading |
| `src/generator.py` | Wraps Groq API (LLaMA 3.3 70B) — formats context from top-5 chunks, calls LLM, returns answer string |
| `query.py` | Interactive loop: embed question → retrieve top-5 chunks → generate answer → print with sources |

**Actual parameters used:**
- Chunking: `chunk_size=500`, `chunk_overlap=50` (10% overlap)
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2` via HF router API
- LLM: `llama-3.3-70b-versatile` via Groq (swapped from Claude — Groq is free and fast)
- ChromaDB collection: `rust_book` — 3821 chunks indexed
- Token env vars: `HF_TOKEN`, `GROQ_API_KEY`

**Key fixes applied (from DEBUGGING_LOG.md):**
- HF API URL changed to `router.huggingface.co` — old endpoint returns 404
- `asyncio.WindowsSelectorEventLoopPolicy()` set before chromadb import — prevents Windows hang
- `ANONYMIZED_TELEMETRY=False` + `CHROMA_TELEMETRY=false` env vars — prevents chromadb telemetry hang
- `embedding_function=None` — prevents local onnxruntime model load

**Phase 1 is fully working** — retrieval is accurate, answers are grounded in the Rust Book, tested on threading and ownership questions.

---

## Phase 2 — Production Quality

### New files

```
src/
  bm25_store.py       — BM25 index (rank_bm25); serialised to disk with pickle
  hybrid_retriever.py — fuses vector scores + BM25 scores → ranked list
  reranker.py         — cross-encoder re-scores top-N candidates → top-k final
  config.py           — ALL prompts, model names, thresholds live here
```

### 2.1 BM25 Keyword Search (`src/bm25_store.py`)

```python
# Dependencies: rank_bm25, nltk (tokeniser)
class BM25Store:
    def build(self, chunks)      # tokenise + fit BM25Okapi, pickle to disk
    def load(self, path)         # unpickle
    def query(self, question, n) # returns list[{text, metadata, bm25_score}]
```

Build step runs once alongside `build_index.py`; saved as `bm25_index.pkl`.

### 2.2 Hybrid Retriever (`src/hybrid_retriever.py`)

Uses **Reciprocal Rank Fusion (RRF)**:

```
rrf_score(d) = Σ  1 / (k + rank_in_list_i)
               i
```

`k=60` (standard default). Both vector and BM25 contribute equal lists; their ranks are fused — no score normalisation needed.

```python
class HybridRetriever:
    def __init__(self, vector_store, bm25_store, rrf_k=60)
    def retrieve(self, question, n_candidates=20) -> list[dict]
        # 1. vector_store.query(question, n_candidates)
        # 2. bm25_store.query(question, n_candidates)
        # 3. RRF merge
        # 4. return top-n_candidates by rrf_score
```

### 2.3 Reranker (`src/reranker.py`)

Cross-encoder scores each (query, chunk) pair; more accurate than bi-encoder but slower — only run on the top-N candidates (not the full index).

```python
# Model: cross-encoder/ms-marco-MiniLM-L-6-v2  (fast, good quality)
class Reranker:
    def __init__(self, model_name)   # loaded from config.py
    def rerank(self, question, candidates, top_k) -> list[dict]
        # score each (question, chunk["text"]) pair
        # sort descending, return top_k
```

Pipeline becomes:
```
HybridRetriever.retrieve(q, n=20)  →  Reranker.rerank(q, candidates, top_k=5)  →  Generator.answer()
```

### 2.4 Citation Enforcement

The LLM is instructed (via prompt in `config.py`) to cite every factual claim with `[N]` where N is the chunk index passed in the context.

`generator.py` parses the response and:
- Verifies every `[N]` in the answer maps to a real chunk.
- Strips or flags uncited sentences if strict mode is on (controlled by `config.py`).

Output format:
```json
{
  "answer": "Ownership means ... [1]. Borrowing rules state ... [2].",
  "citations": [
    {"index": 1, "source": "ch04-01-what-is-ownership.md", "title": "What is Ownership?"},
    {"index": 2, "source": "ch04-02-references-and-borrowing.md", "title": "References and Borrowing"}
  ]
}
```

### 2.5 Config File (`src/config.py`)

Every string that controls model behaviour lives here — nothing hardcoded elsewhere.

```python
# --- Models ---
EMBED_MODEL   = "all-MiniLM-L6-v2"
RERANK_MODEL  = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LLM_MODEL     = "claude-sonnet-4-6"

# --- Retrieval ---
CHUNK_SIZE    = 1500
CHUNK_OVERLAP = 200
N_CANDIDATES  = 20   # hybrid retriever fetches this many before reranking
TOP_K         = 5    # chunks passed to LLM after reranking
RRF_K         = 60

# --- Prompts ---
SYSTEM_PROMPT = """You are an expert on The Rust Programming Language book.
Answer only using the provided context chunks.
Cite every claim with [N] where N is the chunk number.
If the answer is not in the context, say "I don't know based on the Rust Book."
"""

CONTEXT_TEMPLATE = """--- Chunk {index} | {title} ({source}) ---
{text}
"""

# --- Quality ---
FAITHFULNESS_THRESHOLD = 0.75   # Phase 3 CI gate
```

### Phase 2 — Updated `query.py` flow

```
question
   │
   ▼
HybridRetriever.retrieve(q, n=N_CANDIDATES)
   │
   ▼
Reranker.rerank(q, candidates, top_k=TOP_K)
   │
   ▼
Generator.answer(q, top_chunks)  → {answer, citations}
   │
   ▼
print answer + citation list
```

### New dependencies for Phase 2

```
rank-bm25>=0.2.2
nltk>=3.8
sentence-transformers>=2.7.0   # already present; cross-encoder included
anthropic>=0.25.0
```

---

## Phase 3 — Evaluation & CI Gate

### 3.1 Q/A Dataset (`data/eval_dataset.json`)

200–250 question/answer pairs generated from the Rust Book.

**Generation strategy (offline, one-time script `scripts/generate_dataset.py`):**
1. For each chapter chunk, prompt Claude to generate 2–3 factual questions whose answer is contained entirely within that chunk.
2. Store `{question, reference_answer, source_chunk_id}`.
3. Manual spot-check of ~20% before committing the dataset.

```json
[
  {
    "id": "q001",
    "question": "What does the ownership rule state about the number of owners a value can have?",
    "reference_answer": "Each value in Rust has exactly one owner.",
    "source_chunk_id": "ch04-01-what-is-ownership.md::2"
  },
  ...
]
```

### 3.2 Faithfulness Evaluator (`scripts/evaluate.py`)

**Faithfulness** = every claim in the generated answer is supported by the retrieved chunks.

Algorithm per sample:
1. Run the full RAG pipeline on `question` → get `{answer, citations, retrieved_chunks}`.
2. Split answer into atomic claims (sentences).
3. For each claim, ask an LLM judge:
   > "Is this claim: '{claim}' directly supported by the following context? Answer YES or NO."
4. `faithfulness_score = supported_claims / total_claims`

Overall score = mean across all samples.

```
scripts/
  evaluate.py          — runs evaluation, writes results/eval_report.json
  generate_dataset.py  — one-time Q/A dataset generation
```

### 3.3 CI Gate (`scripts/ci_check.py`)

```python
import json, sys
report = json.load(open("results/eval_report.json"))
score  = report["mean_faithfulness"]
threshold = 0.75   # from config.py FAITHFULNESS_THRESHOLD

if score < threshold:
    print(f"FAIL: faithfulness {score:.3f} < threshold {threshold}")
    sys.exit(1)   # non-zero exit → build fails in CI

print(f"PASS: faithfulness {score:.3f}")
sys.exit(0)
```

Wire into CI (GitHub Actions / any runner):
```yaml
- run: python scripts/evaluate.py
- run: python scripts/ci_check.py
```

### 3.4 Evaluation Metrics Summary

| Metric | How measured | Threshold |
|--------|-------------|-----------|
| Faithfulness | LLM judge: claims supported by retrieved chunks | ≥ 0.75 |
| (Optional P3+) Retrieval recall | reference chunk in top-k | ≥ 0.80 |
| (Optional P3+) Answer relevance | LLM judge: does answer address question | ≥ 0.70 |

---

## File Layout After All Three Phases

```
rag_ml/
├── src/
│   ├── __init__.py
│   ├── config.py           ← NEW P2: all prompts + params
│   ├── ingest.py
│   ├── chunker.py
│   ├── vectorstore.py
│   ├── bm25_store.py       ← NEW P2: BM25 index
│   ├── hybrid_retriever.py ← NEW P2: RRF fusion
│   ├── reranker.py         ← NEW P2: cross-encoder
│   └── generator.py        ← NEW P1+: LLM answer + citations
├── scripts/
│   ├── generate_dataset.py ← NEW P3: one-time Q/A generation
│   ├── evaluate.py         ← NEW P3: faithfulness eval
│   └── ci_check.py         ← NEW P3: build gate
├── data/
│   ├── eval_dataset.json   ← NEW P3: 200-250 Q/A pairs
│   └── (cached .md files)
├── results/
│   └── eval_report.json    ← NEW P3: written by evaluate.py
├── build_index.py          ← UPDATE P2: also builds BM25 index
├── query.py                ← UPDATE P1+, P2: full RAG pipeline
├── requirements.txt        ← UPDATE at each phase
└── IMPLEMENTATION_PLAN.md
```

---

## Open Questions / Decisions Needed

1. **LLM for answer generation** — plan uses Claude (`anthropic` SDK). OK to add that dependency, or prefer a local model?
2. **Strict citation mode** — should uncited sentences be hard-blocked (error) or just flagged (warning)?
3. **Faithfulness judge model** — using the same Claude model keeps it simple; a smaller/faster model could cut eval cost.
4. **Phase 3 trigger** — run eval on every push, or only on PRs to main?
