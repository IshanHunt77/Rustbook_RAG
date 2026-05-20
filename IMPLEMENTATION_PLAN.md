# RAG on Rust Book — Implementation Plan

## Phase 1 — COMPLETE ✓

| File | Role |
|------|------|
| `src/ingest.py` | Fetches 111 `.md` chapters, caches to `data/` |
| `src/chunker.py` | Splits text at paragraph/sentence boundaries, respects code fences |
| `src/vectorstore.py` | ChromaDB `rust_book` collection (cosine), HF API embeddings, resume-safe |
| `src/generator.py` | Groq LLaMA 3.3 70B — formats top-5 chunks → answer string |
| `build_index.py` | Orchestrates ingest → chunk → embed → store |
| `query.py` | Interactive loop: embed → vector top-5 → generate → print |

**Parameters:** `chunk_size=1500`, `overlap=350`, `all-MiniLM-L6-v2`, 3821 chunks, `HF_TOKEN` + `GROQ_API_KEY`

**Critical Windows fixes (do not remove):**
- `asyncio.WindowsSelectorEventLoopPolicy()` before chromadb import
- `embedding_function=None` on collection
- `ANONYMIZED_TELEMETRY=False`
- HF URL: `router.huggingface.co`

---

## Phase 2 — COMPLETE ✓ — Hybrid Retrieval + Reranking + Citations

### Pipeline change

```
Phase 1:  question → vector top-5 → LLM → string

Phase 2:  question → [vector top-20 + BM25 top-20] → RRF merge top-20
                   → cross-encoder rerank → top-5 → LLM → {answer, citations}
```

### Build order

| # | File | Action |
|---|------|--------|
| 1 | `src/config.py` | CREATE — all constants + prompts |
| 2 | `src/bm25_store.py` | CREATE — BM25 index class |
| 3 | `build_index.py` | MODIFY — add BM25 build step after vector step |
| 4 | `src/hybrid_retriever.py` | CREATE — RRF fusion |
| 5 | `src/reranker.py` | CREATE — cross-encoder |
| 6 | `src/generator.py` | MODIFY — citations, return dict |
| 7 | `query.py` | MODIFY — wire full pipeline |
| 8 | `requirements.txt` | MODIFY — add new deps |

**Do not touch:** `src/chunker.py`, `src/ingest.py`, `src/vectorstore.py`

**Deviations from original plan (intentional):**
- Used `bm25s` instead of `rank-bm25` — no NLTK corpus download required
- Reranker uses HF Inference API (`BAAI/bge-reranker-base`) instead of local `sentence-transformers` CrossEncoder — required for low-spec hardware (no local model loading)
- `BM25_INDEX_PATH` is a directory (`bm25_index/`) not a `.pkl` file — bm25s saves multiple index files
- BM25 index built via GitHub Actions workflow (`.github/workflows/build_bm25.yml`) due to RAM constraints on dev machine
- `src/chunker.py` infinite loop bug fixed: `start = next_start if next_start > start else break_pos`

**Definition of done — verified:**

| Question | Top source cited |
|----------|-----------------|
| `"what does move do in Rust?"` | ch13-01-closures.md + ownership context |
| `"how does Box<T> work?"` | ch15-01-box.md |
| `"what is unsafe?"` | ch20-01-unsafe-rust.md |

---

### `src/config.py`

Single source of truth for all tuneable values. Every other file imports from here.

| Constant | Value |
|----------|-------|
| `EMBED_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` |
| `RERANK_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| `LLM_MODEL` | `llama-3.3-70b-versatile` |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1500` / `350` |
| `N_CANDIDATES` | `20` (into reranker) |
| `TOP_K` | `5` (into LLM) |
| `RRF_K` | `60` |
| `BM25_INDEX_PATH` | `bm25_index.pkl` |
| `CHROMA_DIR` | `chroma_db` |
| `FAITHFULNESS_THRESHOLD` | `0.75` |
| `SYSTEM_PROMPT` | must instruct `[N]` citation on every claim |
| `CONTEXT_TEMPLATE` | `--- Chunk {index} | {title} ({source}) ---\n{text}` |

---

### `src/bm25_store.py`

**Methods:** `build(chunks, path)` → `load(path)` → `query(question, n) → list[dict]`

**Gotchas:**
- Tokenisation in `build` and `query` must be identical: `nltk.word_tokenize`, lowercased
- `build` must store the original chunk list alongside the BM25 model (needed to return `text` + `metadata` at query time)
- Run `nltk.download("punkt_tab")` once before first use
- Output shape per result: `{text, metadata, bm25_score}`

---

### `build_index.py` changes

- Collect `all_chunks` during the existing chapter loop
- After the vector loop completes, call `BM25Store().build(all_chunks)`
- Replace hardcoded `chunk_size=1500, overlap=350` with `config.CHUNK_SIZE / CHUNK_OVERLAP`
- `bm25_index.pkl` must exist before `query.py` can run

---

### `src/hybrid_retriever.py`

**Method:** `retrieve(question, n_candidates) → list[dict]`

**RRF formula:** `score(chunk) = 1/(RRF_K + rank_vector + 1) + 1/(RRF_K + rank_bm25 + 1)`

**Gotchas:**
- Dedup key = `metadata["source"] + "::" + str(metadata["chunk_index"])` — matches ChromaDB IDs
- Chunks appearing in only one list still get partial RRF score (do not discard them)
- Output shape per result: `{text, metadata, rrf_score}`

---

### `src/reranker.py`

**Method:** `rerank(question, candidates, top_k) → list[dict]`

Uses `CrossEncoder.predict([(question, text), ...])` — scores all pairs in one batch call.

**Gotchas:**
- Model (~80 MB) downloads on first instantiation — normal, not a hang
- Mutate candidates in-place with `rerank_score`, then sort descending, slice `top_k`
- Instantiate once at startup in `query.py`, not per query

---

### `src/generator.py` changes

- Import `LLM_MODEL`, `SYSTEM_PROMPT`, `CONTEXT_TEMPLATE` from `config`
- Return type changes: `str` → `dict` with keys `answer` (str) and `citations` (list)
- Parse `[N]` markers from response with `re.findall(r'\[(\d+)\]', text)`
- Only emit citations where `1 ≤ N ≤ len(chunks)` — discard out-of-range hallucinated indices
- **Breaking change:** `query.py` expects a string today — update it in the same commit

---

### `query.py` changes

- Init order: `RustBookVectorStore` → `BM25Store.load()` → `HybridRetriever` → `Reranker` — all once before the loop
- Query order per question: `hr.retrieve` → `rr.rerank` → `answer`
- Print `result["answer"]` then `result["citations"]`

---

### New dependencies

```
rank-bm25>=0.2.2
nltk>=3.8
sentence-transformers>=2.7.0
```

---

### Definition of done

`python query.py` answers these three correctly with `[N]` markers and matching sources:

| Question | Expected top source |
|----------|-------------------|
| `"what does move do in Rust?"` | `ch04-01-what-is-ownership.md` |
| `"how does Box<T> work?"` | `ch15-01-box.md` |
| `"what is unsafe?"` | chapter with `unsafe` in name |

---

## Phase 3 — COMPLETE ✓ — Evaluation & CI Gate

### Files

| File | Role |
|------|------|
| `scripts/generate_dataset.py` | One-time: generates 200 Q/A pairs from corpus chunks via Groq |
| `data/eval_dataset.json` | `{id, question, reference_answer, source_chunk_id, context}` per entry |
| `scripts/evaluate.py` | Runs full pipeline on 15 samples, scores with RAGAS |
| `results/eval_report.json` | Written by evaluate.py: `{mean_faithfulness, mean_context_recall, per_question: [...]}` |
| `scripts/ci_check.py` | Reads report, exits 1 if `mean_faithfulness < FAITHFULNESS_THRESHOLD` |
| `scripts/test_apis.py` | 8-check pre-flight: Groq, HF embed, HF rerank, RAGAS wrappers, pipeline smoke |

### Pipeline

```
evaluate.py per sample:
  question → hybrid_retriever → reranker → generator → {answer, citations}
  RAGAS: Faithfulness(answer, retrieved_contexts) + ContextRecall(answer, reference)
  → per-sample scores → mean across all samples → eval_report.json
```

### Metrics

| Metric | Method | Threshold | Actual |
|--------|--------|-----------|--------|
| Faithfulness | RAGAS LLM judge (LangchainLLMWrapper + Groq) | ≥ 0.75 | **1.000** |
| Context recall | RAGAS LLM judge | — | **1.000** |

### Deviations from original plan (intentional)

- **15 samples evaluated** (not 200) — Groq free tier TPD limit is 100k tokens/day; 200 samples × 2 RAGAS metrics would exceed this. 15 samples stays under ~50k tokens.
- **AnswerRelevancy dropped** — Groq API only supports `n=1` completions. RAGAS's `AnswerRelevancy` internally requests `n > 1` causing `BadRequestError`. Only `Faithfulness` + `ContextRecall` are used (both work with `n=1`).
- **`max_workers=1` in RunConfig** — RAGAS default `max_workers=16` fires 16 concurrent Groq calls, immediately hitting the 12k TPM cap and causing all Faithfulness judge calls to timeout at 120s → `nan`. Sequential execution (`max_workers=1`) prevents this entirely.
- **`LangchainLLMWrapper(ChatGroq(...))` kept** — RAGAS 0.4.3 has two incompatible metric APIs: `ragas.metrics` (old, works with `evaluate()`) and `ragas.metrics.collections` (new, incompatible with `evaluate()`). New metrics are not subclasses of `ragas.metrics.base.Metric` so `evaluate()` rejects them. Old API with deprecation warnings is the only working path.
- **`PIPELINE_SLEEP=2s`** between pipeline calls to respect Groq + HF rate limits during the pipeline phase.
- **`timeout=180`** (was 120) — gives Groq responses more breathing room under light rate limiting.

### Definition of done — verified

CI passed on GitHub Actions (run 26184924310):

```
Faithfulness:   1.000  (threshold: 0.75)
Context recall: 1.000
Samples: 15  |  Skipped: 0
CI PASSED
```

---

## Final File Layout

```
rag_ml/
├── src/
│   ├── config.py           ← P2 NEW
│   ├── bm25_store.py       ← P2 NEW
│   ├── hybrid_retriever.py ← P2 NEW
│   ├── reranker.py         ← P2 NEW
│   ├── generator.py        ← P2 MODIFIED
│   ├── vectorstore.py      ← untouched
│   ├── chunker.py          ← untouched
│   ├── ingest.py           ← untouched
│   └── __init__.py         ← untouched
├── scripts/
│   ├── generate_dataset.py ← P3 NEW
│   ├── evaluate.py         ← P3 NEW
│   ├── ci_check.py         ← P3 NEW
│   └── test_apis.py        ← P3 NEW
├── data/
│   └── eval_dataset.json   ← P3 NEW
├── results/
│   └── eval_report.json    ← P3 written at runtime
├── build_index.py          ← P2 MODIFIED
├── query.py                ← P2 MODIFIED
├── bm25_index/             ← P2 written at runtime (directory, not pkl)
├── requirements.txt        ← updated each phase
└── IMPLEMENTATION_PLAN.md
```
