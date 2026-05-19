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

## Phase 2 — Hybrid Retrieval + Reranking + Citations

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

## Phase 3 — Evaluation & CI Gate

### Files

| File | Role |
|------|------|
| `scripts/generate_dataset.py` | One-time: generates 200-250 Q/A pairs from chunks via LLM |
| `data/eval_dataset.json` | `{id, question, reference_answer, source_chunk_id}` per entry |
| `scripts/evaluate.py` | Runs full pipeline on each question, LLM-judges faithfulness per claim |
| `results/eval_report.json` | Written by evaluate.py: `{mean_faithfulness, per_question: [...]}` |
| `scripts/ci_check.py` | Reads report, exits 1 if `mean_faithfulness < FAITHFULNESS_THRESHOLD` |

### Faithfulness algorithm

```
for each sample:
  run RAG pipeline → {answer, citations, retrieved_chunks}
  split answer into sentences
  for each sentence → LLM judge: "supported by context? YES/NO"
  score = supported / total

mean_faithfulness = avg across all samples
```

### Metrics

| Metric | Method | Threshold |
|--------|--------|-----------|
| Faithfulness | LLM judge per claim | ≥ 0.75 |
| Retrieval recall *(optional)* | reference chunk in top-k | ≥ 0.80 |
| Answer relevance *(optional)* | LLM judge | ≥ 0.70 |

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
│   └── ci_check.py         ← P3 NEW
├── data/
│   └── eval_dataset.json   ← P3 NEW
├── results/
│   └── eval_report.json    ← P3 written at runtime
├── build_index.py          ← P2 MODIFIED
├── query.py                ← P2 MODIFIED
├── bm25_index.pkl          ← P2 written at runtime
├── requirements.txt        ← updated each phase
└── IMPLEMENTATION_PLAN.md
```
