# build_index.py Debugging Log

## Project Overview
- **Dir:** `d:\rag_ml`
- **Script:** `build_index.py` — loads 111 Rust Book chapters from `data/`, chunks them (1500 chars, 350 overlap), embeds via HuggingFace API, stores in ChromaDB
- **Venv:** `.venv\Scripts\python.exe`

---

## Issue 1: Silent 0-byte Output (Background Process)

**Cause:** Python buffers `print()` when stdout is redirected to a file. Process crashed before flushing.

**Fix:** Replaced all `print()` with `_p()` (a wrapper with `flush=True`), and use `python -u` flag.

---

## Issue 2: HuggingFace API 404

**Cause:** Old endpoint `https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2` returns 404 — HF changed their API.

**Fix:** Updated `src/vectorstore.py` to use new router endpoint:
```
https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction
```

> **Note:** We use the HF Inference API (not local sentence-transformers) because local inference was too heavy for this laptop.

---

## Issue 3: Partial Chunk Resume Bug

**Cause:** `has_source()` used `limit=1` — if even 1 chunk was stored before a crash, the whole chapter got skipped, leaving incomplete data.

**Fix:** Now uses a **sentinel record** (`filepath::done`) written only after ALL chunks succeed. `has_source()` checks for the sentinel. `add_chunks()` also deletes any partial data before re-writing.

---

## Issue 4: ChromaDB Telemetry Hang

**Cause:** `chromadb.PersistentClient()` hangs indefinitely on first init because ChromaDB 1.5.7 makes a telemetry network call that never returns.

**Fix:** Pass `settings=Settings(anonymized_telemetry=False)` to `PersistentClient`, plus set env vars `ANONYMIZED_TELEMETRY=False` and `CHROMA_TELEMETRY=false`.

---

## Issue 5: ChromaDB Slow Init (Mistaken for Hang)

**Cause:** ChromaDB 1.5.7 takes ~20–30 seconds to initialize on first run (loading HNSW library, setting up SQLite). The process was being killed prematurely thinking it was hung.

**Resolution:** It is NOT hung — just slow. After the first init, `chroma.sqlite3` is created (`188416 bytes`) and subsequent inits are fast.

> **Wait at least 30–40 seconds after "Step 2: Initialising vector store..." before concluding it's stuck.**

---

## Current State

- `chroma_db/chroma.sqlite3` exists (188416 bytes) — DB initialized, empty, ready
- 0 chunks indexed — clean slate
- All code fixes applied, ready to run

---

## Correct Run Command (PowerShell)

```powershell
$env:HUGGINGFACE_API_TOKEN="<redacted>"; $env:ANONYMIZED_TELEMETRY="False"; $env:CHROMA_TELEMETRY="false"; .venv\Scripts\python.exe -u build_index.py
```

---

## Key Files Changed

| File | Change |
|------|--------|
| `build_index.py` | `flush=True` on all prints, token validation at startup |
| `src/vectorstore.py` | New HF router URL, sentinel-based resume, `Settings(anonymized_telemetry=False)`, response shape fix for new API |
