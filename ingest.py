#!/usr/bin/env python3
"""
Ingest Rust Book markdown files into ChromaDB using HF Inference API embeddings.

Reads .md files from DATA_DIR (or RUST_BOOK_DIR env var), chunks them with
langchain's MarkdownTextSplitter, embeds via HF Inference API in batches of 32,
and stores results in a persistent ChromaDB collection.

Resume-safe: chunks already stored (by deterministic ID) are skipped on re-run.
"""

import asyncio
import os
import sys
import time
from pathlib import Path

# Must come before importing chromadb on Windows — avoids ProactorEventLoop hang.
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import requests
import chromadb
from chromadb.config import Settings

# Suppress chromadb telemetry before init — without these the client hangs
# indefinitely on first run making a network call that never returns.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("CHROMA_TELEMETRY", "false")

try:
    from langchain_text_splitters import MarkdownTextSplitter
except ImportError:
    from langchain.text_splitter import MarkdownTextSplitter  # type: ignore[no-redef]

# ── Configuration ──────────────────────────────────────────────────────────────
DATA_DIR   = Path(os.environ.get("RUST_BOOK_DIR", r"d:\rag_ml\data"))
CHROMA_DIR = Path(os.environ.get("CHROMA_DIR", "./chroma_db"))
COLLECTION = "rust_book"

HF_MODEL   = "sentence-transformers/all-MiniLM-L6-v2"
# HF changed their API — old api-inference.huggingface.co endpoint returns 404.
HF_API_URL = f"https://router.huggingface.co/hf-inference/models/{HF_MODEL}/pipeline/feature-extraction"

CHUNK_SIZE    = 500
CHUNK_OVERLAP = 50
BATCH_SIZE    = 32
# ──────────────────────────────────────────────────────────────────────────────


def _get_token() -> str:
    tok = os.environ.get("HF_TOKEN", "")
    if not tok:
        sys.exit(
            "Error: HF_TOKEN environment variable is not set.\n"
            "  Get a free token at https://huggingface.co/settings/tokens\n"
            "  Then run:  set HF_TOKEN=hf_..."
        )
    return tok


def _embed(
    session: requests.Session,
    texts: list[str],
    token: str,
) -> list[list[float]]:
    """Embed a batch of texts via HF Inference API with retry logic."""
    headers = {"Authorization": f"Bearer {token}"}
    payload = {"inputs": texts, "options": {"wait_for_model": True}}

    for attempt in range(6):
        resp = session.post(HF_API_URL, headers=headers, json=payload, timeout=60)

        if resp.status_code == 200:
            return _pool_if_needed(resp.json())

        if resp.status_code == 503:
            # Model is still loading on HF side — exponential backoff
            wait = min(2 ** attempt, 60)
            print(f"    [503] Model warming up — retrying in {wait}s…", flush=True)
            time.sleep(wait)
            continue

        if resp.status_code == 429:
            # Rate limited — wait longer each time
            wait = 30 * (attempt + 1)
            print(f"    [429] Rate limited — retrying in {wait}s…", flush=True)
            time.sleep(wait)
            continue

        resp.raise_for_status()

    raise RuntimeError("HF Inference API unavailable after 6 retries.")


def _pool_if_needed(raw: list) -> list[list[float]]:
    """
    Some models return token-level embeddings [batch, seq_len, dim].
    Mean-pool over the sequence dimension to get sentence embeddings [batch, dim].
    """
    if raw and isinstance(raw[0][0], list):
        return [
            [sum(tok[d] for tok in seq) / len(seq) for d in range(len(seq[0]))]
            for seq in raw
        ]
    return raw


def main() -> None:
    # Force line-buffered stdout so output isn't lost if the process crashes.
    sys.stdout.reconfigure(line_buffering=True)

    token = _get_token()

    md_files = sorted(DATA_DIR.glob("**/*.md"))
    if not md_files:
        sys.exit(f"No .md files found under {DATA_DIR}")
    print(f"Found {len(md_files)} markdown file(s) under {DATA_DIR}\n")

    # embedding_function=None is critical — prevents chromadb from loading any
    # local onnxruntime model, which hangs on low-spec hardware.
    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )
    collection = client.get_or_create_collection(
        name=COLLECTION,
        embedding_function=None,
        metadata={"hnsw:space": "cosine"},
    )

    # Preload existing IDs so we can skip already-ingested chunks on re-run.
    existing: set[str] = set(collection.get(include=[])["ids"])
    print(f"Chunks already in DB : {len(existing)}\n")

    splitter  = MarkdownTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    session   = requests.Session()
    run_total = 0

    for file_num, path in enumerate(md_files, 1):
        label = str(path.relative_to(DATA_DIR)).replace("\\", "/")
        print(f"[{file_num}/{len(md_files)}] {label}")

        text   = path.read_text(encoding="utf-8", errors="replace")
        chunks = splitter.split_text(text)
        if not chunks:
            print("  (no chunks generated — skipped)\n")
            continue

        # Build deterministic IDs and filter to only new chunks
        pairs = [
            (f"{label}_{i}", ch)
            for i, ch in enumerate(chunks)
            if f"{label}_{i}" not in existing
        ]

        n_skipped = len(chunks) - len(pairs)
        if n_skipped:
            print(f"  Skipping {n_skipped} already-ingested chunk(s)")
        if not pairs:
            print("  All chunks already in DB\n")
            continue

        n_batches = (len(pairs) + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"  {len(pairs)} new chunk(s) → {n_batches} batch(es)")

        for b in range(n_batches):
            batch      = pairs[b * BATCH_SIZE : (b + 1) * BATCH_SIZE]
            batch_ids  = [p[0] for p in batch]
            batch_docs = [p[1] for p in batch]

            print(f"  Batch {b + 1}/{n_batches} ({len(batch)} chunks) … ", end="", flush=True)
            embeddings = _embed(session, batch_docs, token)

            collection.add(
                ids        = batch_ids,
                embeddings = embeddings,
                documents  = batch_docs,
                metadatas  = [
                    {"source": label, "chunk_index": b * BATCH_SIZE + i}
                    for i in range(len(batch))
                ],
            )
            existing.update(batch_ids)
            run_total += len(batch)
            print("done")

        print()

    print("─" * 50)
    print(f"Ingestion complete.")
    print(f"  Added this run : {run_total}")
    print(f"  Total in DB    : {collection.count()}")


if __name__ == "__main__":
    main()
