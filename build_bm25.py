"""
BM25 index builder — two separate steps to stay within low-memory constraints.

  Step 1  (--chunks-only)  : chunk every .md file → bm25_index/corpus.json
                              Saves after each file, so it is fully resumable.
  Step 2  (--index-only)   : read corpus.json → tokenise → fit → save BM25 files
  default (no flags)       : run both steps in sequence

Usage:
  python build_bm25.py               # full run
  python build_bm25.py --chunks-only # only (re)build corpus.json
  python build_bm25.py --index-only  # only (re)build BM25 from existing corpus.json
"""
import gc
import json
import sys
import time
from pathlib import Path

from src.chunker import chunk_text
from src.config import CHUNK_SIZE, CHUNK_OVERLAP, BM25_INDEX_PATH

CORPUS_FILE = Path(BM25_INDEX_PATH) / "corpus.json"


def _p(*args, **kwargs):
    print(*args, **kwargs, flush=True)


# ---------------------------------------------------------------------------
# Step 1 — chunking
# ---------------------------------------------------------------------------

def step1_build_corpus() -> int:
    """Chunk MD files one at a time; save corpus.json after every file."""
    data_dir = Path("data")
    md_files = sorted(data_dir.glob("*.md"))
    if not md_files:
        _p(f"ERROR: No .md files found in {data_dir}/")
        _p("Run `python build_index.py` first to cache chapters.")
        sys.exit(1)

    Path(BM25_INDEX_PATH).mkdir(exist_ok=True)

    # Resume: load whatever is already saved
    all_chunks: list[dict] = []
    done_sources: set[str] = set()
    if CORPUS_FILE.exists():
        with open(CORPUS_FILE, encoding="utf-8") as f:
            all_chunks = json.load(f)
        done_sources = {c["metadata"]["source"] for c in all_chunks}
        _p(f"Resuming — {len(done_sources)} files already done "
           f"({len(all_chunks)} chunks loaded)")

    _p(f"\n=== Step 1: Chunking ({len(md_files)} files) ===\n")

    new_files = 0
    for i, md_file in enumerate(md_files, 1):
        if md_file.name in done_sources:
            _p(f"[{i}/{len(md_files)}] skip  {md_file.name}")
            continue

        _p(f"[{i}/{len(md_files)}] chunk {md_file.name} ...", end=" ")
        content = md_file.read_text(encoding="utf-8")
        title = next(
            (line.lstrip("# ").strip() for line in content.splitlines() if line.startswith("#")),
            md_file.stem,
        )
        raw = chunk_text(content, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)

        for j, text in enumerate(raw):
            all_chunks.append({
                "id": f"{md_file.name}::{j}",
                "text": text,
                "metadata": {"source": md_file.name, "title": title, "chunk_index": j},
            })
        _p(f"{len(raw)} chunks")
        new_files += 1

        # Persist progress so a crash wastes at most one file
        with open(CORPUS_FILE, "w", encoding="utf-8") as f:
            json.dump(all_chunks, f)

        del content, raw
        gc.collect()
        time.sleep(0.05)

    _p(f"\nCorpus ready: {len(all_chunks)} chunks from {len(md_files)} files "
       f"({new_files} newly chunked)\n")
    return len(all_chunks)


# ---------------------------------------------------------------------------
# Step 2 — BM25 index
# ---------------------------------------------------------------------------

def step2_build_index():
    """Tokenise corpus.json and fit/save BM25 index. bm25s imported here only."""
    if not CORPUS_FILE.exists():
        _p(f"ERROR: {CORPUS_FILE} not found — run Step 1 first.")
        sys.exit(1)

    _p("=== Step 2: Building BM25 index ===\n")

    _p("Loading corpus.json ...", end=" ")
    with open(CORPUS_FILE, encoding="utf-8") as f:
        all_chunks = json.load(f)
    _p(f"{len(all_chunks)} chunks")

    # Import bm25s only now — keeps step 1 free of any bm25s side-effects
    import bm25s  # noqa: PLC0415

    texts = [c["text"] for c in all_chunks]
    _p(f"Tokenising {len(texts)} chunks (no stemmer, no network) ...")
    t0 = time.time()

    # stemmer=None  → no NLTK download, no heavy stemming
    # stopwords="en" → built-in list inside bm25s, no download
    corpus_tokens = bm25s.tokenize(texts, stopwords="en", stemmer=None)
    _p(f"  tokenised in {time.time() - t0:.1f}s")

    gc.collect()
    time.sleep(0.1)

    _p("Fitting BM25 ...")
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens)

    index_dir = Path(BM25_INDEX_PATH)
    retriever.save(str(index_dir))
    _p(f"\nDone in {time.time() - t0:.1f}s  →  index saved to {BM25_INDEX_PATH}/")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = set(sys.argv[1:])
    chunks_only = "--chunks-only" in args
    index_only  = "--index-only"  in args

    if chunks_only and index_only:
        _p("ERROR: --chunks-only and --index-only are mutually exclusive.")
        sys.exit(1)

    if not index_only:
        step1_build_corpus()

    if not chunks_only:
        step2_build_index()


if __name__ == "__main__":
    main()
