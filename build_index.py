import os
import sys
import time

from src.ingest import load_book
from src.chunker import chunk_text
from src.vectorstore import RustBookVectorStore, _hf_embed


def _fmt(seconds):
    if seconds < 60:
        return f"{seconds:.0f}s"
    return f"{int(seconds) // 60}m {int(seconds) % 60:02d}s"


def _p(*args, **kwargs):
    print(*args, **kwargs, flush=True)


def main():
    _p("=== Build Rust Book Vector Index ===\n")

    # Validate token before doing any real work
    token = os.environ.get("HUGGINGFACE_API_TOKEN", "")
    if not token:
        _p("ERROR: HUGGINGFACE_API_TOKEN is not set.")
        _p("  Get a free token at https://huggingface.co/settings/tokens")
        sys.exit(1)
    _p("Step 0: Validating HuggingFace token...")
    try:
        _hf_embed(["test"], token)
        _p("  Token OK — HF Inference API reachable.\n")
    except Exception as e:
        _p(f"  Token/API check FAILED: {e}")
        sys.exit(1)

    _p("Step 1: Loading chapters (cached in data/)...")
    chapters = load_book(cache_dir="data")
    total_chapters = len(chapters)
    _p(f"  {total_chapters} chapters loaded.\n")

    _p("Step 2: Initialising vector store...")
    store = RustBookVectorStore(persist_dir="chroma_db")
    existing = store.count()
    if existing > 0:
        _p(f"  Resuming — {existing} chunks already in index.\n")

    _p(f"Step 3: Chunking + embedding ({total_chapters} chapters)...\n")
    _p(f"  {'#':>3}  {'%':>5}  {'Chapter':<42}  {'Chunks':>6}  {'Chap t':>6}  {'Elapsed':>7}  {'ETA':>7}")
    _p("  " + "-" * 82)

    total_chunks = 0
    skipped = 0
    build_start = time.time()

    for idx, chapter in enumerate(chapters, 1):
        if store.has_source(chapter["filepath"]):
            skipped += 1
            pct = idx / total_chapters * 100
            _p(f"  {idx:>3}  {pct:>4.1f}%  {chapter['title'][:42]:<42}  {'skip':>6}")
            continue

        chap_start = time.time()

        chunks = chunk_text(chapter["content"], chunk_size=1500, overlap=350)
        chapter_chunks = [
            {
                "id": f"{chapter['filepath']}::{i}",
                "text": chunk,
                "metadata": {
                    "source": chapter["filepath"],
                    "title": chapter["title"],
                    "chunk_index": i,
                },
            }
            for i, chunk in enumerate(chunks)
        ]
        store.add_chunks(chapter_chunks)
        total_chunks += len(chapter_chunks)

        chap_time = time.time() - chap_start
        elapsed = time.time() - build_start
        done_so_far = idx - skipped
        remaining = total_chapters - idx
        eta = (elapsed / done_so_far) * remaining if done_so_far > 0 else 0
        pct = idx / total_chapters * 100
        title = chapter["title"][:42]

        _p(
            f"  {idx:>3}  {pct:>4.1f}%  {title:<42}  {len(chapter_chunks):>6}  "
            f"{_fmt(chap_time):>6}  {_fmt(elapsed):>7}  {_fmt(eta):>7}"
        )

    total_time = time.time() - build_start
    _p(f"\n  Done — {total_chunks} new chunks indexed in {_fmt(total_time)}.")
    if skipped:
        _p(f"  ({skipped} chapters were already indexed and skipped.)")


if __name__ == "__main__":
    main()
