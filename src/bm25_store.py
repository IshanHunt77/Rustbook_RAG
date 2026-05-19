import gc
import json
import time
import bm25s
from pathlib import Path

from src.config import BM25_INDEX_PATH

# Must match the params used in build_bm25.py step2 so query tokens align with index tokens
_TOK = {"stopwords": "en", "stemmer": None}


class BM25Store:
    def __init__(self):
        self._retriever = None
        self._corpus = None

    def build(self, chunks: list[dict], path: str = BM25_INDEX_PATH):
        texts = [c["text"] for c in chunks]
        print(f"  Tokenising {len(texts)} chunks...", flush=True)
        corpus_tokens = bm25s.tokenize(texts, **_TOK)
        time.sleep(0.2)
        gc.collect()
        print("  Fitting BM25...", flush=True)
        self._retriever = bm25s.BM25()
        self._retriever.index(corpus_tokens)
        self._corpus = chunks

        index_dir = Path(path)
        index_dir.mkdir(exist_ok=True)
        self._retriever.save(str(index_dir))
        with open(index_dir / "corpus.json", "w", encoding="utf-8") as f:
            json.dump(chunks, f)
        print(f"BM25 index saved to {path}/ ({len(chunks)} chunks)", flush=True)

    def load(self, path: str = BM25_INDEX_PATH):
        index_dir = Path(path)
        self._retriever = bm25s.BM25.load(str(index_dir))
        with open(index_dir / "corpus.json", encoding="utf-8") as f:
            self._corpus = json.load(f)

    def query(self, question: str, n: int) -> list[dict]:
        query_tokens = bm25s.tokenize([question], **_TOK)
        results, scores = self._retriever.retrieve(query_tokens, k=min(n, len(self._corpus)))
        return [
            {
                "text": self._corpus[int(idx)]["text"],
                "metadata": self._corpus[int(idx)]["metadata"],
                "bm25_score": float(scores[0][i]),
            }
            for i, idx in enumerate(results[0])
        ]
