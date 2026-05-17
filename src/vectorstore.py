import asyncio
import os
import sys
import time
import numpy as np
import requests

# ChromaDB 1.x uses an embedded async server; on Windows the default ProactorEventLoop
# causes it to hang indefinitely. Force SelectorEventLoop before importing chromadb.
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import chromadb
from chromadb.config import Settings

COLLECTION_NAME = "rust_book"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_HF_API_URL = f"https://router.huggingface.co/hf-inference/models/{EMBED_MODEL}/pipeline/feature-extraction"


def _hf_embed(texts, token, retries=3):
    headers = {"Authorization": f"Bearer {token}"}
    for attempt in range(retries):
        resp = requests.post(
            _HF_API_URL,
            headers=headers,
            json={"inputs": texts},
            timeout=60,
        )
        if resp.status_code == 503:
            wait = resp.json().get("estimated_time", 20)
            print(f"  Model loading on HF, waiting {wait:.0f}s...", flush=True)
            time.sleep(wait)
            continue
        resp.raise_for_status()
        arr = np.array(resp.json())
        # new router returns sentence embeddings directly (ndim 1 for single, 2 for batch)
        if arr.ndim == 1:
            arr = arr[np.newaxis, :]  # single text → shape (1, dim)
        elif arr.ndim == 3:
            arr = arr.mean(axis=1)   # old token-level fallback
        return arr.tolist()
    raise RuntimeError("HuggingFace API unavailable after retries")


class RustBookVectorStore:
    def __init__(self, persist_dir="chroma_db"):
        self._token = os.environ.get("HF_TOKEN", "")
        if not self._token:
            raise EnvironmentError(
                "Set the HUGGINGFACE_API_TOKEN environment variable before running.\n"
                "  Get a free token at https://huggingface.co/settings/tokens"
            )
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def has_source(self, filepath):
        # Only True if the chapter completed fully (sentinel record present).
        # A chapter with partial chunks (crash mid-way) has no sentinel → re-indexed.
        sentinel_id = f"{filepath}::done"
        results = self.collection.get(ids=[sentinel_id])
        return len(results["ids"]) > 0

    def _delete_partial(self, filepath):
        results = self.collection.get(where={"source": filepath})
        if results["ids"]:
            self.collection.delete(ids=results["ids"])

    def add_chunks(self, chunks, batch_size=16):
        if not chunks:
            return
        filepath = chunks[0]["metadata"]["source"]
        # Remove any partial data from a previous crashed run before re-writing
        self._delete_partial(filepath)
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            texts = [c["text"] for c in batch]
            embeddings = _hf_embed(texts, self._token)
            self.collection.add(
                ids=[c["id"] for c in batch],
                embeddings=embeddings,
                documents=texts,
                metadatas=[c["metadata"] for c in batch],
            )
            time.sleep(0.2)
        # Write sentinel only after all chunks succeed
        sentinel_embed = _hf_embed(["done"], self._token)[0]
        self.collection.add(
            ids=[f"{filepath}::done"],
            embeddings=[sentinel_embed],
            documents=["__done__"],
            metadatas=[{"source": filepath, "chunk_index": -1}],
        )

    def query(self, question, n_results=5):
        embedding = _hf_embed([question], self._token)[0]
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )
        return [
            {
                "text": doc,
                "metadata": meta,
                "score": 1 - dist,
            }
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
        ]

    def count(self):
        return self.collection.count()
