import os
import time
import requests

from src.config import TOP_K

_URL = "https://router.huggingface.co/hf-inference/models/BAAI/bge-reranker-base/pipeline/text-classification"


class Reranker:
    def __init__(self):
        self._token = os.environ.get("HF_TOKEN", "")
        if not self._token:
            raise EnvironmentError("HF_TOKEN not set")
        self._headers = {"Authorization": f"Bearer {self._token}"}

    def _score_batch(self, pairs: list[dict]) -> list[float]:
        """Score one batch of pairs, return a float per pair."""
        for attempt in range(3):
            resp = requests.post(
                _URL,
                headers=self._headers,
                json={"inputs": pairs},
                timeout=60,
            )
            if resp.status_code == 503:
                wait = resp.json().get("estimated_time", 20)
                print(f"  Reranker loading on HF, waiting {wait:.0f}s...", flush=True)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            break
        else:
            raise RuntimeError("Reranker unavailable after 3 retries (503)")

        out = []
        for item in resp.json():
            entry = item[0] if isinstance(item, list) else item
            out.append(entry["score"])
        return out

    def rerank(self, question: str, candidates: list[dict], top_k: int = TOP_K) -> list[dict]:
        # Initialize scores to 0 so sort never fails if a batch returns fewer results
        for c in candidates:
            c["rerank_score"] = 0.0

        # Score in batches of 5 to stay within HF API limits
        batch_size = 5
        for i in range(0, len(candidates), batch_size):
            batch = candidates[i : i + batch_size]
            pairs = [{"text": question, "text_pair": c["text"]} for c in batch]
            scores = self._score_batch(pairs)
            for candidate, score in zip(batch, scores):
                candidate["rerank_score"] = score

        return sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)[:top_k]
