from src.config import RRF_K, N_CANDIDATES
from src.vectorstore import RustBookVectorStore
from src.bm25_store import BM25Store


class HybridRetriever:
    def __init__(self, vector_store: RustBookVectorStore, bm25_store: BM25Store):
        self._vs = vector_store
        self._bm25 = bm25_store

    def retrieve(self, question: str, n_candidates: int = N_CANDIDATES) -> list[dict]:
        vec_results = self._vs.query(question, n_results=n_candidates)
        bm25_results = self._bm25.query(question, n=n_candidates)

        def _key(chunk):
            return chunk["metadata"]["source"] + "::" + str(chunk["metadata"]["chunk_index"])

        vec_ranks = {_key(c): rank for rank, c in enumerate(vec_results)}
        bm25_ranks = {_key(c): rank for rank, c in enumerate(bm25_results)}

        all_chunks: dict[str, dict] = {}
        for c in vec_results + bm25_results:
            k = _key(c)
            if k not in all_chunks:
                all_chunks[k] = c

        for k, chunk in all_chunks.items():
            r_vec = vec_ranks.get(k, n_candidates)
            r_bm25 = bm25_ranks.get(k, n_candidates)
            chunk["rrf_score"] = 1 / (RRF_K + r_vec + 1) + 1 / (RRF_K + r_bm25 + 1)

        return sorted(all_chunks.values(), key=lambda c: c["rrf_score"], reverse=True)[:n_candidates]
