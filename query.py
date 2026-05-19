import sys
from pathlib import Path

from src.vectorstore import RustBookVectorStore
from src.bm25_store import BM25Store
from src.hybrid_retriever import HybridRetriever
from src.reranker import Reranker
from src.generator import answer
from src.config import N_CANDIDATES, TOP_K, BM25_INDEX_PATH


def main():
    if not Path(BM25_INDEX_PATH).exists():
        print(f"ERROR: {BM25_INDEX_PATH} not found. Run `python build_bm25.py` first.")
        sys.exit(1)

    print("Loading components...", flush=True)
    store = RustBookVectorStore(persist_dir="chroma_db")
    if store.count() == 0:
        print("Vector index is empty. Run `python build_index.py` first.")
        sys.exit(1)

    bm25 = BM25Store()
    bm25.load(BM25_INDEX_PATH)

    retriever = HybridRetriever(store, bm25)
    reranker = Reranker()

    print(f"Rust Book RAG — {store.count()} chunks indexed. Type 'quit' to exit.\n")

    while True:
        question = input("Question: ").strip()
        if question.lower() in ("quit", "exit", "q"):
            break
        if not question:
            continue

        candidates = retriever.retrieve(question, n_candidates=N_CANDIDATES)
        top_chunks = reranker.rerank(question, candidates, top_k=TOP_K)
        result = answer(question, top_chunks)

        print(f"\nAnswer:\n{result['answer']}\n")

        if result["citations"]:
            print("Sources:")
            for c in result["citations"]:
                print(f"  [{c['index']}] {c['title']} — {c['source']}")
        print()


if __name__ == "__main__":
    main()
