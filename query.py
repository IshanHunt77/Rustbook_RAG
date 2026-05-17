from src.vectorstore import RustBookVectorStore
from src.generator import answer


def main():
    store = RustBookVectorStore(persist_dir="chroma_db")
    count = store.count()
    if count == 0:
        print("Index is empty. Run `python build_index.py` first.")
        return

    print(f"Rust Book RAG — {count} chunks indexed. Type 'quit' to exit.\n")

    while True:
        question = input("Question: ").strip()
        if question.lower() in ("quit", "exit", "q"):
            break
        if not question:
            continue

        chunks = store.query(question, n_results=5)
        print("\nGenerating answer...\n")
        response = answer(question, chunks)
        print(f"Answer:\n{response}\n")

        print("Sources:")
        for i, chunk in enumerate(chunks, 1):
            print(f"  [{i}] {chunk['metadata'].get('title', chunk['metadata']['source'])} — {chunk['metadata']['source']}")
        print()


if __name__ == "__main__":
    main()
