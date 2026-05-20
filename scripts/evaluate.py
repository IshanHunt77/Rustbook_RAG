"""
Run full RAG pipeline on every sample in eval_dataset.json, then evaluate
with RAGAS (Faithfulness, AnswerRelevancy, ContextRecall) using Groq as the
judge LLM and HF Inference API for embeddings.

Output: results/eval_report.json

Run via GitHub Actions (evaluate.yml) — requires GROQ_API_KEY + HF_TOKEN.
"""
import asyncio
import json
import os
import sys
import time
import warnings
from pathlib import Path

# Ensure project root is on sys.path so `src.*` imports work when running
# from scripts/ or from GitHub Actions working directory
sys.path.insert(0, str(Path(__file__).parent.parent))

warnings.filterwarnings("ignore", category=DeprecationWarning)

# ChromaDB requires SelectorEventLoop on Windows
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from ragas import evaluate, EvaluationDataset
from ragas.run_config import RunConfig
from ragas.metrics import Faithfulness, ContextRecall
from ragas.llms import LangchainLLMWrapper
from langchain_groq import ChatGroq

from src.vectorstore import RustBookVectorStore
from src.bm25_store import BM25Store
from src.hybrid_retriever import HybridRetriever
from src.reranker import Reranker
from src.generator import answer
from src.config import N_CANDIDATES, TOP_K, BM25_INDEX_PATH, FAITHFULNESS_THRESHOLD

EVAL_DATASET_PATH = "data/eval_dataset.json"
REPORT_PATH = "results/eval_report.json"
PIPELINE_SLEEP = 2   # seconds between pipeline calls (Groq + HF rate limits)


def _build_ragas_llm():
    return LangchainLLMWrapper(
        ChatGroq(model="llama-3.3-70b-versatile", api_key=os.environ["GROQ_API_KEY"])
    )




def _run_pipeline(retriever, reranker, sample):
    candidates = retriever.retrieve(sample["question"], n_candidates=N_CANDIDATES)
    top_chunks = reranker.rerank(sample["question"], candidates, top_k=TOP_K)
    result = answer(sample["question"], top_chunks)
    return result["answer"], [c["text"] for c in top_chunks]


def main():
    if not Path(EVAL_DATASET_PATH).exists():
        print(f"ERROR: {EVAL_DATASET_PATH} not found — run generate_dataset.py first.")
        sys.exit(1)

    with open(EVAL_DATASET_PATH, encoding="utf-8") as f:
        samples = json.load(f)[:15]
    print(f"Loaded {len(samples)} eval samples", flush=True)

    print("Loading pipeline components...", flush=True)
    store = RustBookVectorStore(persist_dir="chroma_db")
    if store.count() == 0:
        print("ERROR: Vector index is empty — run build_index.py first.")
        sys.exit(1)

    bm25 = BM25Store()
    bm25.load(BM25_INDEX_PATH)
    retriever = HybridRetriever(store, bm25)
    reranker = Reranker()
    print("Pipeline ready\n", flush=True)

    ragas_rows = []
    skipped = 0

    for i, sample in enumerate(samples):
        print(f"[{i+1}/{len(samples)}] {sample['id']} ...", flush=True)
        try:
            ans_text, ctx_chunks = _run_pipeline(retriever, reranker, sample)
            ragas_rows.append({
                "user_input": sample["question"],
                "response": ans_text,
                "retrieved_contexts": ctx_chunks,
                "reference": sample["reference_answer"],
            })
        except Exception as e:
            print(f"  SKIP: {e}", flush=True)
            skipped += 1

        time.sleep(PIPELINE_SLEEP)

    print(f"\nPipeline done — {len(ragas_rows)} succeeded, {skipped} skipped", flush=True)

    if not ragas_rows:
        print("ERROR: No samples to evaluate.")
        sys.exit(1)

    print("Running RAGAS evaluation...", flush=True)
    ragas_llm = _build_ragas_llm()

    metrics = [Faithfulness(llm=ragas_llm), ContextRecall(llm=ragas_llm)]

    dataset = EvaluationDataset.from_list(ragas_rows)
    run_config = RunConfig(timeout=180, max_retries=3, max_workers=1)
    result = evaluate(dataset, metrics=metrics, run_config=run_config)

    df = result.to_pandas()
    mean_faith = float(df["faithfulness"].mean())
    mean_recall = float(df["context_recall"].mean())

    report = {
        "mean_faithfulness": mean_faith,
        "mean_context_recall": mean_recall,
        "faithfulness_threshold": FAITHFULNESS_THRESHOLD,
        "n_samples": len(ragas_rows),
        "n_skipped": skipped,
        "passed": mean_faith >= FAITHFULNESS_THRESHOLD,
        "per_question": df.to_dict(orient="records"),
    }

    Path(REPORT_PATH).parent.mkdir(exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\n--- Eval Report ---")
    print(f"  faithfulness:     {mean_faith:.3f}  (threshold: {FAITHFULNESS_THRESHOLD})")
    print(f"  context_recall:   {mean_recall:.3f}")
    print(f"  PASSED: {report['passed']}")
    print(f"\nReport written to {REPORT_PATH}", flush=True)


if __name__ == "__main__":
    main()
