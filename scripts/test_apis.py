"""
Pre-flight check for Phase 3 workflows.
Tests every external API call that evaluate.py and generate_dataset.py will make.

Run locally:  python scripts/test_apis.py
All tests must pass before triggering GitHub Actions.
"""
import json
import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Load .env (no dotenv dependency — plain key=value parse)
# ---------------------------------------------------------------------------

def _load_env():
    env_path = Path(".env")
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())

_load_env()

# Ensure project root is on sys.path so `src.*` imports work
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

HF_TOKEN      = os.environ.get("HF_TOKEN", "")
GROQ_API_KEY  = os.environ.get("GROQ_API_KEY", "")

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"

results = []


def check(name, fn):
    print(f"  {name} ... ", end="", flush=True)
    try:
        status, detail = fn()
        ok = (status == 200)
        tag = PASS if ok else FAIL
        print(f"{tag}  [{status}] {detail}")
        results.append((name, ok))
    except Exception as e:
        print(f"{FAIL}  [ERR] {e}")
        results.append((name, False))


# ---------------------------------------------------------------------------
# 1. Groq — simple chat completion
# ---------------------------------------------------------------------------

def test_groq():
    import requests
    resp = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
        json={
            "model": "llama-3.3-70b-versatile",
            "messages": [{"role": "user", "content": "Reply with the single word: ok"}],
            "max_tokens": 5,
        },
        timeout=30,
    )
    reply = resp.json().get("choices", [{}])[0].get("message", {}).get("content", "").strip()
    return resp.status_code, f'reply="{reply}"'


# ---------------------------------------------------------------------------
# 2. HF Inference API — embeddings (all-MiniLM-L6-v2)
# ---------------------------------------------------------------------------

def test_hf_embed():
    import requests
    import time
    url = "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction"
    for _ in range(3):
        resp = requests.post(
            url,
            headers={"Authorization": f"Bearer {HF_TOKEN}"},
            json={"inputs": ["test embedding"]},
            timeout=60,
        )
        if resp.status_code == 503:
            wait = resp.json().get("estimated_time", 20)
            print(f"\n    model loading, waiting {wait:.0f}s ...", end="", flush=True)
            time.sleep(wait)
            continue
        break
    vec = resp.json()
    dim = len(vec[0]) if isinstance(vec, list) and vec else "?"
    return resp.status_code, f"embedding dim={dim}"


# ---------------------------------------------------------------------------
# 3. HF Inference API — reranker (BAAI/bge-reranker-base)
# ---------------------------------------------------------------------------

def test_hf_reranker():
    import requests
    import time
    url = "https://router.huggingface.co/hf-inference/models/BAAI/bge-reranker-base/pipeline/text-classification"
    for _ in range(3):
        resp = requests.post(
            url,
            headers={"Authorization": f"Bearer {HF_TOKEN}"},
            json={"inputs": [{"text": "what is ownership?", "text_pair": "Ownership is Rust's memory model."}]},
            timeout=60,
        )
        if resp.status_code == 503:
            wait = resp.json().get("estimated_time", 20)
            print(f"\n    model loading, waiting {wait:.0f}s ...", end="", flush=True)
            time.sleep(wait)
            continue
        break
    data = resp.json()
    score = data[0][0]["score"] if isinstance(data, list) and data else "?"
    return resp.status_code, f"rerank score={score}"


# ---------------------------------------------------------------------------
# 4. RAGAS — llm_factory with Groq (OpenAI-compatible)
# ---------------------------------------------------------------------------

def test_ragas_llm():
    from openai import OpenAI
    from ragas.llms import llm_factory

    client = OpenAI(
        api_key=GROQ_API_KEY,
        base_url="https://api.groq.com/openai/v1",
    )
    llm = llm_factory("llama-3.3-70b-versatile", client=client)
    # generate() is the RAGAS internal call — use the underlying client directly
    resp = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": "Reply with one word: ok"}],
        max_tokens=5,
    )
    reply = resp.choices[0].message.content.strip()
    return 200, f'llm_factory ready  reply="{reply}"'


# ---------------------------------------------------------------------------
# 5. RAGAS — HuggingFaceEmbeddings with use_api=True
# ---------------------------------------------------------------------------

def test_ragas_embeddings():
    from ragas.embeddings import HuggingFaceEmbeddings

    emb = HuggingFaceEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2",
        use_api=True,
        api_key=HF_TOKEN,
    )
    vec = emb.embed_text("test")
    return 200, f"embedding dim={len(vec)}"


# ---------------------------------------------------------------------------
# 6. RAGAS evaluate() end-to-end with 2 synthetic samples
# ---------------------------------------------------------------------------

def test_ragas_evaluate():
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="ragas")
    from ragas import evaluate, EvaluationDataset
    from ragas.metrics import Faithfulness, ContextRecall
    from ragas.llms import LangchainLLMWrapper
    from langchain_groq import ChatGroq

    llm = LangchainLLMWrapper(ChatGroq(model="llama-3.3-70b-versatile", api_key=GROQ_API_KEY))

    dataset = EvaluationDataset.from_list([
        {
            "user_input": "What is ownership in Rust?",
            "response": "Ownership is Rust's memory management system. Each value has a single owner [1].",
            "retrieved_contexts": [
                "Ownership is a set of rules that govern how a Rust program manages memory. "
                "Every value in Rust has an owner. There can only be one owner at a time."
            ],
            "reference": "Ownership is Rust's approach to memory management without a garbage collector.",
        },
        {
            "user_input": "What does the borrow checker do?",
            "response": "The borrow checker enforces ownership rules at compile time [1].",
            "retrieved_contexts": [
                "Rust's borrow checker validates references at compile time to ensure memory safety."
            ],
            "reference": "The borrow checker validates references to ensure they are always valid.",
        },
    ])

    # Faithfulness + ContextRecall only — no n>1 calls, no embeddings needed
    metrics = [Faithfulness(llm=llm), ContextRecall(llm=llm)]
    result = evaluate(dataset, metrics=metrics)
    df = result.to_pandas()

    faith = float(df["faithfulness"].mean())
    recall = float(df["context_recall"].mean())
    return 200, f"faithfulness={faith:.2f}  context_recall={recall:.2f}"


# ---------------------------------------------------------------------------
# 7. generate_dataset.py JSON parsing — one real Groq call
# ---------------------------------------------------------------------------

def test_groq_json_parsing():
    import requests

    prompt = (
        'Read this excerpt from a Rust book:\n'
        '"Ownership is a set of rules that govern how a Rust program manages memory."\n\n'
        'Write ONE question answerable from this excerpt and a 1-2 sentence reference answer.\n'
        'Respond with valid JSON only, no other text:\n'
        '{"question": "...", "reference_answer": "..."}'
    )
    resp = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
        json={
            "model": "llama-3.3-70b-versatile",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 200,
            "temperature": 0.3,
        },
        timeout=30,
    )
    body = resp.json()
    if "choices" not in body:
        raise ValueError(f"Unexpected response (status {resp.status_code}): {body}")
    raw = body["choices"][0]["message"]["content"].strip()
    # Strip markdown fences if present (same logic as generate_dataset.py)
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    parsed = json.loads(raw.strip())
    q = parsed.get("question", "")
    a = parsed.get("reference_answer", "")
    if not q or not a:
        raise ValueError(f"Missing keys in response: {parsed}")
    return resp.status_code, f'parsed OK  question="{q[:50]}..."'


# ---------------------------------------------------------------------------
# 8. Full pipeline smoke test — 1 question end-to-end
# ---------------------------------------------------------------------------

def test_pipeline_smoke():
    import sys as _sys
    from pathlib import Path as _Path

    bm25_path = "bm25_index"
    chroma_path = "chroma_db"
    if not _Path(bm25_path).exists():
        return 200, "SKIP — bm25_index/ not found locally (will exist in CI)"
    if not _Path(chroma_path).exists():
        return 200, "SKIP — chroma_db/ not found locally (will exist in CI)"

    # Windows event loop fix before importing chromadb
    import asyncio
    if _sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    from src.vectorstore import RustBookVectorStore
    from src.bm25_store import BM25Store
    from src.hybrid_retriever import HybridRetriever
    from src.reranker import Reranker
    from src.generator import answer
    from src.config import N_CANDIDATES, TOP_K

    store = RustBookVectorStore(persist_dir=chroma_path)
    if store.count() == 0:
        return 200, "SKIP — vector index empty"

    bm25 = BM25Store()
    bm25.load(bm25_path)
    retriever = HybridRetriever(store, bm25)
    reranker = Reranker()

    candidates = retriever.retrieve("What is ownership in Rust?", n_candidates=N_CANDIDATES)
    top_chunks = reranker.rerank("What is ownership in Rust?", candidates, top_k=TOP_K)
    result = answer("What is ownership in Rust?", top_chunks)

    ans_preview = result["answer"][:60].replace("\n", " ")
    n_citations = len(result["citations"])
    return 200, f'answer="{ans_preview}..."  citations={n_citations}'


# ---------------------------------------------------------------------------
# Run all checks
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not HF_TOKEN:
        print("ERROR: HF_TOKEN not set")
        sys.exit(1)
    if not GROQ_API_KEY:
        print("ERROR: GROQ_API_KEY not set")
        sys.exit(1)

    print("\n=== Phase 3 API pre-flight check ===\n")

    # Warm up DNS for router.huggingface.co before timed checks.
    # On Windows, first cold lookup can fail inside requests even though nslookup
    # resolves correctly — the recursive lookup races the TCP connect. Subsequent
    # calls to the same host succeed because the OS DNS cache is now populated.
    import socket
    try:
        socket.getaddrinfo("router.huggingface.co", 443)
    except Exception:
        pass

    print("[ Direct API calls ]")
    check("Groq chat completion         ", test_groq)
    check("HF embed  (all-MiniLM-L6-v2)", test_hf_embed)
    check("HF rerank (bge-reranker-base)", test_hf_reranker)

    print("\n[ RAGAS wrappers ]")
    check("RAGAS llm_factory + Groq     ", test_ragas_llm)
    check("RAGAS HuggingFaceEmbeddings  ", test_ragas_embeddings)

    print("\n[ Integration ]")
    check("RAGAS evaluate() end-to-end  ", test_ragas_evaluate)
    check("generate_dataset JSON parse  ", test_groq_json_parsing)
    check("Full pipeline smoke (1 query)", test_pipeline_smoke)

    passed = sum(1 for _, ok in results if ok)
    total  = len(results)
    print(f"\n{'='*40}")
    print(f"  {passed}/{total} passed")

    if passed < total:
        failed = [name for name, ok in results if not ok]
        print(f"  Failed: {', '.join(f.strip() for f in failed)}")
        sys.exit(1)
    else:
        print("  All checks passed — safe to trigger workflows.")
    print()
