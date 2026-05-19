# Models
EMBED_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL = "BAAI/bge-reranker-base"
LLM_MODEL    = "llama-3.3-70b-versatile"

# Chunking
CHUNK_SIZE    = 1500
CHUNK_OVERLAP = 350

# Retrieval
N_CANDIDATES = 20
TOP_K        = 5
RRF_K        = 60

# Paths
BM25_INDEX_PATH = "bm25_index"
CHROMA_DIR      = "chroma_db"

# Prompts
SYSTEM_PROMPT = """You are an expert assistant on The Rust Programming Language book.
Answer ONLY using the provided context chunks.
Cite every factual claim with [N] where N is the chunk number shown in the header.
If the answer is not in the context, say: "I don't know based on the Rust Book.\""""

CONTEXT_TEMPLATE = "--- Chunk {index} | {title} ({source}) ---\n{text}"

# Phase 3 quality gate
FAITHFULNESS_THRESHOLD = 0.75
