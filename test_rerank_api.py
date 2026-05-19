"""
Probe 2 — confirms BAAI/bge-reranker-base score direction and batch behaviour.
Run with:  python test_rerank_api.py
"""
import os, sys, requests

TOKEN = os.environ.get("HF_TOKEN", "")
if not TOKEN:
    sys.exit("ERROR: HF_TOKEN not set.")

HEADERS = {"Authorization": f"Bearer {TOKEN}"}
URL = "https://router.huggingface.co/hf-inference/models/BAAI/bge-reranker-base/pipeline/text-classification"

QUESTION = "what is ownership in Rust?"
PAIRS = [
    ("RELEVANT",   "Ownership is Rust's most unique feature and has deep implications for the rest of the language."),
    ("IRRELEVANT", "Chocolate cake is made with flour, sugar, cocoa powder, eggs, and butter."),
    ("PARTIAL",    "Rust uses a borrow checker to enforce memory safety at compile time."),
]

print(f"Question: {QUESTION!r}\n")

# Test 1: one pair at a time
print("=== Test 1: individual pairs ===")
for label, passage in PAIRS:
    r = requests.post(URL, headers=HEADERS, json={"inputs": [{"text": QUESTION, "text_pair": passage}]}, timeout=30)
    if r.status_code == 200:
        result = r.json()[0]  # first (only) pair result
        print(f"  {label:<12}  response={result}")
    else:
        print(f"  {label:<12}  ERROR {r.status_code}: {r.text[:100]}")

# Test 2: all pairs in one batch call
print("\n=== Test 2: batch (3 pairs in one call) ===")
batch_inputs = [{"text": QUESTION, "text_pair": p} for _, p in PAIRS]
r = requests.post(URL, headers=HEADERS, json={"inputs": batch_inputs}, timeout=30)
if r.status_code == 200:
    for (label, _), result in zip(PAIRS, r.json()):
        print(f"  {label:<12}  response={result}")
else:
    print(f"  ERROR {r.status_code}: {r.text[:200]}")

# Test 3: top_k=2 to see both labels
print("\n=== Test 3: top_k=2 (both labels for one pair) ===")
r = requests.post(URL, headers=HEADERS,
    json={"inputs": [{"text": QUESTION, "text_pair": PAIRS[0][1]}], "parameters": {"top_k": 2}},
    timeout=30)
if r.status_code == 200:
    print(f"  RELEVANT pair, both labels: {r.json()}")
else:
    print(f"  ERROR {r.status_code}: {r.text[:200]}")
