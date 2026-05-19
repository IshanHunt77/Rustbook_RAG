"""
One-time script: sample chunks from corpus, ask Groq to generate Q/A pairs.
Output: data/eval_dataset.json (~200 entries)

Run via GitHub Actions (generate_dataset.yml) — requires GROQ_API_KEY.
Resume-safe: skips chunks already written to the output file.
"""
import json
import os
import random
import time
from pathlib import Path

from groq import Groq

CORPUS_PATH = "bm25_index/corpus.json"
OUTPUT_PATH = "data/eval_dataset.json"
TARGET_SAMPLES = 200
MIN_CHUNK_LEN = 400
SEED = 42

_PROMPT = """\
Read the following excerpt from The Rust Programming Language book.
Write ONE specific question that can be answered using ONLY this excerpt.
Then write a concise reference answer (2-4 sentences) based solely on this excerpt.

Respond with valid JSON only, no other text:
{{"question": "...", "reference_answer": "..."}}

Excerpt:
{text}"""


def _load_corpus():
    with open(CORPUS_PATH, encoding="utf-8") as f:
        return json.load(f)


def _sample_chunks(corpus):
    by_source = {}
    for chunk in corpus:
        if len(chunk["text"]) < MIN_CHUNK_LEN:
            continue
        by_source.setdefault(chunk["metadata"]["source"], []).append(chunk)

    random.seed(SEED)
    pool = []
    for chunks in by_source.values():
        random.shuffle(chunks)
        pool.extend(chunks)

    random.shuffle(pool)
    return pool[:TARGET_SAMPLES]


def main():
    if not Path(CORPUS_PATH).exists():
        raise FileNotFoundError(f"{CORPUS_PATH} not found — run build_bm25.py first")

    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    corpus = _load_corpus()
    sampled = _sample_chunks(corpus)
    print(f"Corpus: {len(corpus)} chunks  |  Sampled: {len(sampled)}", flush=True)

    # Resume: load whatever is already saved
    existing = {}
    if Path(OUTPUT_PATH).exists():
        with open(OUTPUT_PATH, encoding="utf-8") as f:
            for entry in json.load(f):
                existing[entry["source_chunk_id"]] = entry
        print(f"Resuming — {len(existing)} entries already done", flush=True)

    dataset = list(existing.values())

    for idx, chunk in enumerate(sampled):
        chunk_id = chunk["id"]
        if chunk_id in existing:
            print(f"[{idx+1}/{len(sampled)}] skip  {chunk_id}", flush=True)
            continue

        print(f"[{idx+1}/{len(sampled)}] gen   {chunk_id} ...", flush=True)
        try:
            resp = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{
                    "role": "user",
                    "content": _PROMPT.format(text=chunk["text"][:2000]),
                }],
                temperature=0.3,
            )
            raw = resp.choices[0].message.content.strip()
            # Strip markdown code fences if present
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            parsed = json.loads(raw.strip())

            dataset.append({
                "id": f"sample_{len(dataset):04d}",
                "question": parsed["question"],
                "reference_answer": parsed["reference_answer"],
                "source_chunk_id": chunk_id,
                "context": chunk["text"],
            })
        except Exception as e:
            print(f"  SKIP {chunk_id}: {e}", flush=True)
            continue

        Path(OUTPUT_PATH).parent.mkdir(exist_ok=True)
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)

        time.sleep(2)

    print(f"\nDone — {len(dataset)} samples saved to {OUTPUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
