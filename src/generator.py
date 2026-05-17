import os
from groq import Groq

MODEL = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = """You are an expert assistant on The Rust Programming Language book.
Answer the user's question using ONLY the context chunks provided.
If the answer is not in the context, say "I don't know based on the Rust Book."
Be concise and clear."""


def format_context(chunks):
    parts = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk["metadata"]["source"]
        title = chunk["metadata"].get("title", source)
        text = chunk["text"]
        parts.append(f"--- Chunk {i} | {title} ({source}) ---\n{text}")
    return "\n\n".join(parts)


def answer(question, chunks):
    context = format_context(chunks)
    user_message = f"Context:\n{context}\n\nQuestion: {question}"

    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    )
    return response.choices[0].message.content
