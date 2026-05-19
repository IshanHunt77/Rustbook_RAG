import os
import re
from groq import Groq

from src.config import LLM_MODEL, SYSTEM_PROMPT, CONTEXT_TEMPLATE


def _format_context(chunks: list[dict]) -> str:
    return "\n\n".join(
        CONTEXT_TEMPLATE.format(
            index=i,
            title=c["metadata"].get("title", c["metadata"]["source"]),
            source=c["metadata"]["source"],
            text=c["text"],
        )
        for i, c in enumerate(chunks, 1)
    )


def answer(question: str, chunks: list[dict]) -> dict:
    context = _format_context(chunks)
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
    )
    text = response.choices[0].message.content

    raw_indices = re.findall(r'\[(\d+)\]', text)
    cited = sorted({int(n) for n in raw_indices if 1 <= int(n) <= len(chunks)})

    return {
        "answer": text,
        "citations": [
            {
                "index": n,
                "title": chunks[n - 1]["metadata"].get("title", chunks[n - 1]["metadata"]["source"]),
                "source": chunks[n - 1]["metadata"]["source"],
            }
            for n in cited
        ],
    }
