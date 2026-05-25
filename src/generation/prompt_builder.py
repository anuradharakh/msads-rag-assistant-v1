from typing import Dict, List


def build_grounded_prompt(
    question: str,
    chunks: List[Dict],
    fallback_message: str,
) -> str:
    """BUILD GROUNDED RAG PROMPT. **"""

    context_blocks = []

    for index, chunk in enumerate(chunks, start=1):
        metadata = chunk.get("metadata", {})

        source = (
            f"[Source {index}] "
            f"Title: {metadata.get('title', 'Unknown')} | "
            f"Section: {metadata.get('section', 'Unknown')} | "
            f"URL: {metadata.get('url', 'Unknown')}"
        )

        context_blocks.append(
            f"{source}\n{chunk['chunk_text']}"
        )

    context = "\n\n".join(context_blocks)

    return f"""
You are a helpful and empathetic assistant for the University of Chicago MS in Applied Data Science program.

Only answer questions related to the MSADS program, including admissions, curriculum, courses, faculty, capstone, career outcomes, program requirements, and student resources.

If the user asks an unrelated question, politely explain that you can only help with MSADS program-related questions and redirect them to relevant program topics.

Use only the provided context. Do not invent facts. Include citations using [Source N].

Rules:
- Be clear and concise.
- Use only the provided context.
- If the answer is not available, say: "{fallback_message}"
- Include citations using [Source N].
- Do not invent admissions, course, tuition, deadline, or policy information.

Context:
{context}

Question:
{question}

Answer:
""".strip()