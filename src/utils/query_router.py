from typing import List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer


MSADS_INTENT_EXAMPLES = [
    "What are the admission requirements for the MS in Applied Data Science program?",
    "How do I apply to the MSADS program?",
    "What courses are included in the curriculum?",
    "What is the capstone project?",
    "Is the program available online?",
    "What is the difference between online and in-person programs?",
    "What are the tuition and fees?",
    "Does the program support CPT or OPT?",
    "Can international students apply?",
    "What career outcomes do graduates have?",
    "Who are the faculty and instructors?",
    "What are the application deadlines?",
    "What scholarships or financial aid are available?",
    "List all MSADS courses",
    "What courses are offered?",
    "Show curriculum structure",
    "What classes will I take?"
]

GENERAL_CHAT_EXAMPLES = [
    "hello",
    "hi",
    "how are you",
    "thank you",
    "who are you",
    "what can you do",
]


class QueryRouter:
    """SEMANTIC INTENT ROUTER FOR MSADS CHATBOT. **"""

    def __init__(
        self,
        model_name: str = "BAAI/bge-base-en-v1.5",
        threshold: float = 0.55,
    ):
        self.model = SentenceTransformer(model_name)
        self.threshold = threshold

        self.msads_embeddings = self.model.encode(
            MSADS_INTENT_EXAMPLES,
            normalize_embeddings=True,
        )

        self.general_embeddings = self.model.encode(
            GENERAL_CHAT_EXAMPLES,
            normalize_embeddings=True,
        )

    def _max_similarity(self, query_embedding, examples_embeddings) -> float:
        scores = np.dot(examples_embeddings, query_embedding)
        return float(np.max(scores))

    def route(self, query: str) -> Tuple[str, float]:
        query_embedding = self.model.encode(
            query,
            normalize_embeddings=True,
        )

        msads_score = self._max_similarity(
            query_embedding=query_embedding,
            examples_embeddings=self.msads_embeddings,
        )

        general_score = self._max_similarity(
            query_embedding=query_embedding,
            examples_embeddings=self.general_embeddings,
        )

        if general_score > msads_score and general_score >= 0.60:
            return "GENERAL_CHAT", general_score

        if msads_score >= self.threshold:
            return "MSADS_QUERY", msads_score

        return "OUT_OF_SCOPE", msads_score


def direct_response(route: str) -> str:
    if route == "GENERAL_CHAT":
        return (
            "Hi! I’m the MSADS RAG Assistant. I can help with questions about "
            "admissions, curriculum, courses, capstone, tuition, online/in-person options, "
            "career outcomes, CPT/OPT, and other MS in Applied Data Science program details."
        )

    return (
        "I’m designed to help with questions about the University of Chicago MS in Applied "
        "Data Science program. I can help with admissions, curriculum, courses, capstone, "
        "tuition, online/in-person options, career outcomes, CPT/OPT, and related program topics."
    )