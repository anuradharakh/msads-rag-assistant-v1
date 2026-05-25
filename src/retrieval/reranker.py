from typing import Dict, List

from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    """CROSS-ENCODER RERANKER. **"""

    def __init__(self, model_name: str):
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        chunks: List[Dict],
        top_n: int = 5,
    ) -> List[Dict]:
        """RERANK CHUNKS USING CROSS-ENCODER. **"""

        if not chunks:
            return []

        pairs = [
            [query, chunk["chunk_text"]]
            for chunk in chunks
        ]

        scores = self.model.predict(pairs)

        reranked = sorted(
            zip(chunks, scores),
            key=lambda item: item[1],
            reverse=True,
        )

        results = []

        for rank, (chunk, score) in enumerate(reranked[:top_n], start=1):
            updated = chunk.copy()
            updated["rank"] = rank
            updated["reranker_score"] = float(score)
            updated["retrieval_strategy"] = (
                f"{chunk.get('retrieval_strategy', 'retrieval')}+reranker"
            )
            results.append(updated)

        return results