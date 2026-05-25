from typing import Dict, List

from src.indexing.bm25_store import load_bm25_index, tokenize
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.rrf import reciprocal_rank_fusion


class HybridRetriever:
    """HYBRID DENSE + BM25 RETRIEVER. **"""

    def __init__(
        self,
        dense_retriever: DenseRetriever,
        index_dir: str,
        rrf_k: int = 60,
    ):
        self.dense_retriever = dense_retriever
        self.rrf_k = rrf_k

        payload = load_bm25_index(index_dir=index_dir)

        self.bm25 = payload["bm25"]
        self.chunks = payload["chunks"]

    def retrieve_bm25(
        self,
        query: str,
        top_k: int = 20,
    ) -> List[Dict]:
        """RETRIEVE TOP-K CHUNKS USING BM25. **"""

        tokenized_query = tokenize(query)

        scores = self.bm25.get_scores(tokenized_query)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )[:top_k]

        results = []

        for rank, index in enumerate(ranked_indices, start=1):
            chunk = self.chunks[index]

            results.append(
                {
                    "rank": rank,
                    "chunk_id": chunk["chunk_id"],
                    "chunk_text": chunk["chunk_text"],
                    "metadata": chunk["metadata"],
                    "score": float(scores[index]),
                    "retrieval_strategy": "bm25",
                }
            )

        return results

    def retrieve(
        self,
        query: str,
        fetch_k: int = 20,
        top_k: int = 5,
    ) -> List[Dict]:
        """RETRIEVE USING DENSE + BM25 + RRF. **"""

        dense_results = self.dense_retriever.retrieve(
            query=query,
            top_k=fetch_k,
        )

        bm25_results = self.retrieve_bm25(
            query=query,
            top_k=fetch_k,
        )

        return reciprocal_rank_fusion(
            ranked_lists=[dense_results, bm25_results],
            rrf_k=self.rrf_k,
            top_k=top_k,
        )