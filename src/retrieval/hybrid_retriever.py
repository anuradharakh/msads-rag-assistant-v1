from typing import Dict, List

from src.indexing.bm25_store import load_bm25_index, tokenize
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.rrf import reciprocal_rank_fusion


COURSE_QUERY_KEYWORDS = [
    "course",
    "courses",
    "class",
    "classes",
    "curriculum",
    "course list",
    "list names",
    "list all courses",
    "core courses",
]


class HybridRetriever:
    """HYBRID DENSE + BM25 RETRIEVER WITH CURRICULUM-AWARE BOOSTING. **"""

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

    def _is_course_query(self, query: str) -> bool:
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in COURSE_QUERY_KEYWORDS)

    def _get_curriculum_catalog_chunks(self) -> List[Dict]:
        results = []

        for chunk in self.chunks:
            metadata = chunk.get("metadata", {})

            if metadata.get("chunk_type") == "curriculum_catalog":
                results.append(
                    {
                        "rank": 1,
                        "chunk_id": chunk["chunk_id"],
                        "chunk_text": chunk["chunk_text"],
                        "metadata": metadata,
                        "score": 999.0,
                        "retrieval_strategy": "curriculum_catalog_boost",
                    }
                )

        return results

    def _boost_curriculum_catalog(
        self,
        query: str,
        fused_results: List[Dict],
        top_k: int,
    ) -> List[Dict]:
        if not self._is_course_query(query):
            return fused_results[:top_k]

        curriculum_chunks = self._get_curriculum_catalog_chunks()

        if not curriculum_chunks:
            return fused_results[:top_k]

        seen_ids = set()
        boosted_results = []

        for chunk in curriculum_chunks:
            if chunk["chunk_id"] not in seen_ids:
                boosted_results.append(chunk)
                seen_ids.add(chunk["chunk_id"])

        for chunk in fused_results:
            if chunk["chunk_id"] not in seen_ids:
                boosted_results.append(chunk)
                seen_ids.add(chunk["chunk_id"])

        for rank, chunk in enumerate(boosted_results, start=1):
            chunk["rank"] = rank

        return boosted_results[:top_k]

    def retrieve(
        self,
        query: str,
        fetch_k: int = 20,
        top_k: int = 5,
    ) -> List[Dict]:
        dense_results = self.dense_retriever.retrieve(
            query=query,
            top_k=fetch_k,
        )

        bm25_results = self.retrieve_bm25(
            query=query,
            top_k=fetch_k,
        )

        fused_results = reciprocal_rank_fusion(
            ranked_lists=[dense_results, bm25_results],
            rrf_k=self.rrf_k,
            top_k=fetch_k,
        )

        boosted_results = self._boost_curriculum_catalog(
            query=query,
            fused_results=fused_results,
            top_k=top_k,
        )

        for result in boosted_results:
            if result.get("retrieval_strategy") != "curriculum_catalog_boost":
                result["retrieval_strategy"] = "rrf_fused"

        return boosted_results