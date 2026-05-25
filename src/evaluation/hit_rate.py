import json
from pathlib import Path
from typing import Dict, List


def load_json(path: str):
    """LOAD JSON FILE. **"""

    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def compute_hit_rate_at_k(
    retrieval_results: List[Dict],
    qrels: Dict,
    k: int = 5,
) -> Dict:
    """COMPUTE HIT RATE@K. **"""

    hits = 0
    per_query = []

    for result in retrieval_results:
        query_id = result["query_id"]

        expected_urls = qrels.get(query_id, [])

        retrieved_chunks = result["retrieved_chunks"][:k]

        retrieved_urls = []

        for chunk in retrieved_chunks:
            metadata = chunk.get("metadata", {})
            url = metadata.get("url")

            if url:
                retrieved_urls.append(url)

        hit = any(
            url in expected_urls
            for url in retrieved_urls
        )

        if hit:
            hits += 1

        per_query.append(
            {
                "query_id": query_id,
                "question": result["question"],
                "expected_urls": expected_urls,
                "retrieved_urls": retrieved_urls,
                "hit": hit,
            }
        )

    total_queries = len(retrieval_results)

    return {
        "k": k,
        "total_queries": total_queries,
        "hits": hits,
        "hit_rate": hits / total_queries if total_queries > 0 else 0.0,
        "per_query": per_query,
    }