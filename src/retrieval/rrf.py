from typing import Dict, List


def reciprocal_rank_fusion(
    ranked_lists: List[List[Dict]],
    rrf_k: int = 60,
    top_k: int = 5,
) -> List[Dict]:
    """MERGE MULTIPLE RANKED LISTS USING RECIPROCAL RANK FUSION. **"""

    scores = {}
    items = {}

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, start=1):
            chunk_id = item["chunk_id"]

            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (rrf_k + rank)
            items[chunk_id] = item

    fused_ids = sorted(
        scores.keys(),
        key=lambda chunk_id: scores[chunk_id],
        reverse=True,
    )

    fused_results = []

    for rank, chunk_id in enumerate(fused_ids[:top_k], start=1):
        item = items[chunk_id].copy()
        item["rank"] = rank
        item["score"] = scores[chunk_id]
        item["retrieval_strategy"] = "rrf_fused"
        fused_results.append(item)

    return fused_results