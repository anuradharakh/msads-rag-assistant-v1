from typing import Dict, List

import chromadb

from src.indexing.embedding import EmbeddingModel


class DenseRetriever:
    """CHROMADB DENSE RETRIEVER. **"""

    def __init__(
        self,
        index_dir: str,
        embedding_model: EmbeddingModel,
        collection_name: str = "msads_chunks",
    ):
        self.embedding_model = embedding_model

        client = chromadb.PersistentClient(path=index_dir)

        self.collection = client.get_collection(collection_name)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Dict]:
        """RETRIEVE TOP-K CHUNKS USING VECTOR SEARCH. **"""

        query_embedding = self.embedding_model.embed_query(query)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )

        retrieved = []

        ids = results["ids"][0]
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        for rank, (chunk_id, document, metadata, distance) in enumerate(
            zip(ids, documents, metadatas, distances),
            start=1,
        ):
            retrieved.append(
                {
                    "rank": rank,
                    "chunk_id": chunk_id,
                    "chunk_text": document,
                    "metadata": metadata,
                    "score": float(1.0 - distance),
                    "retrieval_strategy": "dense",
                }
            )

        return retrieved