from pathlib import Path
from typing import Dict, List

import chromadb


def build_chroma_index(
    chunks: List[Dict],
    embeddings: List[List[float]],
    index_dir: str,
    collection_name: str = "msads_chunks",
) -> None:
    """BUILD CHROMADB INDEX. **"""

    path = Path(index_dir)
    path.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(path))

    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    ids = [chunk["chunk_id"] for chunk in chunks]
    documents = [chunk["chunk_text"] for chunk in chunks]
    metadatas = [chunk["metadata"] for chunk in chunks]

    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )