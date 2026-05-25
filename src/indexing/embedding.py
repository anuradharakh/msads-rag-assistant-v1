from typing import List

from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    """WRAPPER AROUND SENTENCE TRANSFORMERS EMBEDDING MODEL. **"""

    def __init__(
        self,
        model_name: str,
        normalize_embeddings: bool = True,
    ):
        self.model_name = model_name
        self.normalize_embeddings = normalize_embeddings
        self.model = SentenceTransformer(model_name)

    def embed_texts(
        self,
        texts: List[str],
        batch_size: int = 64,
    ) -> List[List[float]]:
        """EMBED LIST OF TEXTS. **"""

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=True,
        )

        return embeddings.tolist()

    def embed_query(self, query: str) -> List[float]:
        """EMBED SINGLE QUERY. **"""

        embedding = self.model.encode(
            query,
            normalize_embeddings=self.normalize_embeddings,
        )

        return embedding.tolist()