import pickle
from pathlib import Path
from typing import Dict, List

from rank_bm25 import BM25Okapi


def tokenize(text: str) -> List[str]:
    """TOKENIZE TEXT FOR BM25. **"""
    return text.lower().split()


def build_bm25_index(
    chunks: List[Dict],
    index_dir: str,
) -> None:
    """BUILD BM25 INDEX. **"""

    path = Path(index_dir)
    path.mkdir(parents=True, exist_ok=True)

    tokenized_corpus = [
        tokenize(chunk["chunk_text"])
        for chunk in chunks
    ]

    bm25 = BM25Okapi(tokenized_corpus)

    payload = {
        "bm25": bm25,
        "chunks": chunks,
        "tokenized_corpus": tokenized_corpus,
    }

    output_path = path / "bm25.pkl"

    with output_path.open("wb") as file:
        pickle.dump(payload, file)


def load_bm25_index(index_dir: str) -> Dict:
    """LOAD BM25 INDEX. **"""

    input_path = Path(index_dir) / "bm25.pkl"

    if not input_path.exists():
        raise FileNotFoundError(f"BM25 index not found: {input_path}")

    with input_path.open("rb") as file:
        return pickle.load(file)