import json
from pathlib import Path

from src.chunking.header_chunker import create_header_aware_chunks
from src.crawler.web_crawler import crawl_website
from src.evaluation.hit_rate import compute_hit_rate_at_k, load_json
from src.generation.generator import AnswerGenerator
from src.indexing.bm25_store import build_bm25_index
from src.indexing.embedding import EmbeddingModel
from src.indexing.vector_store import build_chroma_index
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import CrossEncoderReranker
from src.utils.config_loader import load_config
from src.utils.io import read_jsonl, write_json, write_jsonl
from src.utils.logger import log_info, log_success, log_warning


def run_crawling(config: dict) -> None:
    pages = crawl_website(
        root_url=config["source"]["root_url"],
        allowed_domain=config["source"]["allowed_domain"],
        max_pages=config["source"]["max_pages"],
    )

    output_path = Path(config["paths"]["processed_dir"]) / "pages.json"
    write_json(pages, str(output_path))

    log_success(f"Saved crawled pages to: {output_path}")


def run_chunking(config: dict) -> None:
    pages_path = Path(config["paths"]["processed_dir"]) / "pages.json"

    with pages_path.open("r", encoding="utf-8") as file:
        pages = json.load(file)

    chunks = create_header_aware_chunks(
        documents=pages,
        chunk_size=config["chunking"]["chunk_size"],
        chunk_overlap=config["chunking"]["chunk_overlap"],
    )

    output_path = Path(config["paths"]["processed_dir"]) / "chunks.jsonl"
    write_jsonl(chunks, str(output_path))

    log_success(f"Saved {len(chunks)} chunks to: {output_path}")


def run_indexing(config: dict) -> None:
    chunks_path = Path(config["paths"]["processed_dir"]) / "chunks.jsonl"
    chunks = read_jsonl(str(chunks_path))

    embedding_config = config["models"]["embedding"]

    embedding_model = EmbeddingModel(
        model_name=embedding_config["name"],
        normalize_embeddings=embedding_config.get("normalize_embeddings", True),
    )

    texts = [chunk["chunk_text"] for chunk in chunks]

    log_info(f"Embedding {len(texts)} chunks")
    embeddings = embedding_model.embed_texts(
        texts=texts,
        batch_size=embedding_config.get("batch_size", 64),
    )

    build_chroma_index(
        chunks=chunks,
        embeddings=embeddings,
        index_dir=config["paths"]["index_dir"],
    )

    build_bm25_index(
        chunks=chunks,
        index_dir=config["paths"]["index_dir"],
    )

    log_success("Indexes created successfully.")


def build_retriever(config: dict):
    embedding_config = config["models"]["embedding"]

    embedding_model = EmbeddingModel(
        model_name=embedding_config["name"],
        normalize_embeddings=embedding_config.get("normalize_embeddings", True),
    )

    dense_retriever = DenseRetriever(
        index_dir=config["paths"]["index_dir"],
        embedding_model=embedding_model,
    )

    retriever = HybridRetriever(
        dense_retriever=dense_retriever,
        index_dir=config["paths"]["index_dir"],
        rrf_k=config["retrieval"]["hybrid"]["rrf_k"],
    )

    return retriever


def retrieve_with_reranking(question: str, config: dict):
    retriever = build_retriever(config)

    fetch_k = config["retrieval"]["fetch_k"]
    top_k = config["retrieval"]["top_k"]

    chunks = retriever.retrieve(
        query=question,
        fetch_k=fetch_k,
        top_k=fetch_k,
    )

    reranker_config = config["models"]["reranker"]

    reranker = CrossEncoderReranker(
        model_name=reranker_config["name"],
    )

    chunks = reranker.rerank(
        query=question,
        chunks=chunks,
        top_n=top_k,
    )

    return chunks


def run_retrieval_eval(config: dict) -> None:
    queries_path = Path(config["paths"]["eval_dir"]) / "queries.json"
    qrels_path = Path(config["paths"]["eval_dir"]) / "qrels.json"

    if not queries_path.exists() or not qrels_path.exists():
        log_warning("queries.json or qrels.json not found. Skipping Hit Rate evaluation.")
        return

    queries = load_json(str(queries_path))
    qrels = load_json(str(qrels_path))

    retrieval_results = []

    for query_id, payload in queries.items():
        question = payload["query"] if isinstance(payload, dict) else str(payload)

        chunks = retrieve_with_reranking(
            question=question,
            config=config,
        )

        retrieval_results.append(
            {
                "query_id": query_id,
                "question": question,
                "retrieved_chunks": chunks,
            }
        )

    output_dir = Path(config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    retrieval_results_path = output_dir / "retrieval_results.json"
    write_json(retrieval_results, str(retrieval_results_path))

    metrics = compute_hit_rate_at_k(
        retrieval_results=retrieval_results,
        qrels=qrels,
        k=config["evaluation"]["hit_rate_k"],
    )

    metrics_path = output_dir / "retrieval_metrics.json"
    write_json(metrics, str(metrics_path))

    log_success(f"Hit Rate@{metrics['k']}: {metrics['hit_rate']:.4f}")


def run_generation_eval(config: dict) -> None:
    from src.evaluation.ragas_eval import run_ragas_evaluation
    queries_path = Path(config["paths"]["eval_dir"]) / "queries.json"
    ground_truth_path = Path(config["paths"]["eval_dir"]) / "answers.json"

    if not queries_path.exists():
        log_warning("queries.json not found. Skipping generation evaluation.")
        return

    queries = load_json(str(queries_path))

    generator = AnswerGenerator(
        llm_config=config["models"]["llm"],
        generation_config=config["generation"],
    )

    generated = []

    for index, (query_id, payload) in enumerate(queries.items(), start=1):
        question = payload["query"] if isinstance(payload, dict) else str(payload)

        log_info(f"Generating answer {index}/{len(queries)}")

        chunks = retrieve_with_reranking(
            question=question,
            config=config,
        )

        answer = generator.generate(
            question=question,
            chunks=chunks[: config["generation"]["max_context_chunks"]],
        )

        generated.append(
            {
                "query_id": query_id,
                "question": question,
                "answer": answer,
                "retrieved_chunks": chunks,
            }
        )

    output_dir = Path(config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    generated_path = output_dir / "generated_answers.json"
    write_json(generated, str(generated_path))

    if ground_truth_path.exists():
        ragas_output_path = output_dir / "ragas_metrics.json"

        result = run_ragas_evaluation(
            generated_answers_path=str(generated_path),
            ground_truth_path=str(ground_truth_path),
            output_path=str(ragas_output_path),
            sample_size=config["evaluation"]["ragas_sample_size"],
        )

        log_success(f"RAGAS metrics saved: {result['metrics']}")


def main() -> None:
    config = load_config("config.yaml")

    log_info(f"Project: {config['project']['name']}")

    if config["pipeline"].get("run_crawling", False):
        run_crawling(config)

    if config["pipeline"].get("run_chunking", False):
        run_chunking(config)

    if config["pipeline"].get("run_indexing", False):
        run_indexing(config)

    if config["pipeline"].get("run_retrieval_eval", False):
        run_retrieval_eval(config)

    if config["pipeline"].get("run_generation_eval", False):
        run_generation_eval(config)

    log_success("Pipeline completed.")


if __name__ == "__main__":
    main()