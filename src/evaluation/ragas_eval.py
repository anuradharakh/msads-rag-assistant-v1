import json
from pathlib import Path
from typing import Dict, List

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    answer_correctness,
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)


def run_ragas_evaluation(
    generated_answers_path: str,
    ground_truth_path: str,
    output_path: str,
    sample_size: int = 10,
) -> Dict:
    """RUN RAGAS EVALUATION. **"""

    with Path(generated_answers_path).open("r", encoding="utf-8") as file:
        generated_answers = json.load(file)

    with Path(ground_truth_path).open("r", encoding="utf-8") as file:
        ground_truth = json.load(file)

    generated_answers = generated_answers[:sample_size]

    dataset_rows = []

    for item in generated_answers:
        query_id = item["query_id"]

        contexts = [
            chunk["chunk_text"]
            for chunk in item["retrieved_chunks"]
        ]

        dataset_rows.append(
            {
                "question": item["question"],
                "answer": item["answer"],
                "contexts": contexts,
                "ground_truth": ground_truth.get(query_id, ""),
            }
        )

    dataset = Dataset.from_list(dataset_rows)

    result = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
            answer_correctness,
        ],
    )

    metrics = {
        metric: float(score)
        for metric, score in result.items()
    }

    output = {
        "sample_size": sample_size,
        "metrics": metrics,
    }

    with Path(output_path).open("w", encoding="utf-8") as file:
        json.dump(output, file, indent=2)

    return output