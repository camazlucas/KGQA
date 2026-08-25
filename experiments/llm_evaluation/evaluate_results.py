import argparse
import json
import re
import string


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate LLM KGQA predictions."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to the predictions JSON file."
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Path to save the evaluation results JSON."
    )

    return parser.parse_args()


def normalize_text(text):
    """
    Normalize text for QA evaluation.

    Operations:
    - Convert to lowercase
    - Remove punctuation
    - Normalize whitespace
    """

    text = str(text).lower()

    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def tokenize(text):
    return normalize_text(text).split()


def exact_match(prediction, gold):
    return int(
        normalize_text(prediction) == normalize_text(gold)
    )


def f1_score(prediction, gold):
    prediction_tokens = tokenize(prediction)
    gold_tokens = tokenize(gold)

    if not prediction_tokens or not gold_tokens:
        return int(prediction_tokens == gold_tokens)

    common = {}

    for token in prediction_tokens:
        common[token] = common.get(token, 0) + 1

    num_common = 0

    for token in gold_tokens:
        if common.get(token, 0) > 0:
            num_common += 1
            common[token] -= 1

    if num_common == 0:
        return 0.0

    precision = num_common / len(prediction_tokens)
    recall = num_common / len(gold_tokens)

    return 2 * precision * recall / (precision + recall)


def hits_at_1(prediction, gold_answers):
    normalized_prediction = normalize_text(prediction)

    for gold in gold_answers:
        normalized_gold = normalize_text(gold)

        if normalized_gold in normalized_prediction:
            return 1

    return 0


def evaluate_prediction(prediction):
    response = prediction.get("response", "")
    gold_answers = prediction.get("gold_answer", [])

    if not gold_answers:
        return {
            "exact_match": 0,
            "f1": 0.0,
            "hits_at_1": 0
        }

    exact_matches = [
        exact_match(response, gold)
        for gold in gold_answers
    ]

    f1_scores = [
        f1_score(response, gold)
        for gold in gold_answers
    ]

    return {
        "exact_match": max(exact_matches),
        "f1": max(f1_scores),
        "hits_at_1": hits_at_1(
            response,
            gold_answers
        )
    }


def evaluate_dataset(data):
    predictions = data["predictions"]

    total_em = 0
    total_f1 = 0.0
    total_hits = 0

    evaluated_predictions = []

    for prediction in predictions:

        metrics = evaluate_prediction(prediction)

        total_em += metrics["exact_match"]
        total_f1 += metrics["f1"]
        total_hits += metrics["hits_at_1"]

        evaluated_predictions.append({
            "qid": prediction["qid"],
            **metrics
        })

    num_examples = len(predictions)

    metadata = data["metadata"]

    results = {
        "metadata": {
            "model": metadata["model"],
            "dataset": metadata["dataset"],
            "num_examples": num_examples,
            "exact_match": total_em / num_examples,
            "f1": total_f1 / num_examples,
            "hits_at_1": total_hits / num_examples,
            "total_time_seconds": metadata.get(
                "total_time_seconds"
            ),
            "average_time_per_question_seconds": metadata.get(
                "average_time_per_question_seconds"
            ),
            "gpu_name": metadata.get("gpu_name"),
            "gpu_total_memory_gb": metadata.get(
                "gpu_total_memory_gb"
            ),
            "gpu_memory_allocated_gb": metadata.get(
                "gpu_memory_allocated_gb"
            ),
            "gpu_memory_reserved_gb": metadata.get(
                "gpu_memory_reserved_gb"
            ),
            "gpu_peak_memory_allocated_gb": metadata.get(
                "gpu_peak_memory_allocated_gb"
            ),
            "gpu_peak_memory_reserved_gb": metadata.get(
                "gpu_peak_memory_reserved_gb"
            )
        },
        "predictions": evaluated_predictions
    }

    return results


def save_results(results, path):

    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            results,
            f,
            ensure_ascii=False,
            indent=2
        )


def main():

    args = parse_args()

    print("Loading results...")

    with open(
        args.input,
        "r",
        encoding="utf-8"
    ) as f:
        data = json.load(f)

    print(
        f"Model: {data['metadata']['model']}"
    )

    print(
        f"Examples: {len(data['predictions'])}"
    )

    results = evaluate_dataset(data)

    save_results(
        results,
        args.output
    )

    metadata = results["metadata"]

    print()
    print("Evaluation completed.")
    print(
        f"Exact Match: {metadata['exact_match']:.4f}"
    )
    print(
        f"F1: {metadata['f1']:.4f}"
    )
    print(
        f"Hits@1: {metadata['hits_at_1']:.4f}"
    )
    print(
        f"Average time: "
        f"{metadata['average_time_per_question_seconds']:.4f}s"
    )
    print(
        f"Peak VRAM: "
        f"{metadata['gpu_peak_memory_allocated_gb']:.2f} GB"
    )
    print()
    print(
        f"Results saved to: {args.output}"
    )


if __name__ == "__main__":
    main()