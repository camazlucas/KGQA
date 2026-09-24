import argparse
import csv
import json
from pathlib import Path


METRICS = [
    "exact_match",
    "f1",
    "hits_at_1",
    "total_time_seconds",
    "average_time_per_question_seconds",
    "gpu_peak_memory_allocated_gb",
    "gpu_peak_memory_reserved_gb",
]


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--results-dir",
        required=True,
        help="Directory containing evaluation.json files"
    )

    parser.add_argument(
        "--output",
        default="results_summary.csv",
        help="Output CSV file"
    )

    return parser.parse_args()


def collect_results(results_dir):
    results = []

    evaluation_files = sorted(
        Path(results_dir).rglob("evaluation.json")
    )

    for file_path in evaluation_files:

        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        metadata = data["metadata"]

        result = {
            "model": metadata["model"],
            "dataset": metadata["dataset"],
            "num_examples": metadata["num_examples"],
        }

        for metric in METRICS:
            result[metric] = metadata.get(metric)

        results.append(result)

    return results


def save_csv(results, output_path):

    if not results:
        print("No evaluation.json files found.")
        return

    fieldnames = results[0].keys()

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(results)


def main():

    args = parse_args()

    results = collect_results(args.results_dir)

    print(f"Evaluation files found: {len(results)}")

    for result in results:
        print(
            f"{result['model']}: "
            f"F1={result['f1']:.4f}, "
            f"Hits@1={result['hits_at_1']:.4f}, "
            f"Time={result['average_time_per_question_seconds']:.4f}s, "
            f"Peak VRAM={result['gpu_peak_memory_allocated_gb']}"
        )

    save_csv(results, args.output)

    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()