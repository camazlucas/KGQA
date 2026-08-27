import argparse
import json
import time

import torch

from experiments.llm_evaluation.prompts import build_paths_prompt
from experiments.llm_evaluation.seq2seq.loader import load_seq2seq_model
from experiments.llm_evaluation.seq2seq.inference import (
    generate_response as generate_seq2seq_response
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        required=True,
        help="Model name defined in the seq2seq loader"
    )

    parser.add_argument(
        "--qid",
        required=True,
        type=int,
        help="Question ID to test"
    )

    parser.add_argument(
        "--max-concrete-paths",
        required=True,
        type=int,
        help="Maximum number of concrete paths to use"
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to the paths dataset JSON"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.input, "r", encoding="utf-8") as file:
        dataset = json.load(file)

    example = next(
        item for item in dataset
        if item["qid"] == args.qid
    )

    paths = example["concrete_paths"][:args.max_concrete_paths]

    kg_paths = "\n\n".join(
        f"Path {path_index}:\n"
        + "\n→ ".join(path)
        for path_index, path in enumerate(paths, start=1)
    )

    prompt = build_paths_prompt(
        example["question"],
        kg_paths
    )

    print(f"QID: {example['qid']}")
    print(f"Question: {example['question']}")
    print(f"Available concrete paths: {len(example['concrete_paths'])}")
    print(f"Concrete paths used: {len(paths)}")
    print(f"Loading model: {args.model}")

    tokenizer, model = load_seq2seq_model(args.model)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    start_time = time.perf_counter()

    response = generate_seq2seq_response(
        model,
        tokenizer,
        prompt
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed_time = time.perf_counter() - start_time

    print("\nResponse:")
    print(response)

    print(f"\nTime: {elapsed_time:.4f}s")

    if torch.cuda.is_available():
        peak_memory = (
            torch.cuda.max_memory_allocated()
            / (1024 ** 3)
        )

        print(f"Peak GPU memory allocated: {peak_memory:.2f} GB")


if __name__ == "__main__":
    main()