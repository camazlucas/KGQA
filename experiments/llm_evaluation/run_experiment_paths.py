import argparse
import json
import time
from pathlib import Path

import torch

from experiments.llm_evaluation.prompts import build_paths_prompt

from experiments.llm_evaluation.causal.loader import load_causal_model
from experiments.llm_evaluation.causal.inference import (
    generate_response as generate_causal_response
)

from experiments.llm_evaluation.seq2seq.loader import load_seq2seq_model
from experiments.llm_evaluation.seq2seq.inference import (
    generate_response as generate_seq2seq_response
)


SEQ2SEQ_MODELS = {
    "bart-base",
    "bart-large",
    "flan-t5-xl",
}


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of examples to process"
    )

    parser.add_argument(
        "--results-dir",
        required=True,
        help="Directory where experiment results will be saved"
    )

    parser.add_argument(
        "--model",
        required=True,
        help="Model name defined in the causal or seq2seq loader"
    )

    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to the paths dataset JSON"
    )

    return parser.parse_args()


def get_gpu_memory_gb():
    if not torch.cuda.is_available():
        return None, None

    allocated = torch.cuda.memory_allocated() / (1024 ** 3)
    reserved = torch.cuda.memory_reserved() / (1024 ** 3)

    return allocated, reserved


def get_peak_gpu_memory_gb():
    if not torch.cuda.is_available():
        return None, None

    peak_allocated = torch.cuda.max_memory_allocated() / (1024 ** 3)
    peak_reserved = torch.cuda.max_memory_reserved() / (1024 ** 3)

    return peak_allocated, peak_reserved


def get_gpu_info():
    if not torch.cuda.is_available():
        return None, None

    gpu_name = torch.cuda.get_device_name(0)

    gpu_total_memory = (
        torch.cuda.get_device_properties(0).total_memory
        / (1024 ** 3)
    )

    return gpu_name, gpu_total_memory


def main():
    args = parse_args()

    results_dir = Path(args.results_dir)

    with open(args.dataset, "r", encoding="utf-8") as file:
        dataset = json.load(file)

    if args.limit is not None:
        dataset = dataset[:args.limit]

    print(f"Examples to process: {len(dataset)}")

    print(f"Loading model: {args.model}")

    if args.model in SEQ2SEQ_MODELS:
        tokenizer, model = load_seq2seq_model(args.model)
    else:
        tokenizer, model = load_causal_model(args.model)

    gpu_name, gpu_total_memory = get_gpu_info()

    # Reset peak statistics after model loading.
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    gpu_allocated, gpu_reserved = get_gpu_memory_gb()

    results = []

    start_time = time.perf_counter()

    for index, example in enumerate(dataset, start=1):

        question_start = time.perf_counter()

        question = example["question"]

        paths = example["concrete_paths"]

        kg_paths = "\n\n".join(
            f"Path {path_index}:\n"
            + "\n→ ".join(path)
            for path_index, path in enumerate(paths, start=1)
        )

        prompt = build_paths_prompt(
            question,
            kg_paths
        )

        if args.model in SEQ2SEQ_MODELS:
            response = generate_seq2seq_response(
                model,
                tokenizer,
                prompt
            )
        else:
            response = generate_causal_response(
                model,
                tokenizer,
                prompt,
                args.model
            )

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        question_time = time.perf_counter() - question_start

        results.append({
            "qid": example["qid"],
            "question": question,
            "kg_paths": paths,
            "response": response,
            "gold_answer": example["answers"],
            "time_seconds": question_time
        })

        print(
            f"[{index}/{len(dataset)}] "
            f"{example['qid']} "
            f"({question_time:.3f}s)"
        )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    total_time = time.perf_counter() - start_time

    average_time = total_time / len(dataset)

    peak_allocated, peak_reserved = get_peak_gpu_memory_gb()

    dataset_name = Path(args.dataset).stem

    output_dir = (
        results_dir
        / dataset_name
        / args.model
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = output_dir / "predictions.json"

    output = {
        "metadata": {
            "model": args.model,
            "dataset": dataset_name,
            "num_examples": len(dataset),
            "total_time_seconds": total_time,
            "average_time_per_question_seconds": average_time,
            "gpu_name": gpu_name,
            "gpu_total_memory_gb": gpu_total_memory,
            "gpu_memory_allocated_gb": gpu_allocated,
            "gpu_memory_reserved_gb": gpu_reserved,
            "gpu_peak_memory_allocated_gb": peak_allocated,
            "gpu_peak_memory_reserved_gb": peak_reserved
        },
        "predictions": results
    }

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(
            output,
            file,
            ensure_ascii=False,
            indent=2
        )

    print("\nExperiment completed.")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average time/question: {average_time:.4f}s")

    if peak_allocated is not None:
        print(
            f"Peak GPU memory allocated: "
            f"{peak_allocated:.2f} GB"
        )

        print(
            f"Peak GPU memory reserved: "
            f"{peak_reserved:.2f} GB"
        )

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()