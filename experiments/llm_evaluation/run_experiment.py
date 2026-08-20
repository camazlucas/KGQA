import argparse
import json
from pathlib import Path

from experiments.llm_evaluation.causal.loader import load_causal_model
from experiments.llm_evaluation.inference import generate_response
from experiments.llm_evaluation.prompts import build_triples_prompt


RESULTS_DIR = Path("results/llm_evaluation/triples")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        required=True,
        help="Model name defined in the causal loader"
    )

    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to the dataset JSON"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.dataset, "r", encoding="utf-8") as file:
        dataset = json.load(file)

    tokenizer, model = load_causal_model(args.model)

    results = []

    for index, example in enumerate(dataset, start=1):
        question = example["question"]

        triples = example["kg_results"][0]["triples"]

        kg_triples = "\n".join(
            f"{triple['head']} | "
            f"{triple['relation']} | "
            f"{triple['tail']}"
            for triple in triples
        )

        prompt = build_triples_prompt(
            question,
            kg_triples
        )

        response = generate_response(
            model,
            tokenizer,
            prompt
        )

        results.append({
            "qid": example["qid"],
            "question": question,
            "kg_triples": triples,
            "response": response,
            "gold_answer": example["answers"]
        })

        print(f"[{index}/{len(dataset)}] {example['qid']}")

    dataset_name = Path(args.dataset).stem

    output_dir = (
        RESULTS_DIR
        / dataset_name
        / args.model
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = output_dir / "predictions.json"

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(
            results,
            file,
            ensure_ascii=False,
            indent=2
        )

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()