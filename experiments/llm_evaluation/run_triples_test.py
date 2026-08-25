import argparse
import json

from experiments.llm_evaluation.causal.loader import load_causal_model
from experiments.llm_evaluation.inference import generate_response
from experiments.llm_evaluation.prompts import build_triples_prompt


DATASET_PATH = "src/kgqa/data/grailqa/grailqa_triples_text.json"


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        required=True,
        help="Model name defined in the causal loader"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        dataset = json.load(file)

    examples = dataset[:2]

    tokenizer, model = load_causal_model(args.model)

    for example in examples:
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
            prompt,
            args.model
        )

        print("\n" + "=" * 60)
        print("QUESTION:")
        print(question)

        print("\nKG TRIPLES:")
        print(kg_triples)

        print("\nGOLD ANSWER:")
        print(example["answers"])

        print("\nRESPONSE:")
        print(response)


if __name__ == "__main__":
    main()