import json
import argparse

from explore_golden_paths import load_grailqa, find_gold_paths


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate gold paths dataset from GrailQA."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to the GrailQA JSON file."
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Path to the generated gold paths JSON file."
    )

    return parser.parse_args()

def format_path(relations):
    return [
        f"(R {item['relation']})"
        if item["reverse"]
        else item["relation"]
        for item in relations
    ]


def generate_dataset(input_path, output_path):
    dataset = load_grailqa(input_path)

    output = []

    for example in dataset:

        # Descarta exemplos sem topic entity
        if not example.get("topic_entity"):
            continue

        # Descarta funções que não representam apenas caminhos
        if example.get("function") not in (None, "none"):
            continue

        paths = find_gold_paths(example)

        # Segurança: só salva exemplos para os quais
        # conseguimos reconstruir pelo menos um caminho
        if not paths:
            continue

        output.append({
            "qid": example["qid"],
            "question": example["question"],
            "topic_entities": list(
                example["topic_entity"].keys()
            ),
            "paths": [
                format_path(path_info["relations"])
                for path_info in paths
            ],
            "answers": [
                answer["answer_argument"]
                for answer in example["answer"]
            ]
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            output,
            f,
            ensure_ascii=False,
            indent=2
        )

    print(f"Total de exemplos gerados: {len(output)}")
    print(f"Output: {output_path}")


def main():
    args = parse_args()

    generate_dataset(
        input_path=args.input,
        output_path=args.output
    )


if __name__ == "__main__":
    main()