import argparse
import json


def parse_args():
    parser = argparse.ArgumentParser(
        description="Remove a GrailQA example by QID."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Input JSON file."
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output JSON file."
    )

    parser.add_argument(
        "--qid",
        required=True,
        help="QID to remove."
    )

    return parser.parse_args()


def main():

    args = parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    original_size = len(dataset)

    dataset = [
        example
        for example in dataset
        if str(example["qid"]) != str(args.qid)
    ]

    removed = original_size - len(dataset)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(
            dataset,
            f,
            ensure_ascii=False,
            indent=2
        )

    print(f"Exemplos originais: {original_size}")
    print(f"Exemplos removidos: {removed}")
    print(f"Exemplos restantes: {len(dataset)}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()