import argparse
import json


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate GrailQA answers against retrieved KG triples."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to the validated GrailQA JSON file."
    )

    return parser.parse_args()


def validate_answers(input_path):

    with open(input_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    total = len(dataset)
    answers_found = 0
    answers_not_found = 0

    failed_qids = []

    for example in dataset:

        answers = set(example.get("answers", []))

        entities_in_kg = set()

        for result in example.get("kg_results", []):

            for triple in result.get("triples", []):

                entities_in_kg.add(triple["head"])
                entities_in_kg.add(triple["tail"])

        if answers.issubset(entities_in_kg):
            answers_found += 1
        else:
            answers_not_found += 1
            failed_qids.append(example["qid"])

    print("=" * 60)
    print("GrailQA Answer Validation")
    print("=" * 60)

    print(f"Total:                 {total}")
    print(f"Answers encontradas:   {answers_found}")
    print(f"Answers não encontradas: {answers_not_found}")

    print("\nQIDs sem answer no KG:")

    for qid in failed_qids:
        print(f"  {qid}")


def main():

    args = parse_args()

    validate_answers(args.input)


if __name__ == "__main__":
    main()