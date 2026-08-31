import argparse
import json


def filter_by_answer_count(dataset, threshold=200):
    """
    Remove examples whose gold answer set exceeds `threshold`.

    Returns (filtered_dataset, removed_count).
    """
    filtered = []
    removed = 0

    for example in dataset:
        if len(example.get("answers", [])) > threshold:
            removed += 1
            continue
        filtered.append(example)

    return filtered, removed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Filter out examples with more than N gold answers."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--threshold", type=int, default=200)
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    filtered_data, removed = filter_by_answer_count(data, args.threshold)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(filtered_data, f, ensure_ascii=False, indent=2)

    print(f"Examples input: {len(data)}")
    print(f"Examples removed (>{args.threshold} answers): {removed}")
    print(f"Examples output: {len(filtered_data)}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()