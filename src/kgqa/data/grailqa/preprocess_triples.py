import argparse
import json
import pickle


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert Freebase MIDs in KG triples to entity labels."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to the validated GrailQA JSON file."
    )

    parser.add_argument(
        "--dictionary",
        required=True,
        help="Path to the MID-to-label dictionary."
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Path to the preprocessed JSON file."
    )

    return parser.parse_args()


def load_dataset(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_dictionary(path):
    print("Loading MID-to-label dictionary...")

    with open(path, "rb") as f:
        dictionary = pickle.load(f)

    print(f"Dictionary entries: {len(dictionary)}")

    return dictionary


def get_label(mid, dictionary):
    """
    Returns the label associated with a Freebase MID.

    If the MID is not present in the dictionary,
    the MID itself is returned.
    """

    return dictionary.get(mid, mid)


def preprocess_dataset(dataset, dictionary):

    output = []

    missing_labels = 0

    for example in dataset:

        kg_results = []

        for result in example.get("kg_results", []):

            triples = []

            for triple in result.get("triples", []):

                head = triple["head"]
                tail = triple["tail"]

                head_label = get_label(head, dictionary)
                tail_label = get_label(tail, dictionary)

                if head_label == head:
                    missing_labels += 1

                if tail_label == tail:
                    missing_labels += 1

                triples.append({
                    "head": head_label,
                    "relation": triple["relation"],
                    "tail": tail_label
                })

            kg_results.append({
                "triples": triples
            })

        output.append({
            "qid": example["qid"],
            "question": example["question"],
            "topic_entities": example["topic_entities"],
            "paths": example["paths"],
            "answers": example["answers"],
            "kg_results": kg_results
        })

    print(f"Missing labels: {missing_labels}")

    return output


def save_dataset(dataset, path):

    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            dataset,
            f,
            ensure_ascii=False,
            indent=2
        )


def main():

    args = parse_args()

    print("Loading dataset...")
    dataset = load_dataset(args.input)

    dictionary = load_dictionary(args.dictionary)

    print("Preprocessing triples...")

    processed_dataset = preprocess_dataset(
        dataset,
        dictionary
    )

    print("Saving dataset...")

    save_dataset(
        processed_dataset,
        args.output
    )

    print()
    print(f"Examples processed: {len(processed_dataset)}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()