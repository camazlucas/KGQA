import argparse
import json


def parse_args():
    parser = argparse.ArgumentParser(
        description="Filter KG context triples using gold paths and answers."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to the input JSON file."
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Path to save the filtered JSON file."
    )

    return parser.parse_args()


def normalize_relation(relation):
    """
    Normalize a path relation for comparison with KG triples.

    Examples:
        "(R relation.name)" -> "relation.name"
        "relation.name"    -> "relation.name"
    """
    if relation.startswith("(R ") and relation.endswith(")"):
        return relation[3:-1]

    return relation


def get_relation_direction(path_relation):
    """
    Return the traversal direction of a path relation.

    Normal relation:
        Head -> Tail

    Reverse relation:
        Tail -> Head
    """
    if path_relation.startswith("(R ") and path_relation.endswith(")"):
        return "reverse"

    return "forward"


def get_next_entity(triple, path_relation):
    """
    Return the entity reached when following the path relation.
    """
    direction = get_relation_direction(path_relation)

    if direction == "forward":
        return triple["tail"]

    return triple["head"]


def get_previous_entity(triple, path_relation):
    """
    Return the entity from which the traversal starts.
    """
    direction = get_relation_direction(path_relation)

    if direction == "forward":
        return triple["head"]

    return triple["tail"]


def triple_matches_relation(triple, path_relation):
    """
    Check whether a KG triple belongs to the relation
    represented by a path step.
    """
    return triple["relation"] == normalize_relation(path_relation)


def triple_contains_answer(triple, answers):
    """
    Check whether the head or tail of a triple is a gold answer.
    """
    return (
        triple["head"] in answers
        or triple["tail"] in answers
    )


def filter_path_triples(path, triples, answers):
    """
    Filter triples for a single gold path using backward traversal.

    The final hop is restricted to triples containing a gold answer.

    For previous hops, only triples that connect to an entity
    that can reach the answer are retained.
    """

    if not path:
        return []

    hop_triples = []

    for path_relation in path:
        relation_triples = [
            triple
            for triple in triples
            if triple_matches_relation(triple, path_relation)
        ]

        hop_triples.append(relation_triples)

    last_index = len(path) - 1
    last_relation = path[last_index]

    # Final hop: keep only triples containing a gold answer.
    selected_last = [
        triple
        for triple in hop_triples[last_index]
        if triple_contains_answer(triple, answers)
    ]

    selected_by_hop = {
        last_index: selected_last
    }

    # Entities that must be reached by the previous hop.
    required_entities = {
        get_previous_entity(triple, last_relation)
        for triple in selected_last
    }

    # Traverse backwards through the remaining hops.
    for index in range(last_index - 1, -1, -1):

        path_relation = path[index]

        selected = []

        for triple in hop_triples[index]:

            next_entity = get_next_entity(
                triple,
                path_relation
            )

            if next_entity in required_entities:
                selected.append(triple)

        selected_by_hop[index] = selected

        # Entities required by the preceding hop.
        required_entities = {
            get_previous_entity(triple, path_relation)
            for triple in selected
        }

    # Preserve hop order and remove duplicate triples.
    selected_triples = []
    seen = set()

    for index in range(len(path)):

        for triple in selected_by_hop.get(index, []):

            key = (
                triple["head"],
                triple["relation"],
                triple["tail"],
            )

            if key not in seen:
                seen.add(key)
                selected_triples.append(triple)

    return selected_triples


def filter_example(example):
    """
    Filter KG triples according to all gold paths.
    """

    paths = example.get("paths", [])
    answers = set(example.get("answers", []))

    all_triples = []

    for result in example.get("kg_results", []):
        all_triples.extend(
            result.get("triples", [])
        )

    filtered_triples = []

    for path in paths:

        path_triples = filter_path_triples(
            path,
            all_triples,
            answers
        )

        filtered_triples.extend(path_triples)

    # Remove duplicate triples.
    unique_triples = []
    seen = set()

    for triple in filtered_triples:

        key = (
            triple["head"],
            triple["relation"],
            triple["tail"],
        )

        if key not in seen:
            seen.add(key)
            unique_triples.append(triple)

    filtered_example = example.copy()

    filtered_example["kg_results"] = [
        {
            "triples": unique_triples
        }
    ]

    return filtered_example


def main():
    args = parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    filtered_data = []
    removed_examples = 0

    for example in data:

        # Remove questions with more than 200 gold answers.
        if len(example.get("answers", [])) > 200:
            removed_examples += 1
            continue

        filtered_example = filter_example(example)
        filtered_data.append(filtered_example)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(
            filtered_data,
            f,
            ensure_ascii=False,
            indent=2
        )

    print(f"Examples input: {len(data)}")
    print(
        f"Examples removed (>200 answers): "
        f"{removed_examples}"
    )
    print(f"Examples output: {len(filtered_data)}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()