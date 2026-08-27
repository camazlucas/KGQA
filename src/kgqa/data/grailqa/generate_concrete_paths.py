import argparse
import json
from pathlib import Path


def parse_relation(path_relation):
    """Return (relation, reverse)."""
    if path_relation.startswith("(R ") and path_relation.endswith(")"):
        return path_relation[3:-1], True

    return path_relation, False


def find_concrete_paths(topic_entities, path, triples):
    """
    Reconstruct all concrete paths compatible with a gold path
    using the triples from a single kg_result.
    """

    relations = [parse_relation(step) for step in path]

    concrete_paths = []

    def dfs(current_entity, step_idx, sequence, visited_edges):
        if step_idx == len(relations):
            concrete_paths.append(sequence.copy())
            return

        relation, reverse = relations[step_idx]

        for triple_idx, triple in enumerate(triples):
            if triple_idx in visited_edges:
                continue

            if triple["relation"] != relation:
                continue

            head = triple["head"]
            tail = triple["tail"]

            if reverse:
                if tail != current_entity:
                    continue

                next_entity = head
            else:
                if head != current_entity:
                    continue

                next_entity = tail

            dfs(
                next_entity,
                step_idx + 1,
                sequence + [path[step_idx], next_entity],
                visited_edges | {triple_idx},
            )

    for topic_entity in topic_entities:
        dfs(
            topic_entity,
            0,
            [topic_entity],
            set(),
        )

    return concrete_paths


def filter_paths_by_answer(concrete_paths, answers):
    """
    Keep at most one concrete path for each gold answer.
    """

    selected = []
    answers = set(answers)
    found_answers = set()

    for path in concrete_paths:
        if not path:
            continue

        final_entity = path[-1]

        if (
            final_entity in answers
            and final_entity not in found_answers
        ):
            selected.append(path)
            found_answers.add(final_entity)

    return selected


def generate_concrete_paths(data):
    output = []

    total_kg_results = 0
    total_concrete_paths_before_filter = 0
    total_concrete_paths = 0
    examples_without_paths = 0

    for example in data:
        concrete_paths = []

        paths = example.get("paths", [])
        topic_entities = example.get("topic_entities", [])
        kg_results = example.get("kg_results", [])
        answers = example.get("answers", [])

        total_kg_results += len(kg_results)

        for kg_result in kg_results:
            triples = kg_result.get("triples", [])

            for path in paths:
                found_paths = find_concrete_paths(
                    topic_entities,
                    path,
                    triples,
                )

                for concrete_path in found_paths:
                    if concrete_path not in concrete_paths:
                        concrete_paths.append(concrete_path)

        total_concrete_paths_before_filter += len(concrete_paths)

        # Keep at most one concrete path for each gold answer.
        concrete_paths = filter_paths_by_answer(
            concrete_paths,
            answers,
        )

        if not concrete_paths:
            examples_without_paths += 1

        total_concrete_paths += len(concrete_paths)

        new_example = dict(example)
        new_example["concrete_paths"] = concrete_paths

        output.append(new_example)

    print(f"Examples: {len(output)}")
    print(f"KG results: {total_kg_results}")
    print(
        f"Concrete paths before filter: "
        f"{total_concrete_paths_before_filter}"
    )
    print(
        f"Concrete paths after filter: "
        f"{total_concrete_paths}"
    )
    print(f"Examples without concrete paths: {examples_without_paths}")

    return output


def main():
    parser = argparse.ArgumentParser(
        description="Generate concrete KG paths from filtered GrailQA triples."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Input filtered triples JSON.",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output JSON with concrete paths.",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    output = generate_concrete_paths(data)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(
            output,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()