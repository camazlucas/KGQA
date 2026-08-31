import argparse
import json

from src.kgqa.data.common.filter_by_answer_count import filter_by_answer_count


def normalize_relation(relation):
    if relation.startswith("(R ") and relation.endswith(")"):
        return relation[3:-1]
    return relation


def get_relation_direction(path_relation):
    if path_relation.startswith("(R ") and path_relation.endswith(")"):
        return "reverse"
    return "forward"


def get_next_entity(triple, path_relation):
    direction = get_relation_direction(path_relation)
    return triple["tail"] if direction == "forward" else triple["head"]


def get_previous_entity(triple, path_relation):
    direction = get_relation_direction(path_relation)
    return triple["head"] if direction == "forward" else triple["tail"]


def triple_matches_relation(triple, path_relation):
    return triple["relation"] == normalize_relation(path_relation)


def triple_contains_answer(triple, answers):
    return triple["head"] in answers or triple["tail"] in answers


def filter_path_triples(path, triples, answers):
    if not path:
        return []

    hop_triples = [
        [t for t in triples if triple_matches_relation(t, rel)]
        for rel in path
    ]

    last_index = len(path) - 1
    last_relation = path[last_index]

    selected_last = [
        t for t in hop_triples[last_index]
        if triple_contains_answer(t, answers)
    ]

    selected_by_hop = {last_index: selected_last}
    required_entities = {get_previous_entity(t, last_relation) for t in selected_last}

    for index in range(last_index - 1, -1, -1):
        path_relation = path[index]
        selected = [
            t for t in hop_triples[index]
            if get_next_entity(t, path_relation) in required_entities
        ]
        selected_by_hop[index] = selected
        required_entities = {get_previous_entity(t, path_relation) for t in selected}

    selected_triples = []
    seen = set()
    for index in range(len(path)):
        for triple in selected_by_hop.get(index, []):
            key = (triple["head"], triple["relation"], triple["tail"])
            if key not in seen:
                seen.add(key)
                selected_triples.append(triple)

    return selected_triples


def filter_example(example):
    paths = example.get("paths", [])
    answers = set(example.get("answers", []))

    all_triples = []
    for result in example.get("kg_results", []):
        all_triples.extend(result.get("triples", []))

    filtered_triples = []
    for path in paths:
        filtered_triples.extend(filter_path_triples(path, all_triples, answers))

    unique_triples = []
    seen = set()
    for triple in filtered_triples:
        key = (triple["head"], triple["relation"], triple["tail"])
        if key not in seen:
            seen.add(key)
            unique_triples.append(triple)

    filtered_example = example.copy()
    filtered_example["kg_results"] = [{"triples": unique_triples}]
    return filtered_example


def parse_args():
    parser = argparse.ArgumentParser(
        description="Filter noisy SPARQL triples in GrailQA using gold paths (GrailQA-specific)."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--threshold", type=int, default=200)
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Etapa 1 (genérica): descarta perguntas com respostas demais
    data, removed = filter_by_answer_count(data, args.threshold)

    # Etapa 2 (específica do GrailQA): remove triplas ruidosas do SPARQL
    filtered_data = [filter_example(example) for example in data]

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(filtered_data, f, ensure_ascii=False, indent=2)

    print(f"Examples removed (>{args.threshold} answers): {removed}")
    print(f"Examples output: {len(filtered_data)}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()