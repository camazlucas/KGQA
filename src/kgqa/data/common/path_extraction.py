import argparse
import json
import networkx as nx
from datasets import load_dataset


def build_graph(triples):
    G = nx.DiGraph()
    for h, r, t in triples:
        G.add_edge(h, t, relation=r)
        G.add_edge(t, h, relation=f"(R {r})")  # antes: f"{r}_inv"
    return G


def extract_gold_paths(example, all_shortest=False):
    G = build_graph(example["graph"])
    results = []
    for q_ent in example["q_entity"]:
        for a_ent in example["a_entity"]:
            if q_ent not in G or a_ent not in G:
                continue
            try:
                node_paths = (
                    list(nx.all_shortest_paths(G, q_ent, a_ent))
                    if all_shortest
                    else [nx.shortest_path(G, q_ent, a_ent)]
                )
                for node_path in node_paths:
                    rel_path = [
                        G[node_path[i]][node_path[i + 1]]["relation"]
                        for i in range(len(node_path) - 1)
                    ]
                    results.append({"nodes": node_path, "relations": rel_path})
            except nx.NetworkXNoPath:
                continue
    return results


def _interleave(nodes, relations):
    result = [nodes[0]]
    for rel, node in zip(relations, nodes[1:]):
        result.append(rel)
        result.append(node)
    return result


def process_dataset(dataset_name, split, all_shortest, output_path):
    dataset = load_dataset(dataset_name, split=split)
    output = []
    no_path_ids = []

    for ex in dataset:
        raw_paths = extract_gold_paths(ex, all_shortest=all_shortest)
        if not raw_paths:
            no_path_ids.append(ex["id"])

        output.append({
            "qid": ex["id"],
            "question": ex["question"],
            "topic_entities": ex["q_entity"],
            "paths": [p["relations"] for p in raw_paths],
            "answers": ex["a_entity"],
            "concrete_paths": [_interleave(p["nodes"], p["relations"]) for p in raw_paths],
        })

    print(f"{dataset_name}/{split}: {len(no_path_ids)}/{len(dataset)} sem caminho encontrado")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    report_path = output_path.replace(".json", "_no_path_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "dataset": dataset_name,
            "split": split,
            "total": len(dataset),
            "no_path_count": len(no_path_ids),
            "no_path_ids": no_path_ids,
        }, f, indent=2, ensure_ascii=False)

    print(f"Relatório salvo em: {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="rmanluo/RoG-webqsp ou rmanluo/RoG-cwq")
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"])
    parser.add_argument("--all_shortest", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    process_dataset(args.dataset, args.split, args.all_shortest, args.output)