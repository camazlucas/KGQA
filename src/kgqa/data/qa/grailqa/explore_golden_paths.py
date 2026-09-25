import json
import networkx as nx


def load_grailqa(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_graph(example):
    graph = nx.DiGraph()

    for node in example["graph_query"]["nodes"]:
        graph.add_node(
            node["nid"],
            node_type=node["node_type"],
            entity_id=node.get("id"),
            question_node=node.get("question_node", 0),
        )

    for edge in example["graph_query"]["edges"]:
        graph.add_edge(
            edge["start"],
            edge["end"],
            relation=edge["relation"],
        )

    return graph


def find_gold_paths(example):
    graph = build_graph(example)

    answer_nodes = [
        node["nid"]
        for node in example["graph_query"]["nodes"]
        if node.get("question_node") == 1
    ]

    topic_entities = set(example.get("topic_entity", {}).keys())

    topic_nodes = [
        node["nid"]
        for node in example["graph_query"]["nodes"]
        if node.get("id") in topic_entities
    ]

    node_ids = {
        node["nid"]: node.get("id")
        for node in example["graph_query"]["nodes"]
    }

    results = []

    # Grafo não direcionado apenas para encontrar o caminho
    search_graph = graph.to_undirected()

    for topic_node in topic_nodes:
        for answer_node in answer_nodes:

            if not nx.has_path(search_graph, topic_node, answer_node):
                continue

            nodes = nx.shortest_path(
                search_graph,
                topic_node,
                answer_node
            )

            relations = []

            for source, target in zip(nodes[:-1], nodes[1:]):

                # Aresta original: source -> target
                if graph.has_edge(source, target):
                    relation = graph[source][target]["relation"]

                    relations.append({
                        "relation": relation,
                        "reverse": False
                    })

                # Aresta original: target -> source
                else:
                    relation = graph[target][source]["relation"]

                    relations.append({
                        "relation": relation,
                        "reverse": True
                    })

            results.append({
                "relations": relations
            })

    return results


from collections import Counter


def main():
    path = r"D:\KGQAArmazenamento\DATASET DoG\grailqa.json"

    dataset = load_grailqa(path)

    stats = Counter()

    for example in dataset:

        # 1. Sem topic entity
        if not example.get("topic_entity"):
            stats["sem_topic_entity"] += 1
            continue

        # 2. Função especial
        function = example.get("function")

        if function not in (None, "none"):
            stats[f"function_{function}"] += 1
            continue

        # 3. Tenta reconstruir o caminho
        paths = find_gold_paths(example)

        if paths:
            stats["valido"] += 1
        else:
            stats["sem_path"] += 1

    print("=" * 60)
    print("GrailQA - Final Validation")
    print("=" * 60)

    print(f"Total:             {len(dataset)}")
    print(f"Válidos:           {stats['valido']}")
    print(f"Sem topic entity:  {stats['sem_topic_entity']}")
    print(f"Sem path:          {stats['sem_path']}")

    print("\nFunctions descartadas:")

    for key, value in stats.items():
        if key.startswith("function_"):
            print(f"  {key.replace('function_', '')}: {value}")

    descartados = (
        stats["sem_topic_entity"]
        + stats["sem_path"]
        + sum(
            value
            for key, value in stats.items()
            if key.startswith("function_")
        )
    )

    print(f"\nTotal descartados: {descartados}")
    print(f"Total contabilizado: {stats['valido'] + descartados}")


if __name__ == "__main__":
    main()