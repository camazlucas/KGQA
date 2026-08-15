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

    results = []

    for topic_node in topic_nodes:
        for answer_node in answer_nodes:

            # Caminho na direção original
            if nx.has_path(graph, topic_node, answer_node):
                nodes = nx.shortest_path(
                    graph,
                    topic_node,
                    answer_node
                )

                relations = []

                for source, target in zip(nodes[:-1], nodes[1:]):
                    relations.append({
                        "relation": graph[source][target]["relation"],
                        "reverse": False
                    })

                results.append({
                    "relations": relations
                })

            # Caminho na direção inversa
            elif nx.has_path(graph, answer_node, topic_node):
                nodes = nx.shortest_path(
                    graph,
                    answer_node,
                    topic_node
                )

                nodes = nodes[::-1]

                relations = []

                for source, target in zip(nodes[:-1], nodes[1:]):
                    relations.append({
                        "relation": graph[target][source]["relation"],
                        "reverse": True
                    })

                results.append({
                    "relations": relations
                })

    return results


def main():
    path = r"D:\KGQAArmazenamento\DATASET DoG\grailqa.json"

    dataset = load_grailqa(path)

    for example in dataset[:20]:
        paths = find_gold_paths(example)

        print("=" * 80)
        print(f"Question: {example['question']}")

        print("\nTopic Entities:")

        for topic_entity in example.get("topic_entity", {}):
            print(f"  {topic_entity}")

        print("\nPaths:")

        for i, path_info in enumerate(paths, 1):
            print(f"  Path {i}:")

            for relation in path_info["relations"]:
                direction = (
                    "REVERSE"
                    if relation["reverse"]
                    else "FORWARD"
                )

                print(
                    f"    [{direction}] "
                    f"{relation['relation']}"
                )

        print("\nAnswers:")

        for answer in example["answer"]:
            print(
                f"  {answer['answer_argument']}"
            )


if __name__ == "__main__":
    main()