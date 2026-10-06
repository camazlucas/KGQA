import argparse
import os


def main():
    parser = argparse.ArgumentParser(
        description="Extrai as entidades e relacoes distintas de um grafo em TSV "
                     "(head\\trelation\\ttail por linha), salvando uma lista ordenada de cada."
    )
    parser.add_argument(
        "--graph", required=True,
        help="Caminho do grafo (head\\trelation\\ttail por linha)."
    )
    parser.add_argument(
        "--entities_output",
        help="Caminho de saida da lista de entidades distintas (uma por linha, ordenada). "
             "Default: entities.txt na mesma pasta do --graph."
    )
    parser.add_argument(
        "--relations_output",
        help="Caminho de saida da lista de relacoes distintas (uma por linha, ordenada). "
             "Default: relations.txt na mesma pasta do --graph."
    )
    args = parser.parse_args()

    graph_dir = os.path.dirname(args.graph)
    entities_output = args.entities_output or os.path.join(graph_dir, "entities.txt")
    relations_output = args.relations_output or os.path.join(graph_dir, "relations.txt")

    entities = set()
    relations = set()

    with open(args.graph, "r", encoding="utf-8") as f:
        for line in f:
            h, r, t = line.rstrip("\n").split("\t")
            entities.add(h)
            entities.add(t)
            relations.add(r)

    os.makedirs(os.path.dirname(entities_output) or ".", exist_ok=True)
    with open(entities_output, "w", encoding="utf-8") as f:
        for entity in sorted(entities):
            f.write(f"{entity}\n")

    os.makedirs(os.path.dirname(relations_output) or ".", exist_ok=True)
    with open(relations_output, "w", encoding="utf-8") as f:
        for relation in sorted(relations):
            f.write(f"{relation}\n")

    print(f"{len(entities)} entidades distintas salvas em: {entities_output}")
    print(f"{len(relations)} relacoes distintas salvas em: {relations_output}")


if __name__ == "__main__":
    main()
