import argparse
import os


def main():
    parser = argparse.ArgumentParser(
        description="Extrai as entidades e relacoes distintas do subgrafo global mesclado "
                     "(kg/merged_subgraph.tsv), salvando uma lista ordenada de cada em kg/."
    )
    parser.add_argument(
        "--graph",
        default="src/kgqa/data/kgs/freebase/rog-subgraph/kg/merged_subgraph.tsv",
        help="Caminho do subgrafo mesclado (head\\trelation\\ttail por linha)."
    )
    parser.add_argument(
        "--entities_output",
        default="src/kgqa/data/kgs/freebase/rog-subgraph/kg/entities.txt",
        help="Caminho de saida da lista de entidades distintas (uma por linha, ordenada)."
    )
    parser.add_argument(
        "--relations_output",
        default="src/kgqa/data/kgs/freebase/rog-subgraph/kg/relations.txt",
        help="Caminho de saida da lista de relacoes distintas (uma por linha, ordenada)."
    )
    args = parser.parse_args()

    entities = set()
    relations = set()

    with open(args.graph, "r", encoding="utf-8") as f:
        for line in f:
            h, r, t = line.rstrip("\n").split("\t")
            entities.add(h)
            entities.add(t)
            relations.add(r)

    os.makedirs(os.path.dirname(args.entities_output) or ".", exist_ok=True)
    with open(args.entities_output, "w", encoding="utf-8") as f:
        for entity in sorted(entities):
            f.write(f"{entity}\n")

    os.makedirs(os.path.dirname(args.relations_output) or ".", exist_ok=True)
    with open(args.relations_output, "w", encoding="utf-8") as f:
        for relation in sorted(relations):
            f.write(f"{relation}\n")

    print(f"{len(entities)} entidades distintas salvas em: {args.entities_output}")
    print(f"{len(relations)} relacoes distintas salvas em: {args.relations_output}")


if __name__ == "__main__":
    main()
