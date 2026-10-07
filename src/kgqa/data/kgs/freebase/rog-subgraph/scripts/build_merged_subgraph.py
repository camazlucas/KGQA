import argparse
import json
import os

from datasets import load_dataset

DATASET_NAME_MAP = {
    "webqsp": "rmanluo/RoG-webqsp",
    "cwq": "rmanluo/RoG-cwq",
}
SPLIT_CHOICES = ["train", "validation", "test"]


def has_answer_in_subgraph(example):
    """True se pelo menos uma a_entity do exemplo aparece no seu proprio subgrafo (graph)."""
    answers = example["a_entity"]
    if not answers:
        return False

    nodes = set()
    for h, _r, t in example["graph"]:
        nodes.add(h)
        nodes.add(t)

    return any(a in nodes for a in answers)


def parse_dataset_pairs(pairs):
    """Valida e normaliza os pares (dataset, split) passados via --dataset."""
    parsed = []
    for dataset_short, split in pairs:
        if dataset_short not in DATASET_NAME_MAP:
            raise ValueError(
                f"dataset '{dataset_short}' invalido. Opcoes: {list(DATASET_NAME_MAP)}"
            )
        if split not in SPLIT_CHOICES:
            raise ValueError(f"split '{split}' invalido. Opcoes: {SPLIT_CHOICES}")
        parsed.append((dataset_short, split))
    return parsed


def main():
    parser = argparse.ArgumentParser(
        description="Junta, em um unico subgrafo global, os subgrafos (campo 'graph') de todos "
                     "os exemplos do RoG-webqsp e RoG-cwq em que pelo menos uma a_entity aparece "
                     "no proprio subgrafo pre-extraido."
    )
    parser.add_argument(
        "--dataset", action="append", nargs=2, metavar=("DATASET", "SPLIT"),
        help="Par dataset/split a processar, ex: --dataset webqsp train. Repetivel. "
             f"Dataset in {list(DATASET_NAME_MAP)}, split in {SPLIT_CHOICES}. "
             "Se omitido, processa todos os 6 combos."
    )
    parser.add_argument(
        "--output",
        default="src/kgqa/data/kgs/freebase/rog-subgraph/kg/merged_subgraph.tsv",
        help="Caminho do subgrafo global mesclado, uma tripla (head\\trelation\\ttail) por linha."
    )
    parser.add_argument(
        "--stats_output",
        default="src/kgqa/data/kgs/freebase/rog-subgraph/kg/merged_subgraph_stats.json",
        help="Caminho do relatorio com contagens e qids inclusos por dataset/split."
    )
    args = parser.parse_args()

    if args.dataset:
        pairs = parse_dataset_pairs(args.dataset)
    else:
        pairs = [(d, s) for d in DATASET_NAME_MAP for s in SPLIT_CHOICES]

    merged_triples = set()
    stats = {}

    for dataset_short, split in pairs:
        dataset_name = DATASET_NAME_MAP[dataset_short]
        print(f"Carregando {dataset_name}/{split} ...")
        dataset = load_dataset(dataset_name, split=split)

        qualifying_qids = []
        for ex in dataset:
            if not has_answer_in_subgraph(ex):
                continue
            qualifying_qids.append(ex["id"])
            for h, r, t in ex["graph"]:
                merged_triples.add((h, r, t))

        stats[f"{dataset_short}/{split}"] = {
            "total_examples": len(dataset),
            "qualifying_examples": len(qualifying_qids),
            "qids": qualifying_qids,
        }
        print(
            f"  {len(qualifying_qids)}/{len(dataset)} exemplos com alguma resposta no subgrafo "
            f"| triplas acumuladas (unicas): {len(merged_triples)}"
        )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for h, r, t in sorted(merged_triples):
            f.write(f"{h}\t{r}\t{t}\n")

    os.makedirs(os.path.dirname(args.stats_output) or ".", exist_ok=True)
    with open(args.stats_output, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"\nSubgrafo global mesclado salvo em: {args.output} ({len(merged_triples)} triplas unicas)")
    print(f"Estatisticas por dataset/split salvas em: {args.stats_output}")


if __name__ == "__main__":
    main()
