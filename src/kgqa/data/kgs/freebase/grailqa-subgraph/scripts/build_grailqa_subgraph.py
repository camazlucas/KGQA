import argparse
import json
import os


def main():
    parser = argparse.ArgumentParser(
        description="Monta um subgrafo do Freebase a partir das triplas (kg_results[].triples) "
                     "do GrailQA, removendo duplicadas."
    )
    parser.add_argument(
        "--input",
        default="src/kgqa/data/qa/grailqa/outputs/grailqa_paths_text.json",
        help="Json do GrailQA com o campo kg_results[].triples em cada pergunta."
    )
    parser.add_argument(
        "--output",
        default="src/kgqa/data/kgs/freebase/grailqa-subgraph/kg/grailqa_subgraph.tsv",
        help="Caminho do subgrafo, uma tripla (head\\trelation\\ttail) por linha."
    )
    parser.add_argument(
        "--stats_output",
        default="src/kgqa/data/kgs/freebase/grailqa-subgraph/kg/grailqa_subgraph_stats.json",
        help="Caminho do relatorio com contagens."
    )
    args = parser.parse_args()

    print(f"Carregando {args.input} ...")
    with open(args.input, "r", encoding="utf-8") as f:
        examples = json.load(f)

    triples = set()
    raw_count = 0
    invalid = 0
    for ex in examples:
        for result in ex["kg_results"]:
            for t in result["triples"]:
                raw_count += 1
                h, r, tail = t["head"], t["relation"], t["tail"]
                if any(c in s for s in (h, r, tail) for c in "\t\n"):
                    invalid += 1  # quebraria o formato TSV
                    continue
                triples.add((h, r, tail))

    stats = {
        "input": args.input,
        "total_examples": len(examples),
        "raw_triples": raw_count,
        "invalid_triples": invalid,
        "unique_triples": len(triples),
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for h, r, t in sorted(triples):
            f.write(f"{h}\t{r}\t{t}\n")

    os.makedirs(os.path.dirname(args.stats_output) or ".", exist_ok=True)
    with open(args.stats_output, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(
        f"{len(examples)} perguntas | {raw_count} triplas brutas | "
        f"{invalid} descartadas (tab/quebra de linha)"
    )
    print(f"\nSubgrafo salvo em: {args.output} ({len(triples)} triplas unicas)")
    print(f"Estatisticas salvas em: {args.stats_output}")


if __name__ == "__main__":
    main()
