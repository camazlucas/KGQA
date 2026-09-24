import argparse
import json
import os

from datasets import load_dataset

DATASETS = ["rmanluo/RoG-webqsp", "rmanluo/RoG-cwq"]
SPLITS = ["train", "validation", "test"]


def graph_nodes(graph):
    """Retorna o conjunto de todas as entidades (head/tail) presentes no subgrafo do exemplo."""
    nodes = set()
    for h, r, t in graph:
        nodes.add(h)
        nodes.add(t)
    return nodes


def check_example(example):
    """
    Verifica se as entidades-resposta (a_entity) estão presentes no subgrafo (graph)
    do próprio exemplo.

    Retorna: (n_found, n_total, missing_answers)
    """
    answers = example["a_entity"]
    if len(answers) == 0:
        return 0, 0, []

    nodes = graph_nodes(example["graph"])

    found = [a for a in answers if a in nodes]
    missing = [a for a in answers if a not in nodes]

    return len(found), len(answers), missing


def check_dataset_split(dataset_name, split):
    dataset = load_dataset(dataset_name, split=split)

    total_examples = len(dataset)
    fully_covered = 0
    partially_covered = 0
    zero_covered = 0
    no_answers = 0

    total_answers = 0
    total_answers_found = 0

    problem_qids = []

    for ex in dataset:
        n_found, n_total, missing = check_example(ex)

        if n_total == 0:
            no_answers += 1
            continue

        total_answers += n_total
        total_answers_found += n_found

        if n_found == n_total:
            fully_covered += 1
        elif n_found == 0:
            zero_covered += 1
            problem_qids.append({
                "qid": ex["id"],
                "n_answers": n_total,
                "n_found": n_found,
                "missing_answers": missing,
            })
        else:
            partially_covered += 1
            problem_qids.append({
                "qid": ex["id"],
                "n_answers": n_total,
                "n_found": n_found,
                "missing_answers": missing,
            })

    answer_coverage = (total_answers_found / total_answers) if total_answers > 0 else 1.0

    summary = {
        "dataset": dataset_name,
        "split": split,
        "total_examples": total_examples,
        "no_answers_examples": no_answers,
        "fully_covered_examples": fully_covered,
        "partially_covered_examples": partially_covered,
        "zero_covered_examples": zero_covered,
        "total_individual_answers": total_answers,
        "total_individual_answers_found": total_answers_found,
        "answer_coverage_ratio": round(answer_coverage, 4),
    }

    return summary, problem_qids


def main():
    parser = argparse.ArgumentParser(
        description="Verifica se as entidades-resposta de cada exemplo estão presentes no "
                     "subgrafo (campo 'graph') do próprio exemplo, nos datasets RoG-webqsp e RoG-cwq."
    )
    parser.add_argument(
        "--datasets", nargs="+", default=DATASETS,
        help="Datasets HuggingFace a verificar (default: RoG-webqsp e RoG-cwq)."
    )
    parser.add_argument(
        "--splits", nargs="+", default=SPLITS, choices=SPLITS,
        help="Splits a verificar (default: train, validation, test)."
    )
    parser.add_argument(
        "--output", default="src/kgqa/data/freebase/outputs/answer_coverage_report.json",
        help="Caminho do relatório de saída."
    )
    args = parser.parse_args()

    all_summaries = []
    all_problems = {}

    for dataset_name in args.datasets:
        for split in args.splits:
            print(f"Verificando {dataset_name} / {split} ...")
            summary, problem_qids = check_dataset_split(dataset_name, split)
            all_summaries.append(summary)
            all_problems[f"{dataset_name}/{split}"] = problem_qids

            print(
                f"  {summary['fully_covered_examples']}/{summary['total_examples']} exemplos com "
                f"todas as respostas no subgrafo | "
                f"cobertura de respostas individuais: {summary['answer_coverage_ratio']:.2%} | "
                f"zero cobertura: {summary['zero_covered_examples']} | "
                f"parcial: {summary['partially_covered_examples']} | "
                f"sem answers: {summary['no_answers_examples']}"
            )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({
            "summaries": all_summaries,
            "problem_examples": all_problems,
        }, f, indent=2, ensure_ascii=False)

    print(f"\nRelatório completo salvo em: {args.output}")

    print("\n" + "=" * 70)
    print("Resumo geral")
    print("=" * 70)
    for s in all_summaries:
        print(
            f"{s['dataset']:<20} {s['split']:<12} "
            f"cobertura={s['answer_coverage_ratio']:.2%}  "
            f"exemplos_ok={s['fully_covered_examples']}/{s['total_examples']}"
        )


if __name__ == "__main__":
    main()
