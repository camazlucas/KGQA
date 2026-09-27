import argparse
import csv
import json
import os
import re

HOP_CHOICES = [1, 2, 3]
PREDICTIONS_FILENAME = "predictions_metaqa_{hop}hop_wscores.txt"
REL_SPLIT_PATTERN = re.compile(r"\s+")


def load_gold(gold_dir, hop):
    """Le o csv gold (metaqa_<hop>hop_test.csv) gerado por build_relpaths_dataset.py."""
    path = os.path.join(gold_dir, f"metaqa_{hop}hop_test.csv")
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows.append({
                "qid": row["qid"],
                "question": row["question"],
                "topic_entity": row["topic_entities"],  # sempre 1 entidade no MetaQA
                "gold_path": row["paths"].split("|"),
            })
    return rows


def load_predictions(predictions_path):
    """
    Le o arquivo de predicoes originais do BeamQA (question com a topic entity
    mascarada como 'ne', top-3 cadeias de relacao candidatas separadas por '|',
    relacoes dentro de cada cadeia separadas por espaco).

    Retorna um dict {pergunta_mascarada: [cadeia_top1, cadeia_top2, cadeia_top3]}.
    """
    predictions = {}
    with open(predictions_path, "r", encoding="utf-8") as f:
        next(f)  # header
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            _, question_masked, rel_field, _, _ = line.split("\t")
            candidates = [REL_SPLIT_PATTERN.split(c) for c in rel_field.split("|")]
            predictions[question_masked] = candidates
    return predictions


def evaluate_hop(gold_dir, predictions_dir, hop):
    gold_rows = load_gold(gold_dir, hop)
    predictions = load_predictions(
        os.path.join(predictions_dir, PREDICTIONS_FILENAME.format(hop=hop))
    )

    total = len(gold_rows)
    no_prediction = 0
    hits_at_1 = 0
    hits_at_3 = 0
    mismatches = []

    for row in gold_rows:
        question_masked = row["question"].replace(row["topic_entity"], "ne", 1)
        candidates = predictions.get(question_masked)

        if candidates is None:
            no_prediction += 1
            continue

        top1_hit = candidates[0] == row["gold_path"]
        top3_hit = row["gold_path"] in candidates[:3]

        hits_at_1 += int(top1_hit)
        hits_at_3 += int(top3_hit)

        if not top3_hit:
            mismatches.append({
                "qid": row["qid"],
                "question": row["question"],
                "gold_path": row["gold_path"],
                "predicted_top3": candidates[:3],
            })

    evaluated = total - no_prediction

    return {
        "hop": hop,
        "total_gold_examples": total,
        "no_prediction_found": no_prediction,
        "evaluated": evaluated,
        "hits_at_1": hits_at_1,
        "hits_at_3": hits_at_3,
        "hits_at_1_ratio": round(hits_at_1 / evaluated, 4) if evaluated else None,
        "hits_at_3_ratio": round(hits_at_3 / evaluated, 4) if evaluated else None,
    }, mismatches


def main():
    parser = argparse.ArgumentParser(
        description="Avalia a precisao do gerador de cadeias de relacao original do BeamQA "
                     "(Data/Path_gen/outputs/predictions_metaqa_<hop>hop_wscores.txt) contra o "
                     "caminho gold do split de test do MetaQA, por numero de hops. Casa cada "
                     "exemplo gold com a predicao mascarando a topic entity como 'ne' na "
                     "pergunta (a predicao nao depende de qual entidade e, so do template da "
                     "pergunta). Reporta Hits@1 e Hits@3 (match exato da cadeia inteira, na "
                     "ordem certa)."
    )
    parser.add_argument(
        "--hops", type=int, nargs="+", choices=HOP_CHOICES, default=HOP_CHOICES,
        help="Niveis de hop a avaliar (default: 1 2 3)."
    )
    parser.add_argument(
        "--gold_dir", default="src/kgqa/data/metaqa/outputs",
        help="Diretorio com os csv gold (metaqa_<hop>hop_test.csv)."
    )
    parser.add_argument(
        "--predictions_dir",
        default="methods/_original_reference/BeamQA/Data/Path_gen/outputs",
        help="Diretorio com os arquivos de predicao original do BeamQA."
    )
    parser.add_argument(
        "--output", default="src/kgqa/data/metaqa/evaluation/beamqa/beamqa_relation_eval_report.json",
        help="Caminho do relatorio de saida."
    )
    args = parser.parse_args()

    report = {"per_hop": []}

    for hop in args.hops:
        summary, mismatches = evaluate_hop(args.gold_dir, args.predictions_dir, hop)
        report["per_hop"].append(summary)

        print(
            f"{hop}-hop: {summary['evaluated']}/{summary['total_gold_examples']} avaliados "
            f"({summary['no_prediction_found']} sem predicao encontrada) | "
            f"Hits@1={summary['hits_at_1']}/{summary['evaluated']} ({summary['hits_at_1_ratio']:.2%}) | "
            f"Hits@3={summary['hits_at_3']}/{summary['evaluated']} ({summary['hits_at_3_ratio']:.2%})"
        )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\nRelatorio salvo em: {args.output}")


if __name__ == "__main__":
    main()
