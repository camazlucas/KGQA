import argparse
import csv
import json
import os
import re

HOP_PATTERN = re.compile(r"metaqa_(\d)hop_")


def load_gold(path):
    """Le o csv gold e devolve {qid: {question, gold_path, hop}}."""
    gold = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            gold[row["qid"]] = {
                "question": row["question"],
                "gold_path": row["paths"].split("|"),
                "hop": int(HOP_PATTERN.match(row["qid"]).group(1)),
            }
    return gold


def load_predictions(path):
    """Le o jsonl gerado por predict.py e devolve {qid: registro}."""
    preds = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                preds[rec["qid"]] = rec
    return preds


def new_counter():
    return {"total_gold_examples": 0, "no_prediction_found": 0, "evaluated": 0,
            "hits_at_1_greedy": 0, "hits_at_1_beam": 0, "hits_at_k_beam": 0}


def finalize(counter):
    n = counter["evaluated"]
    out = dict(counter)
    for key in ("hits_at_1_greedy", "hits_at_1_beam", "hits_at_k_beam"):
        out[key + "_ratio"] = round(counter[key] / n, 4) if n else None
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Avalia as predicoes do BART de geracao de cadeias de relacao (predict.py) "
                     "contra o caminho gold do MetaQA, casando por qid (nao ha mascara de "
                     "entidade aqui, ao contrario do avaliador do BeamQA). Reporta, por hop e "
                     "no total, Hits@1 (greedy e beam) e Hits@K (beam), todos com match exato "
                     "da cadeia inteira, na ordem certa."
    )
    parser.add_argument("--gold_csv", default="src/kgqa/data/metaqa/outputs/metaqa_allhops_test.csv")
    parser.add_argument("--predictions", default="raw_data/bart_relation_gen/predictions/allhops_test_predictions.jsonl")
    parser.add_argument("--output", default="src/kgqa/data/metaqa/evaluation/bart_relation_gen/bart_relation_gen_eval_report.json")
    parser.add_argument("--beamqa_report",
                        default="src/kgqa/data/metaqa/evaluation/beamqa/beamqa_relation_eval_report.json",
                        help="Relatorio do BeamQA original, para comparacao lado a lado (opcional).")
    args = parser.parse_args()

    gold = load_gold(args.gold_csv)
    preds = load_predictions(args.predictions)

    per_hop = {}
    overall = new_counter()
    mismatches = []

    for qid, g in gold.items():
        counters = [overall, per_hop.setdefault(g["hop"], new_counter())]
        for c in counters:
            c["total_gold_examples"] += 1

        rec = preds.get(qid)
        if rec is None:
            for c in counters:
                c["no_prediction_found"] += 1
            continue

        greedy_hit = rec["greedy"] == g["gold_path"]
        beam = rec["beam_topk"]
        beam1_hit = bool(beam) and beam[0] == g["gold_path"]
        beamk_hit = g["gold_path"] in beam

        for c in counters:
            c["evaluated"] += 1
            c["hits_at_1_greedy"] += int(greedy_hit)
            c["hits_at_1_beam"] += int(beam1_hit)
            c["hits_at_k_beam"] += int(beamk_hit)

        if not (greedy_hit and beamk_hit):
            mismatches.append({
                "qid": qid, "question": g["question"], "gold_path": g["gold_path"],
                "greedy": rec["greedy"], "beam_topk": beam,
            })

    report = {
        "per_hop": [{"hop": hop, **finalize(c)} for hop, c in sorted(per_hop.items())],
        "overall": finalize(overall),
        "num_mismatches": len(mismatches),
        "mismatches": mismatches,
    }

    def fmt(summary):
        n = summary["evaluated"]
        return (f"{n}/{summary['total_gold_examples']} avaliados | "
                f"Hits@1 greedy={summary['hits_at_1_greedy_ratio']:.2%} | "
                f"Hits@1 beam={summary['hits_at_1_beam_ratio']:.2%} | "
                f"Hits@K beam={summary['hits_at_k_beam_ratio']:.2%}") if n else "nenhum avaliado"

    for entry in report["per_hop"]:
        print(f"{entry['hop']}-hop: {fmt(entry)}")
    print(f"Total : {fmt(report['overall'])}")
    print(f"Exemplos com erro (greedy ou fora do top-K): {len(mismatches)}")

    if os.path.exists(args.beamqa_report):
        with open(args.beamqa_report, "r", encoding="utf-8") as f:
            beamqa = json.load(f)
        print("\nBeamQA original (referencia; avalia so as perguntas com predicao encontrada):")
        for entry in beamqa["per_hop"]:
            print(f"{entry['hop']}-hop: {entry['evaluated']}/{entry['total_gold_examples']} avaliados | "
                  f"Hits@1={entry['hits_at_1_ratio']:.2%} | Hits@3={entry['hits_at_3_ratio']:.2%}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nRelatorio salvo em: {args.output}")


if __name__ == "__main__":
    main()
