import argparse
import json
import os

from ..mid_label import load_label_to_mid_dict, looks_like_mid

DATASET_NAME_MAP = {
    "webqsp": "rmanluo/RoG-webqsp",
    "cwq": "rmanluo/RoG-cwq",
}
SPLIT_CHOICES = ["train", "validation", "test"]


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


def classify_labels(labels, label2mids):
    """
    Classifica cada label em resolvido (1 MID candidato), ambiguo (2+ candidatos)
    ou sem MID (0 candidatos). Valores que ja parecem MID sao mantidos como estao.

    Retorna (resolved_mids, ambiguous, unresolved), onde ambiguous e uma lista de
    (label, candidates) e unresolved e uma lista de labels.
    """
    resolved = []
    ambiguous = []
    unresolved = []

    for label in labels:
        if looks_like_mid(label):
            resolved.append(label)
            continue

        candidates = label2mids.get(label)
        if not candidates:
            unresolved.append(label)
        elif len(candidates) == 1:
            resolved.append(candidates[0])
        else:
            ambiguous.append((label, candidates))

    return resolved, ambiguous, unresolved


def convert_entries(entries, dataset_short, split, label2mids):
    """
    Converte topic_entities/answers de label para MID em uma lista de entradas
    {qid, topic_entities, answers}. Itens ambiguos ou sem MID sao omitidos do
    campo e retornados separadamente para os relatorios consolidados. Se isso
    deixar topic_entities ou answers vazio, o exemplo inteiro fica de fora do
    resultado principal (ja fica rastreado via os labels problematicos nos
    relatorios consolidados).
    """
    converted = []
    ambiguous_records = []
    unresolved_records = []

    for entry in entries:
        qid = entry["qid"]

        topic_mids, topic_amb, topic_unres = classify_labels(entry["topic_entities"], label2mids)
        answer_mids, answer_amb, answer_unres = classify_labels(entry["answers"], label2mids)

        if topic_mids and answer_mids:
            converted.append({
                "qid": qid,
                "topic_entities": topic_mids,
                "answers": answer_mids,
            })

        for field, amb in [("topic_entities", topic_amb), ("answers", answer_amb)]:
            for label, candidates in amb:
                ambiguous_records.append({
                    "dataset": dataset_short, "split": split, "qid": qid,
                    "field": field, "label": label, "candidates": candidates,
                })

        for field, unres in [("topic_entities", topic_unres), ("answers", answer_unres)]:
            for label in unres:
                unresolved_records.append({
                    "dataset": dataset_short, "split": split, "qid": qid,
                    "field": field, "label": label,
                })

    return converted, ambiguous_records, unresolved_records


def load_json(path, default):
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def merge_consolidated(existing_records, dataset_short, split, new_records):
    """Remove registros antigos do mesmo dataset/split e adiciona os novos."""
    kept = [
        r for r in existing_records
        if not (r["dataset"] == dataset_short and r["split"] == split)
    ]
    return kept + new_records


def main():
    parser = argparse.ArgumentParser(
        description="Converte topic_entities/answers de label para MID nos jsons gerados por "
                     "extract_missing_answers.py, usando o dicionario mid2label.pkl. Labels "
                     "ambiguos (2+ MIDs candidatos) ou sem MID correspondente sao omitidos do "
                     "json principal e listados em relatorios consolidados separados."
    )
    parser.add_argument(
        "--dataset", action="append", nargs=2, metavar=("DATASET", "SPLIT"),
        help="Par dataset/split a processar, ex: --dataset webqsp train. Repetivel. "
             f"Dataset in {list(DATASET_NAME_MAP)}, split in {SPLIT_CHOICES}. "
             "Se omitido, processa todos os 6 combos."
    )
    parser.add_argument(
        "--dictionary", required=True,
        help="Caminho para o dicionario mid2label.pkl (MID -> label)."
    )
    parser.add_argument(
        "--input_dir", default="src/kgqa/data/kgs/freebase/rog-subgraph/outputs",
        help="Diretorio com os arquivos missing_answers_<dataset>_<split>.json de entrada."
    )
    parser.add_argument(
        "--output_dir", default="src/kgqa/data/kgs/freebase/rog-subgraph/outputs",
        help="Diretorio de saida."
    )
    args = parser.parse_args()

    if args.dataset:
        pairs = parse_dataset_pairs(args.dataset)
    else:
        pairs = [(d, s) for d in DATASET_NAME_MAP for s in SPLIT_CHOICES]

    label2mids = load_label_to_mid_dict(args.dictionary)

    os.makedirs(args.output_dir, exist_ok=True)

    ambiguous_path = os.path.join(args.output_dir, "ambiguous_labels.json")
    unresolved_path = os.path.join(args.output_dir, "unresolved_labels.json")
    ambiguous_records = load_json(ambiguous_path, [])
    unresolved_records = load_json(unresolved_path, [])

    for dataset_short, split in pairs:
        input_path = os.path.join(args.input_dir, f"missing_answers_{dataset_short}_{split}.json")
        entries = load_json(input_path, None)
        if entries is None:
            print(f"  [aviso] {input_path} nao encontrado, pulando {dataset_short}/{split}")
            continue

        converted, new_ambiguous, new_unresolved = convert_entries(
            entries, dataset_short, split, label2mids
        )

        output_path = os.path.join(args.output_dir, f"missing_answers_mid_{dataset_short}_{split}.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(converted, f, indent=2, ensure_ascii=False)

        ambiguous_records = merge_consolidated(ambiguous_records, dataset_short, split, new_ambiguous)
        unresolved_records = merge_consolidated(unresolved_records, dataset_short, split, new_unresolved)

        print(
            f"{dataset_short}/{split}: {len(converted)}/{len(entries)} exemplos salvos em {output_path} "
            f"({len(entries) - len(converted)} descartados por topic_entities/answers vazio; "
            f"labels ambiguos: {len(new_ambiguous)}, sem MID: {len(new_unresolved)})"
        )

    with open(ambiguous_path, "w", encoding="utf-8") as f:
        json.dump(ambiguous_records, f, indent=2, ensure_ascii=False)
    with open(unresolved_path, "w", encoding="utf-8") as f:
        json.dump(unresolved_records, f, indent=2, ensure_ascii=False)

    print(f"\nAmbiguos consolidado: {ambiguous_path} ({len(ambiguous_records)} registros)")
    print(f"Sem MID consolidado: {unresolved_path} ({len(unresolved_records)} registros)")


if __name__ == "__main__":
    main()
