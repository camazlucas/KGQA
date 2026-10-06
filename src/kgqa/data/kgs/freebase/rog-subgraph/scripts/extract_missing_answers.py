import argparse
import json
import os

from datasets import load_dataset

DATASET_NAME_MAP = {
    "webqsp": "rmanluo/RoG-webqsp",
    "cwq": "rmanluo/RoG-cwq",
}
SPLIT_CHOICES = ["train", "validation", "test"]


def load_examples_by_qid(dataset_name, split):
    """Carrega o dataset do HuggingFace e indexa por qid, para recuperar topic_entities."""
    dataset = load_dataset(dataset_name, split=split)
    return {ex["id"]: ex for ex in dataset}


def extract_missing_answers(report, dataset_short, split):
    """
    A partir do relatorio de check_answer_coverage.py, monta a lista
    {qid, topic_entities, answers} para os exemplos com resposta faltante
    de um dataset/split especifico. 'answers' contem apenas as respostas
    faltantes no subgrafo pre-extraido.
    """
    dataset_name = DATASET_NAME_MAP[dataset_short]
    key = f"{dataset_name}/{split}"
    problems = report["problem_examples"].get(key, [])

    if not problems:
        return []

    print(f"Carregando {dataset_name}/{split} para localizar topic_entities...")
    examples_by_qid = load_examples_by_qid(dataset_name, split)

    entries = []
    for problem in problems:
        qid = problem["qid"]
        example = examples_by_qid.get(qid)

        if example is None:
            print(f"  [aviso] qid {qid} nao encontrado no dataset, pulando")
            continue

        entries.append({
            "qid": qid,
            "topic_entities": example["q_entity"],
            "answers": problem["missing_answers"],
        })

    return entries


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
        description="Extrai, por dataset/split do RoG-webqsp e RoG-cwq, a lista completa de "
                     "perguntas com respostas faltantes no subgrafo pre-extraido "
                     "(qid, topic_entities, answers faltantes), uma em cada arquivo de saida."
    )
    parser.add_argument(
        "--dataset", action="append", nargs=2, metavar=("DATASET", "SPLIT"),
        help="Par dataset/split a processar, ex: --dataset webqsp train. Repetivel. "
             f"Dataset in {list(DATASET_NAME_MAP)}, split in {SPLIT_CHOICES}. "
             "Se omitido, processa todos os 6 combos."
    )
    parser.add_argument(
        "--report", default="src/kgqa/data/kgs/freebase/rog-subgraph/outputs/answer_coverage_report.json",
        help="Caminho do relatorio gerado por check_answer_coverage.py."
    )
    parser.add_argument(
        "--output_dir", default="src/kgqa/data/kgs/freebase/rog-subgraph/outputs",
        help="Diretorio de saida (um arquivo missing_answers_<dataset>_<split>.json por combo)."
    )
    args = parser.parse_args()

    if args.dataset:
        pairs = parse_dataset_pairs(args.dataset)
    else:
        pairs = [(d, s) for d in DATASET_NAME_MAP for s in SPLIT_CHOICES]

    with open(args.report, "r", encoding="utf-8") as f:
        report = json.load(f)

    os.makedirs(args.output_dir, exist_ok=True)

    for dataset_short, split in pairs:
        entries = extract_missing_answers(report, dataset_short, split)

        output_path = os.path.join(args.output_dir, f"missing_answers_{dataset_short}_{split}.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(entries, f, indent=2, ensure_ascii=False)

        print(f"{dataset_short}/{split}: {len(entries)} exemplos salvos em {output_path}")


if __name__ == "__main__":
    main()
