import argparse
import csv
import os
import re

HOP_CHOICES = [1, 2, 3]
SPLIT_CHOICES = ["train", "valid", "test"]
TOPIC_ENTITY_PATTERN = re.compile(r"\[([^\]]+)\]")


def parse_hop_file(path, hop, split):
    """
    Cada linha do arquivo original do MetaQA (BeamQA) tem 3 campos separados por tab:
    pergunta com a topic entity entre colchetes, respostas separadas por '|', e a
    cadeia de relacoes separada por '|'.
    """
    entries = []

    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.rstrip("\n")
            if not line:
                continue

            question_raw, answers_raw, relations_raw = line.split("\t")

            match = TOPIC_ENTITY_PATTERN.search(question_raw)
            topic_entity = match.group(1)
            question = TOPIC_ENTITY_PATTERN.sub(topic_entity, question_raw)

            entries.append({
                "qid": f"metaqa_{hop}hop_{split}_{i:04d}",
                "question": question,
                "topic_entities": [topic_entity],
                "answers": answers_raw.split("|"),
                "paths": relations_raw.split("|"),
            })

    return entries


def write_csv(entries, path):
    """
    Salva como CSV, com topic_entities/answers/paths juntados por '|' dentro da
    celula (mesma convencao do arquivo original) -- bem mais compacto que json,
    que repete o nome de cada campo em todo registro.
    """
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["qid", "question", "topic_entities", "answers", "paths"])
        for e in entries:
            writer.writerow([
                e["qid"], e["question"],
                "|".join(e["topic_entities"]), "|".join(e["answers"]), "|".join(e["paths"]),
            ])


def main():
    parser = argparse.ArgumentParser(
        description="Converte os arquivos originais de pergunta+caminho do MetaQA "
                     "(BeamQA/Data/QA_data/MetaQA) para o schema unificado do projeto "
                     "(qid, question, topic_entities, answers, paths), um csv por "
                     "hop/split, mais um csv 'allhops' consolidando os 3 hops por split. "
                     "topic_entities/answers/paths ficam como listas separadas por '|' "
                     "dentro da celula."
    )
    parser.add_argument(
        "--hops", type=int, nargs="+", choices=HOP_CHOICES, default=HOP_CHOICES,
        help="Niveis de hop a processar (default: 1 2 3)."
    )
    parser.add_argument(
        "--splits", nargs="+", choices=SPLIT_CHOICES, default=SPLIT_CHOICES,
        help="Splits a processar (default: train valid test)."
    )
    parser.add_argument(
        "--input_dir",
        default="methods/_original_reference/BeamQA/Data/QA_data/MetaQA",
        help="Diretorio com os arquivos originais train_Nhop.txt/valid_Nhop.txt/test_Nhop.txt."
    )
    parser.add_argument(
        "--output_dir", default="src/kgqa/data/metaqa/outputs",
        help="Diretorio de saida."
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    allhops_by_split = {split: [] for split in args.splits}

    for hop in args.hops:
        for split in args.splits:
            input_path = os.path.join(args.input_dir, f"{split}_{hop}hop.txt")
            entries = parse_hop_file(input_path, hop, split)

            output_path = os.path.join(args.output_dir, f"metaqa_{hop}hop_{split}.csv")
            write_csv(entries, output_path)

            print(f"{hop}hop/{split}: {len(entries)} exemplos salvos em {output_path}")

            allhops_by_split[split].extend(entries)

    for split, entries in allhops_by_split.items():
        output_path = os.path.join(args.output_dir, f"metaqa_allhops_{split}.csv")
        write_csv(entries, output_path)

        print(f"allhops/{split}: {len(entries)} exemplos salvos em {output_path}")


if __name__ == "__main__":
    main()
