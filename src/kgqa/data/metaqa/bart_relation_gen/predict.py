import argparse
import json
import os
import time

import pandas as pd
import torch
from transformers import BartForConditionalGeneration, BartTokenizerFast

from src.kgqa.data.metaqa.bart_relation_gen.relation_head import (
    EOS_ID,
    SPECIAL_TOKENS,
    attach_relation_head,
    load_relation_vocab,
)

MAX_HOPS = 3
MAX_LENGTH = MAX_HOPS + 2  # BOS + ate 3 relacoes + EOS (o BOS conta como 1o token)


def ids_to_chain(ids, vocab):
    """Converte ids gerados em lista de relacoes: le ate o primeiro EOS e ignora PAD/BOS."""
    chain = []
    for t in ids:
        t = int(t)
        if t == EOS_ID:
            break
        if t >= len(SPECIAL_TOKENS):
            chain.append(vocab[t])
    return chain


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def main():
    parser = argparse.ArgumentParser(
        description="Inferencia do BART de geracao de cadeias de relacao (MetaQA) no split "
                     "escolhido. Gera, para cada pergunta, a predicao greedy e as top-K cadeias "
                     "por beam search (para Hits@K), alem do tempo de inferencia de cada modo."
    )
    parser.add_argument("--model_dir", default="raw_data/bart_relation_gen/checkpoints/allhops",
                        help="Pasta com model_state_dict.pt, relation_vocab.json e o tokenizer.")
    parser.add_argument("--base_model", default="facebook/bart-base")
    parser.add_argument("--input_csv", default="src/kgqa/data/metaqa/outputs/metaqa_allhops_test.csv")
    parser.add_argument("--output", default="raw_data/bart_relation_gen/predictions/allhops_test_predictions.jsonl")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_beams", type=int, default=3,
                        help="Tamanho do beam e numero de cadeias retornadas (K do Hits@K).")
    parser.add_argument("--max_input_length", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None,
                        help="Se definido, usa so as primeiras N perguntas (teste rapido).")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    tokenizer = BartTokenizerFast.from_pretrained(args.base_model)
    model = BartForConditionalGeneration.from_pretrained(args.base_model)
    vocab = load_relation_vocab(os.path.join(args.model_dir, "relation_vocab.json"))
    # Reconstroi a arquitetura com a cabeca de relacoes antes de carregar os pesos
    # (o treino salva so o state_dict, ver nota em train.py / PROGRESSO.md).
    attach_relation_head(model, tokenizer, vocab)
    state_dict = torch.load(os.path.join(args.model_dir, "model_state_dict.pt"), map_location="cpu")
    model.load_state_dict(state_dict)
    model.to(device).eval()

    df = pd.read_csv(args.input_csv)
    if args.limit is not None:
        df = df.head(args.limit)
    qids = df["qid"].tolist()
    questions = df["question"].tolist()

    k = args.num_beams
    greedy_seconds = 0.0
    beam_seconds = 0.0
    records = []

    for start in range(0, len(questions), args.batch_size):
        batch_questions = questions[start:start + args.batch_size]
        enc = tokenizer(
            batch_questions, return_tensors="pt", padding=True,
            truncation=True, max_length=args.max_input_length,
        ).to(device)

        with torch.no_grad():
            sync(device)
            t0 = time.time()
            greedy = model.generate(**enc, max_length=MAX_LENGTH, num_beams=1)
            sync(device)
            greedy_seconds += time.time() - t0

            t0 = time.time()
            beam = model.generate(
                **enc, max_length=MAX_LENGTH, num_beams=k, num_return_sequences=k
            )
            sync(device)
            beam_seconds += time.time() - t0

        beam = beam.view(len(batch_questions), k, -1)
        for i, question in enumerate(batch_questions):
            records.append({
                "qid": qids[start + i],
                "question": question,
                "greedy": ids_to_chain(greedy[i].tolist(), vocab),
                "beam_topk": [ids_to_chain(beam[i, j].tolist(), vocab) for j in range(k)],
            })

        done = min(start + args.batch_size, len(questions))
        print(f"{done}/{len(questions)}", end="\r")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    timing_path = os.path.splitext(args.output)[0] + "_timing.json"
    with open(timing_path, "w", encoding="utf-8") as f:
        json.dump({
            "device": str(device),
            "num_examples": len(records),
            "batch_size": args.batch_size,
            "num_beams": k,
            "greedy_seconds": greedy_seconds,
            "beam_seconds": beam_seconds,
            "greedy_examples_per_second": len(records) / greedy_seconds if greedy_seconds else None,
            "beam_examples_per_second": len(records) / beam_seconds if beam_seconds else None,
        }, f, ensure_ascii=False, indent=2)

    print(f"\n{len(records)} predicoes salvas em: {args.output}")
    print(f"Tempos de inferencia salvos em: {timing_path}")


if __name__ == "__main__":
    main()
