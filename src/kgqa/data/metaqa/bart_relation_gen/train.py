import argparse
import json
import os
import time

import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    BartForConditionalGeneration,
    BartTokenizerFast,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
)

from src.kgqa.data.metaqa.bart_relation_gen.relation_head import (
    EOS_ID,
    SPECIAL_TOKENS,
    attach_relation_head,
    build_relation_vocab,
    save_relation_vocab,
)

MAX_HOPS = 3


def load_examples(csv_path, rel_to_id):
    df = pd.read_csv(csv_path)
    questions = df["question"].tolist()
    label_ids = [
        [rel_to_id[r] for r in paths.split("|")] + [EOS_ID]
        for paths in df["paths"].tolist()
    ]
    return questions, label_ids


def build_dataset(questions, label_ids, tokenizer, max_input_length):
    encodings = tokenizer(questions, max_length=max_input_length, truncation=True)
    return Dataset.from_dict({
        "input_ids": encodings["input_ids"],
        "attention_mask": encodings["attention_mask"],
        "labels": label_ids,
    })


def make_compute_metrics(num_special):
    def compute_metrics(eval_preds):
        predictions, labels = eval_preds
        if isinstance(predictions, tuple):
            predictions = predictions[0]

        exact_matches = []
        for pred_row, label_row in zip(predictions, labels):
            gold = [int(t) for t in label_row if t != -100][:-1]  # remove o EOS final
            pred = []
            for t in pred_row:
                t = int(t)
                if t == EOS_ID:
                    break
                if t >= num_special:
                    pred.append(t)
            exact_matches.append(int(pred == gold))

        return {"exact_match": sum(exact_matches) / len(exact_matches)}

    return compute_metrics


class PeriodicCheckpointCallback(TrainerCallback):
    """
    Salva uma copia do modelo a cada N epocas (torch.save direto, mesmo esquema
    do salvamento final -- nao save_pretrained, ver nota em main()). Serve de
    ponto de restauracao independente do save_strategy do Trainer caso o
    treino seja interrompido no meio.
    """

    def __init__(self, model, vocab, output_dir, every_n_epochs):
        self.model = model
        self.vocab = vocab
        self.output_dir = output_dir
        self.every_n_epochs = every_n_epochs

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch = round(state.epoch)
        if epoch > 0 and epoch % self.every_n_epochs == 0:
            ckpt_dir = os.path.join(self.output_dir, f"epoch_{epoch}")
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save(self.model.state_dict(), os.path.join(ckpt_dir, "model_state_dict.pt"))
            save_relation_vocab(self.vocab, os.path.join(ckpt_dir, "relation_vocab.json"))
            print(f"[checkpoint periodico] epoca {epoch} salva em {ckpt_dir}")
        return control


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tuning do BART para gerar a cadeia de relacoes do MetaQA a "
                     "partir da pergunta (com a entidade real, sem mascara). A cabeca do "
                     "decoder e substituida para prever apenas o vocabulario fechado de "
                     "relacoes do KB (ver relation_head.py), em vez do vocabulario inteiro "
                     "do BART. Um unico modelo e treinado com os 3 hops juntos (allhops)."
    )
    parser.add_argument("--train_csv", default="src/kgqa/data/metaqa/outputs/metaqa_allhops_train.csv")
    parser.add_argument("--valid_csv", default="src/kgqa/data/metaqa/outputs/metaqa_allhops_valid.csv")
    parser.add_argument("--output_dir", default="raw_data/bart_relation_gen/checkpoints/allhops")
    parser.add_argument("--base_model", default="facebook/bart-base")
    parser.add_argument("--num_train_epochs", type=int, default=10)
    parser.add_argument("--per_device_train_batch_size", type=int, default=32)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--max_input_length", type=int, default=64)
    parser.add_argument("--early_stopping_patience", type=int, default=2)
    parser.add_argument("--checkpoint_every_n_epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = BartTokenizerFast.from_pretrained(args.base_model)
    model = BartForConditionalGeneration.from_pretrained(args.base_model)

    vocab = build_relation_vocab()
    rel_to_id = {rel: i for i, rel in enumerate(vocab)}
    attach_relation_head(model, tokenizer, vocab)

    train_questions, train_labels = load_examples(args.train_csv, rel_to_id)
    valid_questions, valid_labels = load_examples(args.valid_csv, rel_to_id)

    train_dataset = build_dataset(train_questions, train_labels, tokenizer, args.max_input_length)
    valid_dataset = build_dataset(valid_questions, valid_labels, tokenizer, args.max_input_length)

    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, label_pad_token_id=-100)

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        predict_with_generate=True,
        generation_max_length=MAX_HOPS + 2,  # BOS + ate 3 relacoes + EOS
        generation_num_beams=1,
        load_best_model_at_end=True,
        metric_for_best_model="exact_match",
        greater_is_better=True,
        seed=args.seed,
        report_to=[],
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=collator,
        # Nao passamos processing_class=tokenizer aqui: o Trainer chamaria
        # align_special_tokens() e sobrescreveria o pad/bos/eos_token_id
        # customizados da cabeca de relacoes com os ids do vocabulario grande
        # do tokenizer original. O tokenizer ja vai pro DataCollatorForSeq2Seq
        # acima, que e quem precisa dele de fato.
        compute_metrics=make_compute_metrics(len(SPECIAL_TOKENS)),
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience),
            PeriodicCheckpointCallback(model, vocab, args.output_dir, args.checkpoint_every_n_epochs),
        ],
    )

    start_time = time.time()
    trainer.train()
    training_seconds = time.time() - start_time
    print(f"Tempo de treino: {training_seconds / 3600:.2f} h ({training_seconds:.0f} s)")

    final_metrics = trainer.evaluate()
    print(f"Metricas finais no valid: {final_metrics}")

    # Nao usamos model.save_pretrained aqui: ele salvaria config.vocab_size ja
    # reduzido pro vocabulario pequeno do decoder, o que quebraria o encoder se
    # alguem tentasse reconstruir com from_pretrained (ver relation_head.py).
    # predict.py reconstroi a arquitetura chamando attach_relation_head de novo
    # e so entao carrega estes pesos.
    torch.save(model.state_dict(), os.path.join(args.output_dir, "model_state_dict.pt"))
    save_relation_vocab(vocab, os.path.join(args.output_dir, "relation_vocab.json"))
    tokenizer.save_pretrained(args.output_dir)

    with open(os.path.join(args.output_dir, "training_args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    with open(os.path.join(args.output_dir, "training_time.json"), "w", encoding="utf-8") as f:
        json.dump({
            "training_seconds": training_seconds,
            "training_hours": training_seconds / 3600,
            "epochs_trained": trainer.state.epoch,
        }, f, ensure_ascii=False, indent=2)

    print(f"Modelo, vocabulario e tokenizer salvos em: {args.output_dir}")


if __name__ == "__main__":
    main()
