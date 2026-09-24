import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


SEQ2SEQ_MODELS = {
    "bart-base": "facebook/bart-base",
    "bart-large": "facebook/bart-large",
    "flan-t5-xl": "google/flan-t5-xl",
}


def load_seq2seq_model(model_name):
    model_id = SEQ2SEQ_MODELS[model_name]

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_id,
        dtype=torch.float16,
        device_map="auto",
    )

    return tokenizer, model