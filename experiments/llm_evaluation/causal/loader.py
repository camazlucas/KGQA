import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


CAUSAL_MODELS = {
    "qwen-0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "llama-8b": "meta-llama/Llama-3.1-8B-Instruct",
}


def load_causal_model(model_name):
    model_id = CAUSAL_MODELS[model_name]

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float16,
        device_map="auto",
    )

    return tokenizer, model