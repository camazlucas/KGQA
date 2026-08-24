import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


CAUSAL_MODELS = {
    "qwen-0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "llama-8b": "meta-llama/Llama-3.1-8B-Instruct",
    "llama-3.2-3b": "meta-llama/Llama-3.2-3B-Instruct",
    "llama-2-chat-7b": "meta-llama/Llama-2-7b-chat-hf",
    "qwen-2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
    "llama-3-8b": "meta-llama/Meta-Llama-3-8B-Instruct",
    "qwen-14b": "Qwen/Qwen2.5-14B-Instruct",
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