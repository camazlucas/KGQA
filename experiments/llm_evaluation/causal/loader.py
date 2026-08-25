import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Mistral3ForConditionalGeneration,
    MistralCommonBackend,
)


CAUSAL_MODELS = {
    "qwen-0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "llama-8b": "meta-llama/Llama-3.1-8B-Instruct",
    "llama-3.2-3b": "meta-llama/Llama-3.2-3B-Instruct",
    "llama-2-chat-7b": "meta-llama/Llama-2-7b-chat-hf",
    "qwen-2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
    "llama-3-8b": "meta-llama/Meta-Llama-3-8B-Instruct",
    "qwen-14b": "Qwen/Qwen2.5-14B-Instruct",
    "deepseek-llm-7b-chat": "deepseek-ai/deepseek-llm-7b-chat",
    "deepseek-coder-6.7b": "deepseek-ai/deepseek-coder-6.7b-instruct",
    "deepseek-coder-1.3b": "deepseek-ai/deepseek-coder-1.3b-instruct",
    "ministral-3-3b": "mistralai/Ministral-3-3B-Instruct-2512",
    "ministral-3-8b-reasoning": "mistralai/Ministral-3-8B-Reasoning-2512",
    "ministral-3-8b": "mistralai/Ministral-3-8B-Instruct-2512",
}

MINISTRAL_MODELS = {
    "ministral-3-3b",
    "ministral-3-8b-reasoning",
    "ministral-3-8b",
}


def load_causal_model(model_name):
    model_id = CAUSAL_MODELS[model_name]

    if model_name in MINISTRAL_MODELS:
        tokenizer = MistralCommonBackend.from_pretrained(
            model_id
        )

        model = Mistral3ForConditionalGeneration.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
            device_map="auto",
        )

    else:
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch.float16,
            device_map="auto",
        )

    return tokenizer, model