def generate_response(model, tokenizer, prompt, model_name):

    if model_name == "alpaca-7b":
        inputs = tokenizer(
            prompt,
            return_tensors="pt"
        )
    else:
        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]

        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        )

    inputs = inputs.to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=100
    )

    input_length = inputs["input_ids"].shape[1]

    response = tokenizer.decode(
        outputs[0][input_length:],
        skip_special_tokens=True
    )

    return response.strip()