def generate_response(model, tokenizer, prompt):

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    )

    inputs = inputs.to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=100
    )

    response = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return response.strip()