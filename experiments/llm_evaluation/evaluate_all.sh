#!/bin/bash

BASE_DIR="results/llm_evaluation/triples/grailqa_triples_text"

MODELS=(
    "qwen-0.5b"
    "llama-3.2-3b"
    "llama-2-chat-7b"
    "qwen-2.5-7b"
    "llama-8b"
    "llama-3-8b"
)

for MODEL in "${MODELS[@]}"; do

    INPUT="$BASE_DIR/$MODEL/predictions.json"
    OUTPUT="$BASE_DIR/$MODEL/evaluation.json"

    echo "========================================"
    echo "Evaluating: $MODEL"
    echo "========================================"

    python -m experiments.llm_evaluation.evaluate_results \
        --input "$INPUT" \
        --output "$OUTPUT"

    STATUS=$?

    if [ $STATUS -eq 0 ]; then
        echo "$MODEL evaluated successfully."
    else
        echo "$MODEL evaluation FAILED."
    fi

done

echo "========================================"
echo "All evaluations completed."
echo "========================================"