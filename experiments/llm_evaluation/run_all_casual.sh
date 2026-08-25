#!/bin/bash

MODELS=(
    "qwen-0.5b"
    "llama-3.2-3b"
    "llama-2-chat-7b"
    "qwen-2.5-7b"
    "llama-8b"
    "llama-3-8b"
)

DATASET="src/kgqa/data/grailqa/grailqa_triples_text.json"
RESULTS_DIR="results/llm_evaluation/triples"
LOG_DIR="results/llm_evaluation/logs"

mkdir -p "$LOG_DIR"

for MODEL in "${MODELS[@]}"; do

    echo "========================================"
    echo "Starting model: $MODEL"
    echo "========================================"

    python -m experiments.llm_evaluation.run_experiment \
        --model "$MODEL" \
        --dataset "$DATASET" \
        --results-dir "$RESULTS_DIR" \
        > "$LOG_DIR/${MODEL}.log" 2>&1

    STATUS=$?

    if [ $STATUS -eq 0 ]; then
        echo "$MODEL completed successfully."
    else
        echo "$MODEL failed with status $STATUS."
    fi

done

echo "========================================"
echo "All models finished."
echo "========================================"