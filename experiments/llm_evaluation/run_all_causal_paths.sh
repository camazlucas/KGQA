#!/bin/bash

MODELS=(
    "qwen-0.5b"
    "llama-3.2-3b"
    "llama-2-chat-7b"
    "qwen-2.5-7b"
    "llama-8b"
    "llama-3-8b"
    "deepseek-llm-7b-chat"
    "deepseek-r1-distill-qwen-1.5b"
    "ministral-3-3b"
    "ministral-3-8b"
)

DATASET="src/kgqa/data/grailqa/grailqa_filtered_paths2.json"
RESULTS_DIR="results/llm_evaluation/paths"
LOG_DIR="results/llm_evaluation/logs"

mkdir -p "$LOG_DIR"

for MODEL in "${MODELS[@]}"; do

    echo "========================================"
    echo "Starting model: $MODEL"
    echo "========================================"

    python -m experiments.llm_evaluation.run_experiment_paths \
        --model "$MODEL" \
        --dataset "$DATASET" \
        --results-dir "$RESULTS_DIR" \
        > "$LOG_DIR/${MODEL}_paths.log" 2>&1

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