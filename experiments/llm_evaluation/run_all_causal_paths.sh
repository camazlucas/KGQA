#!/bin/bash
# uso: ./run_all_causal_paths.sh grailqa | webqsp | cwq

DATASET_NAME="$1"

if [ -z "$DATASET_NAME" ]; then
    echo "Uso: $0 <grailqa|webqsp|cwq>"
    exit 1
fi

case "$DATASET_NAME" in
    grailqa)
        DATASET="src/kgqa/data/grailqa/outputs/grailqa_filtered_paths.json"
        ;;
    webqsp)
        DATASET="src/kgqa/data/webqsp/outputs/webqsp_gold_paths.json"
        ;;
    cwq)
        DATASET="src/kgqa/data/cwq/outputs/cwq_gold_paths.json"
        ;;
    *)
        echo "Dataset desconhecido: $DATASET_NAME"
        exit 1
        ;;
esac

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

RESULTS_DIR="results/llm_evaluation/${DATASET_NAME}/paths"
LOG_DIR="results/llm_evaluation/${DATASET_NAME}/logs"

mkdir -p "$LOG_DIR"

for MODEL in "${MODELS[@]}"; do
    echo "========================================"
    echo "Starting model: $MODEL ($DATASET_NAME)"
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
echo "All models finished ($DATASET_NAME)."
echo "========================================"