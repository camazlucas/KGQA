$baseDir = "results/llm_evaluation/paths/grailqa_paths_text"

$models = @(
    "qwen-0.5b",
    "llama-3.2-3b",
    "llama-2-chat-7b",
    "qwen-2.5-7b",
    "llama-8b",
    "llama-3-8b",
    "deepseek-llm-7b-chat",
    "deepseek-r1-distill-qwen-1.5b"
)

foreach ($model in $models) {

    $input = "$baseDir/$model/predictions.json"
    $output = "$baseDir/$model/evaluation.json"

    Write-Host "========================================"
    Write-Host "Evaluating: $model"
    Write-Host "========================================"

    if (-not (Test-Path $input)) {
        Write-Host "WARNING: predictions.json not found for $model"
        continue
    }

    python -m experiments.llm_evaluation.evaluate_results `
        --input $input `
        --output $output

    if ($LASTEXITCODE -eq 0) {
        Write-Host "$model evaluated successfully."
    }
    else {
        Write-Host "$model evaluation FAILED."
    }
}

Write-Host "========================================"
Write-Host "All evaluations completed."
Write-Host "========================================"