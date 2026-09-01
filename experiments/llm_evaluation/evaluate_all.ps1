param(
    [Parameter(Mandatory = $true)]
    [string]$BaseDir,

    [string[]]$Models = @(
        "qwen-0.5b",
        "llama-3.2-3b",
        "llama-2-chat-7b",
        "qwen-2.5-7b",
        "llama-8b",
        "llama-3-8b",
        "deepseek-llm-7b-chat",
        "deepseek-r1-distill-qwen-1.5b",
        "rog"
    )
)

foreach ($model in $Models) {
    $modelDir = Join-Path $BaseDir $model
    $input = Join-Path $modelDir "predictions.json"
    $output = Join-Path $modelDir "evaluation.json"

    Write-Host "========================================"
    Write-Host "Evaluating: $model"
    Write-Host "Results directory: $BaseDir"
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