import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


MODEL_SIZE = {
    "qwen-0.5b": 0.5,
    "deepseek-r1-distill-qwen-1.5b": 1.5,
    "flan-t5-xl": 3.0,
    "llama-3.2-3b": 3.0,
    "llama-2-chat-7b": 7.0,
    "qwen-2.5-7b": 7.0,
    "deepseek-llm-7b-chat": 7.0,
    "rog": 7.0,
    "llama-3-8b": 8.0,
    "llama-8b": 8.0,
}


MODEL_LABELS = {
    "qwen-0.5b": "Qwen-0.5B",
    "deepseek-r1-distill-qwen-1.5b": "DeepSeek-R1-Distill-Qwen-1.5B",
    "flan-t5-xl": "Flan-T5-XL",
    "llama-3.2-3b": "Llama-3.2-3B",
    "llama-2-chat-7b": "Llama-2-Chat-7B",
    "qwen-2.5-7b": "Qwen-2.5-7B",
    "deepseek-llm-7b-chat": "DeepSeek-LLM-7B-Chat",
    "llama-3-8b": "Llama-3-8B",
    "llama-8b": "Llama-8B",
    "rog": "ROG"
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate plots for LLM evaluation results."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to the evaluation CSV file.",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where plots will be saved.",
    )

    return parser.parse_args()


def load_data(path):
    df = pd.read_csv(path)

    df["parameters_b"] = df["model"].map(MODEL_SIZE)
    df["model_label"] = df["model"].map(MODEL_LABELS)

    missing_models = df[df["parameters_b"].isna()]["model"].unique()

    if len(missing_models) > 0:
        raise ValueError(
            "Models without parameter size mapping: "
            + ", ".join(missing_models)
        )

    return df


def create_scatter(
    df,
    x,
    y,
    xlabel,
    ylabel,
    filename,
    output_dir,
):
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.scatter(
        df[x],
        df[y],
        s=80,
    )

    for _, row in df.iterrows():
        ax.annotate(
            row["model_label"],
            (row[x], row[y]),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=9,
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.grid(
        True,
        alpha=0.3,
    )

    fig.tight_layout()

    fig.savefig(
        output_dir / filename,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = load_data(args.input)

    # 1. Quantidade de parâmetros vs Hits@1
    create_scatter(
        df=df,
        x="parameters_b",
        y="hits_at_1",
        xlabel="Quantidade de parâmetros (B)",
        ylabel="Hits@1",
        filename="parameters_vs_hits_at_1.png",
        output_dir=output_dir,
    )

    # 2. F1 vs Tempo
    create_scatter(
        df=df,
        x="average_time_per_question_seconds",
        y="f1",
        xlabel="Tempo médio por pergunta (s)",
        ylabel="F1",
        filename="f1_vs_time.png",
        output_dir=output_dir,
    )

    # 3. Hits@1 vs Tempo
    create_scatter(
        df=df,
        x="average_time_per_question_seconds",
        y="hits_at_1",
        xlabel="Tempo médio por pergunta (s)",
        ylabel="Hits@1",
        filename="hits_at_1_vs_time.png",
        output_dir=output_dir,
    )

    # 4. F1 vs VRAM
    create_scatter(
        df=df,
        x="gpu_peak_memory_allocated_gb",
        y="f1",
        xlabel="VRAM máxima alocada (GB)",
        ylabel="F1",
        filename="f1_vs_vram.png",
        output_dir=output_dir,
    )

    # 5. Hits@1 vs VRAM
    create_scatter(
        df=df,
        x="gpu_peak_memory_allocated_gb",
        y="hits_at_1",
        xlabel="VRAM máxima alocada (GB)",
        ylabel="Hits@1",
        filename="hits_at_1_vs_vram.png",
        output_dir=output_dir,
    )

    print(f"Plots saved to: {output_dir}")


if __name__ == "__main__":
    main()