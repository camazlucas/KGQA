from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


MODEL_SIZE = {
    "qwen-0.5b": 0.5,
    # "deepseek-r1-distill-qwen-1.5b": 1.5,
    "flan-t5-xl": 3.0,
    "llama-3.2-3b": 3.0,

    "llama-2-chat-7b": 7.0,
    "qwen-2.5-7b": 7.0,
    # "deepseek-llm-7b-chat": 7.0,
    "rog": 7.0,

    "llama-3-8b": 8.0,
    "llama-8b": 8.0,
}


MODEL_LABELS = {
    "qwen-0.5b": "Qwen-0.5B",
    # "deepseek-r1-distill-qwen-1.5b": "DeepSeek-R1-Distill-Qwen-1.5B",
    "flan-t5-xl": "Flan-T5-XL",
    "llama-3.2-3b": "Llama-3.2-3B",
    "llama-2-chat-7b": "Llama-2-Chat-7B",
    "qwen-2.5-7b": "Qwen-2.5-7B",
    # "deepseek-llm-7b-chat": "DeepSeek-LLM-7B-Chat",
    "rog": "Fine-Tunned Llama-7B RoG",
    "llama-3-8b": "Llama-3-8B",
    "llama-8b": "Llama-8B",
}


def load_data(path):
    """Load and prepare LLM evaluation data."""

    df = pd.read_csv(path)

    df["parameters_b"] = df["model"].map(MODEL_SIZE)
    df["model_label"] = df["model"].map(MODEL_LABELS)

    known_models = set(MODEL_SIZE) & set(MODEL_LABELS)

    df = df[
        df["model"].isin(known_models)
    ].copy()

    return df

def create_scatter(
    df,
    x,
    y,
    xlabel,
    ylabel,
    output,
    figsize=(10, 6),
):
    """Create and save a scatter plot."""

    fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(
        df[x],
        df[y],
        s=80,
    )

    for _, row in df.iterrows():
        ax.annotate(
            row["model_label"],
            (
                row[x],
                row[y],
            ),
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

    save_figure(
        fig,
        output,
    )

    plt.close(fig)


def save_figure(fig, output):
    """Save a matplotlib figure."""

    output_path = Path(output)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

def sort_by_model_size(df):
    """Sort models by number of parameters."""

    return df.sort_values(
        by=["parameters_b", "model_label"],
        ascending=[True, True],
    ).copy()


def create_bar_chart(
    df,
    metric,
    xlabel,
    ylabel,
    output,
    figsize=(10, 6),
):
    """Create and save a horizontal bar chart."""

    plot_df = sort_by_model_size(df)

    fig, ax = plt.subplots(figsize=figsize)

    ax.barh(
        plot_df["model_label"],
        plot_df[metric],
        height=0.7,
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.grid(
        axis="x",
        alpha=0.3,
    )

    fig.tight_layout()

    save_figure(
        fig,
        output,
    )

    plt.close(fig)


def normalize_metric(series, inverse=False):
    """Normalize a metric to the [0, 1] range."""

    min_value = series.min()
    max_value = series.max()

    if max_value == min_value:
        return pd.Series(
            1.0,
            index=series.index,
        )

    normalized = (
        series - min_value
    ) / (
        max_value - min_value
    )

    if inverse:
        normalized = 1 - normalized

    return normalized

def create_all_metrics_bar_chart(
    df,
    output,
    figsize=(14, 8),
):
    """Create and save a grouped bar chart with normalized metrics."""

    plot_df = sort_by_model_size(df)

    plot_df["hits_at_1_normalized"] = normalize_metric(
        plot_df["hits_at_1"],
    )

    plot_df["f1_normalized"] = normalize_metric(
        plot_df["f1"],
    )

    plot_df["vram_normalized"] = normalize_metric(
        plot_df["gpu_peak_memory_allocated_gb"],
        inverse=True,
    )

    plot_df["time_normalized"] = normalize_metric(
        plot_df["average_time_per_question_seconds"],
        inverse=True,
    )

    x = range(len(plot_df))
    width = 0.2

    fig, ax = plt.subplots(figsize=figsize)

    ax.bar(
        [i - 1.5 * width for i in x],
        plot_df["hits_at_1_normalized"],
        width,
        label="Hits@1",
    )

    ax.bar(
        [i - 0.5 * width for i in x],
        plot_df["f1_normalized"],
        width,
        label="F1",
    )

    ax.bar(
        [i + 0.5 * width for i in x],
        plot_df["vram_normalized"],
        width,
        label="VRAM",
    )

    ax.bar(
        [i + 1.5 * width for i in x],
        plot_df["time_normalized"],
        width,
        label="Tempo",
    )

    ax.set_xticks(list(x))
    ax.set_xticklabels(
        plot_df["model_label"],
        rotation=45,
        ha="right",
    )

    ax.set_xlabel("Modelo")
    ax.set_ylabel("Score normalizado")

    ax.set_ylim(0, 1.05)

    ax.grid(
        axis="y",
        alpha=0.3,
    )

    ax.legend()

    fig.tight_layout()

    save_figure(
        fig,
        output,
    )

    plt.close(fig)