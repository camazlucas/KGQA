import argparse

import matplotlib.pyplot as plt

from plot_utils import load_data, save_figure


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    df = load_data(args.input)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.scatter(
        df["average_time_per_question_seconds"],
        df["f1"],
        s=80,
    )

    for _, row in df.iterrows():
        ax.annotate(
            row["model_label"],
            (
                row["average_time_per_question_seconds"],
                row["f1"],
            ),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=9,
        )

    ax.set_xlabel("Tempo médio por pergunta (s)")
    ax.set_ylabel("F1")

    ax.grid(
        True,
        alpha=0.3,
    )

    fig.tight_layout()

    save_figure(
        fig,
        args.output,
    )

    plt.close(fig)


if __name__ == "__main__":
    main()