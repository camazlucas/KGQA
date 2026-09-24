import argparse

from plot_utils import create_scatter, load_data


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

    create_scatter(
        df=df,
        x="average_time_per_question_seconds",
        y="hits_at_1",
        xlabel="Tempo médio por pergunta (s)",
        ylabel="Hits@1",
        output=args.output,
    )


if __name__ == "__main__":
    main()