import argparse
import subprocess
import sys
from pathlib import Path


SCRIPTS = [
    "parameters_vs_hits_at_1.py",
    "f1_vs_time.py",
    "hits_at_1_vs_time.py",
    "f1_vs_vram.py",
    "hits_at_1_vs_vram.py",
    "models_vs_hits_at_1.py",
    "models_vs_f1.py",
    "models_vs_vram.py",
    "models_vs_time.py",
    "models_vs_all_metrics.py",
]


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

    output_dir = Path(args.output)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    script_dir = Path(__file__).resolve().parent

    for script in SCRIPTS:
        script_path = script_dir / script
        output_path = output_dir / script.replace(
            ".py",
            ".png",
        )

        print(f"Generating {script}...")

        subprocess.run(
            [
                sys.executable,
                str(script_path),
                "--input",
                args.input,
                "--output",
                str(output_path),
            ],
            check=True,
        )

    print("All plots generated successfully.")


if __name__ == "__main__":
    main()