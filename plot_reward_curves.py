"""Plot reward curves from Feature-Q training metrics.

Reads:
- train_metrics.csv (episode_reward / avg_reward_window)
- eval_metrics.csv (mean_reward)

Example:
    python plot_reward_curves.py --run-dir output/feature_q_run_20260416_195024
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot reward curves from training/eval CSV files")
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Directory containing train_metrics.csv and eval_metrics.csv",
    )
    parser.add_argument(
        "--train-csv",
        type=str,
        default=None,
        help="Path to train_metrics.csv (overrides --run-dir default)",
    )
    parser.add_argument(
        "--eval-csv",
        type=str,
        default=None,
        help="Path to eval_metrics.csv (overrides --run-dir default)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output image path (default: <run-dir>/reward_curves.png)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show plot window after saving",
    )
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    if args.run_dir is not None:
        run_dir = Path(args.run_dir)
        train_csv = Path(args.train_csv) if args.train_csv else run_dir / "train_metrics.csv"
        eval_csv = Path(args.eval_csv) if args.eval_csv else run_dir / "eval_metrics.csv"
        output = Path(args.output) if args.output else run_dir / "reward_curves.png"
        return train_csv, eval_csv, output

    if args.train_csv is None or args.eval_csv is None:
        raise ValueError("Either provide --run-dir, or provide both --train-csv and --eval-csv")

    train_csv = Path(args.train_csv)
    eval_csv = Path(args.eval_csv)
    output = Path(args.output) if args.output else Path("reward_curves.png")
    return train_csv, eval_csv, output


def main() -> None:
    args = parse_args()
    train_csv, eval_csv, output_path = resolve_paths(args)

    if not train_csv.exists():
        raise FileNotFoundError(f"train csv not found: {train_csv}")
    if not eval_csv.exists():
        raise FileNotFoundError(f"eval csv not found: {eval_csv}")

    train_episode: list[float] = []
    train_reward: list[float] = []
    train_avg_reward: list[float] = []
    with train_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required_train_cols = {"episode", "episode_reward", "avg_reward_window"}
        if reader.fieldnames is None:
            raise ValueError("train csv has no header")
        missing_train = required_train_cols - set(reader.fieldnames)
        if missing_train:
            raise ValueError(f"train csv missing required columns: {sorted(missing_train)}")
        for row in reader:
            train_episode.append(float(row["episode"]))
            train_reward.append(float(row["episode_reward"]))
            train_avg_reward.append(float(row["avg_reward_window"]))

    eval_episode: list[float] = []
    eval_mean_reward: list[float] = []
    with eval_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required_eval_cols = {"episode", "mean_reward"}
        if reader.fieldnames is None:
            raise ValueError("eval csv has no header")
        missing_eval = required_eval_cols - set(reader.fieldnames)
        if missing_eval:
            raise ValueError(f"eval csv missing required columns: {sorted(missing_eval)}")
        for row in reader:
            eval_episode.append(float(row["episode"]))
            eval_mean_reward.append(float(row["mean_reward"]))

    train_episode_arr = np.asarray(train_episode, dtype=float)
    train_reward_arr = np.asarray(train_reward, dtype=float)
    train_avg_reward_arr = np.asarray(train_avg_reward, dtype=float)
    eval_episode_arr = np.asarray(eval_episode, dtype=float)
    eval_mean_reward_arr = np.asarray(eval_mean_reward, dtype=float)

    plt.figure(figsize=(10, 6))

    plt.plot(
        train_episode_arr,
        train_reward_arr,
        alpha=0.25,
        linewidth=1.0,
        label="Train episode reward",
    )

    plt.plot(
        train_episode_arr,
        train_avg_reward_arr,
        linewidth=2.0,
        label="Train moving average reward",
    )

    if len(eval_episode_arr) > 0:
        plt.plot(
            eval_episode_arr,
            eval_mean_reward_arr,
            marker="o",
            linewidth=1.8,
            label="Eval mean reward",
        )

    plt.title("Feature-Q Reward Curves")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=160)
    print(f"Saved plot: {output_path}")

    if args.show:
        plt.show()

    plt.close()


if __name__ == "__main__":
    main()
