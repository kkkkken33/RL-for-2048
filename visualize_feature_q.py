"""Visualize and evaluate a saved Feature-Q model on 2048-v0.

Usage examples:
    python visualize_feature_q.py --model output/feature_q_run_xxx/feature_q_final_xxx.npz --episodes 3 --mode human
    python visualize_feature_q.py --model output/feature_q_run_xxx/feature_q_final_xxx.npz --episodes 5 --mode ansi
    python visualize_feature_q.py --model output/feature_q_run_xxx/feature_q_final_xxx.npz --episodes 3 --mode video
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import numpy as np

import env  # noqa: F401 - ensure 2048-v0 is registered
from feature_q_learning import FeatureExtractor, LinearQAgent


def build_agent(model_path: Path, seed: int) -> LinearQAgent:
    """Create agent and load saved linear Q weights."""
    extractor = FeatureExtractor()
    agent = LinearQAgent(
        feature_extractor=extractor,
        alpha=0.01,
        gamma=0.99,
        epsilon=0.0,
        epsilon_min=0.0,
        epsilon_decay=1.0,
        seed=seed,
    )
    agent.load(model_path)
    agent.epsilon = 0.0
    return agent


def make_env(mode: str, video_dir: Path | None) -> gym.Env:
    """Construct visualization environment based on mode."""
    if mode == "video":
        if video_dir is None:
            raise ValueError("video_dir must be provided when mode=video")
        try:
            import moviepy  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "Video mode requires moviepy. Install with: "
                "./venv/Scripts/python.exe -m pip install moviepy"
            ) from exc
        base_env = gym.make("2048-v0", render_mode="rgb_array")
        return RecordVideo(
            base_env,
            video_folder=str(video_dir),
            episode_trigger=lambda _: True,
            name_prefix="feature_q",
            disable_logger=True,
        )

    if mode == "ansi":
        return gym.make("2048-v0", render_mode="ansi")

    return gym.make("2048-v0", render_mode="human")


def play_episode(env_instance: gym.Env, agent: LinearQAgent, mode: str) -> dict[str, float | int]:
    """Run one greedy episode and optionally render frames/text."""
    env_instance.reset()
    board = np.array(env_instance.unwrapped.get_board(), copy=True)

    done = False
    total_reward = 0.0
    steps = 0
    highest = int(np.max(board))

    while not done:
        action = agent.greedy_action(board)
        _, reward, terminated, truncated, info = env_instance.step(action)
        done = bool(terminated or truncated)
        board = np.array(env_instance.unwrapped.get_board(), copy=True)

        total_reward += float(reward)
        steps += 1
        highest = max(highest, int(info.get("highest", np.max(board))))

        if mode == "human":
            env_instance.render()
        elif mode == "ansi":
            text = env_instance.render()
            if hasattr(text, "getvalue"):
                print(text.getvalue(), end="")
            else:
                print(text)

    return {
        "reward": float(total_reward),
        "highest": int(highest),
        "steps": int(steps),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize saved Feature-Q model on 2048-v0")
    parser.add_argument("--model", required=True, help="Path to saved .npz model file")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", choices=["human", "ansi", "video"], default="human")
    parser.add_argument("--output-dir", type=str, default="./output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    run_stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / f"feature_q_visualize_{run_stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    video_dir = run_dir / "videos"
    if args.mode == "video":
        video_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "model": str(model_path),
        "episodes": args.episodes,
        "seed": args.seed,
        "mode": args.mode,
        "output_dir": str(run_dir),
    }
    with (run_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    agent = build_agent(model_path=model_path, seed=args.seed)
    env_instance = make_env(args.mode, video_dir if args.mode == "video" else None)

    rewards = []
    highs = []
    metrics_path = run_dir / "episode_metrics.csv"

    try:
        with metrics_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["episode", "reward", "highest", "steps"])

            for episode in range(1, args.episodes + 1):
                stats = play_episode(env_instance, agent, args.mode)
                writer.writerow([episode, stats["reward"], stats["highest"], stats["steps"]])

                rewards.append(float(stats["reward"]))
                highs.append(int(stats["highest"]))

                print(
                    f"ep={episode:4d} reward={stats['reward']:8.2f} "
                    f"highest={stats['highest']:6d} steps={stats['steps']:4d}"
                )

        summary = {
            "episodes": args.episodes,
            "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
            "max_reward": float(np.max(rewards)) if rewards else 0.0,
            "mean_highest": float(np.mean(highs)) if highs else 0.0,
            "max_highest": int(np.max(highs)) if highs else 0,
        }
        with (run_dir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
    finally:
        # Ensure VideoRecorder flushes mp4 and avoids destructor warnings.
        env_instance.close()

    print("Visualization complete.")
    print(f"Artifacts saved to: {run_dir}")
    if args.mode == "video":
        print(f"Videos saved under: {video_dir}")
    print(f"Metrics: {metrics_path}")
    print(f"Summary: {run_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
