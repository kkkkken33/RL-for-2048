from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import env  # noqa: F401 - register 2048-v0
import gymnasium as gym
import numpy as np


ACTION_NAMES: dict[int, str] = {
    0: "Up",
    1: "Right",
    2: "Down",
    3: "Left",
}


def observation_to_board(obs: np.ndarray) -> np.ndarray:
    """将环境观测转换为 4x4 的整数棋盘。

    支持：
    - (16, 4, 4) one-hot channels-first
    - (4, 4) 直接棋盘

    Args:
        obs: 环境返回的观测。

    Returns:
        形状为 (4, 4) 的棋盘，元素为 0 或 2 的幂。
    """
    if obs.shape == (4, 4):
        return obs.astype(np.int64)

    if obs.shape == (16, 4, 4):
        exp = np.argmax(obs, axis=0)  # 0 表示空，1 表示 2，2 表示 4，...
        board = np.where(exp == 0, 0, 2 ** exp)
        return board.astype(np.int64)

    raise ValueError(f"Unsupported observation shape: {obs.shape}")


def evaluate_board(board: np.ndarray) -> float:
    """对棋盘打分（越高越好）。

    简单启发式：
    - 空格越多越好
    - 最大块越大越好
    - 最大块在角上更好
    - 相邻格子 log2 差异越小越好（平滑性）

    Args:
        board: 4x4 棋盘。

    Returns:
        浮点分数。
    """
    empty = float(np.count_nonzero(board == 0))
    max_tile = int(board.max())
    max_log = float(np.log2(max_tile)) if max_tile > 0 else 0.0

    # 最大块角落奖励
    corners = [board[0, 0], board[0, 3], board[3, 0], board[3, 3]]
    corner_bonus = 1.0 if max_tile in corners else 0.0

    # 平滑性：相邻非空格子的 log2 差异越小越好
    log_board = np.zeros_like(board, dtype=np.float32)
    nonzero_mask = board > 0
    log_board[nonzero_mask] = np.log2(board[nonzero_mask]).astype(np.float32)

    smooth_penalty = 0.0
    for r in range(4):
        for c in range(4):
            if board[r, c] == 0:
                continue
            if r + 1 < 4 and board[r + 1, c] > 0:
                smooth_penalty += abs(log_board[r, c] - log_board[r + 1, c])
            if c + 1 < 4 and board[r, c + 1] > 0:
                smooth_penalty += abs(log_board[r, c] - log_board[r, c + 1])

    score = (
        3.0 * empty
        + 2.0 * corner_bonus
        + 1.0 * max_log
        - 0.1 * smooth_penalty
    )
    return float(score)


def choose_action(
    obs: np.ndarray,
    env_id: str = "2048-v0",
    seed: Optional[int] = None,
) -> int:
    """基于一步前瞻选择动作。

    对 4 个动作分别模拟一步，选取启发式分数最高的动作。

    Args:
        obs: 当前观测。
        env_id: Gymnasium 环境 ID。
        seed: 随机种子（用于复现）。

    Returns:
        动作编号（0=Up, 1=Right, 2=Down, 3=Left）。
    """
    current_board = observation_to_board(obs)
    best_action = 0
    best_score = -1e18

    for action in range(4):
        sim_env = gym.make(env_id)
        try:
            sim_env.reset(seed=seed)
            sim_env.unwrapped.set_board(current_board.copy())

            next_obs, reward, terminated, truncated, _ = sim_env.step(action)
            next_board = observation_to_board(next_obs)

            # 无效动作（棋盘不变）强惩罚
            if np.array_equal(next_board, current_board):
                score = -1e15
            else:
                score = evaluate_board(next_board) + 0.05 * float(reward)
                if terminated or truncated:
                    score -= 50.0

            if score > best_score:
                best_score = score
                best_action = action
        finally:
            sim_env.close()

    return best_action


@dataclass
class EpisodeResult:
    """单局结果。"""

    total_reward: float
    steps: int
    highest_tile: int


def play_episode(
    env_id: str = "2048-v0",
    seed: Optional[int] = None,
    render: bool = False,
    max_steps: int = 10_000,
) -> EpisodeResult:
    """运行一局 2048。"""
    env = gym.make(env_id, render_mode="human" if render else None)
    try:
        obs, _ = env.reset(seed=seed)
        total_reward = 0.0
        steps = 0

        done = False
        while not done and steps < max_steps:
            action = choose_action(obs, env_id=env_id, seed=None if seed is None else seed + steps)
            obs, reward, terminated, truncated, _ = env.step(action)

            total_reward += float(reward)
            steps += 1
            done = bool(terminated or truncated)

        board = observation_to_board(obs)
        highest_tile = int(board.max())

        return EpisodeResult(
            total_reward=total_reward,
            steps=steps,
            highest_tile=highest_tile,
        )
    finally:
        env.close()


def main() -> None:
    """命令行入口。"""
    parser = argparse.ArgumentParser(description="Simple heuristic player for 2048-v0.")
    parser.add_argument("--episodes", type=int, default=3, help="运行局数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--render", action="store_true", help="可视化运行")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output/heuristic_policy",
        help="CSV 输出目录",
    )
    parser.add_argument(
        "--csv-name",
        type=str,
        default="heuristic_policy_results.csv",
        help="CSV 文件名",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / args.csv_name

    rewards: list[float] = []
    highests: list[int] = []

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "episode",
            "episode_reward",
            "episode_highest",
            "average_reward",
            "average_highest",
        ])

        for i in range(args.episodes):
            result = play_episode(
                env_id="2048-v0",
                seed=args.seed + i,
                render=args.render,
            )

            rewards.append(float(result.total_reward))
            highests.append(int(result.highest_tile))

            avg_reward = float(np.mean(rewards)) if rewards else 0.0
            avg_highest = float(np.mean(highests)) if highests else 0.0

            writer.writerow([
                i + 1,
                float(result.total_reward),
                int(result.highest_tile),
                avg_reward,
                avg_highest,
            ])

            print(
                f"[Episode {i + 1}] "
                f"reward={result.total_reward:.1f}, "
                f"steps={result.steps}, "
                f"highest={result.highest_tile}, "
                f"avg_reward={avg_reward:.1f}, "
                f"avg_highest={avg_highest:.1f}"
            )

    print(f"CSV saved to: {csv_path}")


if __name__ == "__main__":
    main()