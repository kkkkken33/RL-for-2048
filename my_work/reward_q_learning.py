"""Reward-shaped linear Q-learning for 2048.

This script is a reward-driven variant of feature_q_learning.py.
It keeps the same Q-learning structure, but replaces the training signal
with a weighted sum of reward components:

    R_total = w1 * r1 + w2 * r2 + w3 * r3 + ...

The components are derived from board changes after each action, so you can
adjust the weights to encourage higher score, better board shape, and fewer
illegal moves.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import gymnasium as gym
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import env  # noqa: F401 - register 2048-v0
from env.envs.game2048_env import Game2048Env, IllegalMove


ACTION_NAMES = ("up", "right", "down", "left")


class TeeStream:
    """Mirror writes to multiple streams (console + file)."""

    def __init__(self, *streams) -> None:
        self._streams = streams

    def write(self, data: str) -> int:
        for stream in self._streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()


@dataclass
class TransitionStats:
    """Per-episode statistics for logging."""

    env_reward: float
    shaped_reward: float
    highest: int
    steps: int


class RewardShaper:
    """Build weighted reward components from a 2048 board transition."""

    def __init__(self) -> None:
        self._model_env = Game2048Env()

    @staticmethod
    def _log2_board(board: np.ndarray) -> np.ndarray:
        log_board = np.zeros_like(board, dtype=np.float64)
        mask = board > 0
        log_board[mask] = np.log2(board[mask])
        return log_board

    @staticmethod
    def _count_empty(board: np.ndarray) -> float:
        return float(np.sum(board == 0)) / 16.0

    @staticmethod
    def _max_tile_feature(board: np.ndarray) -> float:
        max_tile = float(np.max(board))
        if max_tile <= 0.0:
            return 0.0
        return float(np.log2(max_tile) / 15.0)

    @staticmethod
    def _monotonicity(board: np.ndarray) -> float:
        log_board = RewardShaper._log2_board(board)
        row_diff = np.diff(log_board, axis=1)
        col_diff = np.diff(log_board, axis=0)

        row_dec_viol = np.sum(np.maximum(0.0, row_diff))
        row_inc_viol = np.sum(np.maximum(0.0, -row_diff))
        col_dec_viol = np.sum(np.maximum(0.0, col_diff))
        col_inc_viol = np.sum(np.maximum(0.0, -col_diff))

        row_best = min(row_dec_viol, row_inc_viol)
        col_best = min(col_dec_viol, col_inc_viol)
        return float(-(row_best + col_best) / 24.0)

    @staticmethod
    def _smoothness(board: np.ndarray) -> float:
        log_board = RewardShaper._log2_board(board)

        right = np.abs(log_board[:, 1:] - log_board[:, :-1])
        down = np.abs(log_board[1:, :] - log_board[:-1, :])

        right_mask = (board[:, 1:] > 0) & (board[:, :-1] > 0)
        down_mask = (board[1:, :] > 0) & (board[:-1, :] > 0)

        penalty = np.sum(right[right_mask]) + np.sum(down[down_mask])
        return float(-penalty / 32.0)

    @staticmethod
    def _merge_potential(board: np.ndarray) -> float:
        horizontal = (board[:, :-1] == board[:, 1:]) & (board[:, :-1] > 0)
        vertical = (board[:-1, :] == board[1:, :]) & (board[:-1, :] > 0)
        pairs = int(np.sum(horizontal) + np.sum(vertical))
        return float(pairs / 24.0)

    @staticmethod
    def _corner_max(board: np.ndarray) -> float:
        max_tile = np.max(board)
        corners = [board[0, 0], board[0, 3], board[3, 0], board[3, 3]]
        return 1.0 if max_tile > 0 and max_tile in corners else 0.0

    def _state_features(self, board: np.ndarray) -> np.ndarray:
        return np.array(
            [
                self._count_empty(board),
                self._max_tile_feature(board),
                self._monotonicity(board),
                self._smoothness(board),
                self._merge_potential(board),
                self._corner_max(board),
            ],
            dtype=np.float64,
        )

    @staticmethod
    def get_weights(max_tile: int, empty_cells: int) -> np.ndarray:
        """Return adaptive reward weights based on the current board state."""
        safe_tile = max(2, int(max_tile))
        return np.array(
            [
                2.0,  # w1_env_score
                6.0 if empty_cells <= 4 else 1.5,  # w2_empty
                1.0 * np.log2(safe_tile),  # w3_max_tile
                4.0 if safe_tile >= 512 else 1.5,  # w4_monotonicity
                1.0,  # w5_smoothness
                2.0,  # w6_merge_potential
                8.0 if safe_tile >= 256 else 3.0,  # w7_corner_max
                -500.0,  # w8_illegal
            ],
            dtype=np.float64,
        )

    def simulate(self, board: np.ndarray, action: int) -> tuple[bool, np.ndarray, float]:
        """Apply action on a copied board without random tile spawn."""
        self._model_env.set_board(np.array(board, copy=True))
        try:
            merge_score = float(self._model_env.move(action))
        except IllegalMove:
            return False, np.array(board, copy=True), 0.0
        return True, np.array(self._model_env.get_board(), copy=True), merge_score

    def components(self, board: np.ndarray, action: int) -> tuple[np.ndarray, bool, np.ndarray]:
        """Return reward components, legality, and the after-board."""
        legal, after_board, merge_score = self.simulate(board, action)
        before_features = self._state_features(board)

        if not legal:
            components = np.array(
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                ],
                dtype=np.float64,
            )
            return components, False, after_board

        after_features = self._state_features(after_board)
        components = np.array(
            [
                merge_score / 2048.0,
                after_features[0] - before_features[0],
                after_features[1] - before_features[1],
                after_features[2] - before_features[2],
                after_features[3] - before_features[3],
                after_features[4] - before_features[4],
                after_features[5] - before_features[5],
                0.0,
            ],
            dtype=np.float64,
        )
        return components, True, after_board

    @staticmethod
    def component_names() -> tuple[str, ...]:
        return (
            "env_score",
            "empty_delta",
            "max_tile_delta",
            "monotonicity_delta",
            "smoothness_delta",
            "merge_potential_delta",
            "corner_max_delta",
            "illegal_move",
        )


def get_board_state(env_instance: gym.Env) -> np.ndarray:
    """Return the current board from the underlying Game2048Env."""
    game_env = cast(Game2048Env, env_instance.unwrapped)
    return np.array(game_env.get_board(), copy=True)


class LinearQAgent:
    """Linear function approximation Q-learning agent."""

    def __init__(
        self,
        feature_builder: RewardShaper,
        alpha: float,
        gamma: float,
        epsilon: float,
        epsilon_min: float,
        epsilon_decay: float,
        seed: int,
        reward_weights: np.ndarray,
        weight_scales: np.ndarray,
    ) -> None:
        self.feature_builder = feature_builder
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.reward_weights = reward_weights.astype(np.float64)
        self.weight_scales = weight_scales.astype(np.float64)

        self.num_actions = 4
        phi_dim = 7
        self.weights = np.zeros((self.num_actions, phi_dim), dtype=np.float64)
        self.rng = np.random.default_rng(seed)

    def _phi(self, board: np.ndarray, action: int) -> np.ndarray:
        legal, after_board, _ = self.feature_builder.simulate(board, action)
        if not legal:
            return np.array(
                [
                    1.0,
                    -1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                dtype=np.float64,
            )

        after_features = self.feature_builder._state_features(after_board)
        return np.array(
            [
                1.0,
                after_features[0],
                after_features[1],
                after_features[2],
                after_features[3],
                after_features[4],
                after_features[5],
            ],
            dtype=np.float64,
        )

    def q_value(self, board: np.ndarray, action: int) -> float:
        phi = self._phi(board, action)
        return float(np.dot(self.weights[action], phi))

    def legal_actions(self, board: np.ndarray) -> list[int]:
        legal = []
        for action in range(self.num_actions):
            _, is_legal, _ = self.feature_builder.components(board, action)
            if is_legal:
                legal.append(action)
        return legal

    def select_action(self, board: np.ndarray) -> int:
        legal = self.legal_actions(board)
        if not legal:
            return int(self.rng.integers(0, self.num_actions))

        if self.rng.random() < self.epsilon:
            return int(self.rng.choice(legal))

        q_legal = [(a, self.q_value(board, a)) for a in legal]
        max_q = max(v for _, v in q_legal)
        best = [a for a, v in q_legal if np.isclose(v, max_q)]
        return int(self.rng.choice(best))

    def greedy_action(self, board: np.ndarray) -> int:
        legal = self.legal_actions(board)
        if not legal:
            return 0
        q_legal = [(a, self.q_value(board, a)) for a in legal]
        max_q = max(v for _, v in q_legal)
        best = [a for a, v in q_legal if np.isclose(v, max_q)]
        return int(self.rng.choice(best))

    def shaped_reward(self, board: np.ndarray, action: int) -> tuple[float, np.ndarray, bool]:
        components, legal, after_board = self.feature_builder.components(board, action)
        if not legal:
            reward = -500.0
            return reward, components, legal

        max_tile = int(np.max(after_board))
        empty_cells = int(np.sum(after_board == 0))
        adaptive_weights = self.feature_builder.get_weights(max_tile, empty_cells)
        scaled_weights = adaptive_weights * self.weight_scales
        reward = float(np.dot(scaled_weights, components))
        return reward, components, legal

    def update(
        self,
        board: np.ndarray,
        action: int,
        reward: float,
        next_board: np.ndarray,
        done: bool,
    ) -> float:
        phi_sa = self._phi(board, action)
        q_sa = float(np.dot(self.weights[action], phi_sa))

        if done:
            target = reward
        else:
            next_legal = self.legal_actions(next_board)
            if next_legal:
                next_q = max(self.q_value(next_board, a) for a in next_legal)
            else:
                next_q = 0.0
            target = reward + self.gamma * next_q

        td_error = target - q_sa
        self.weights[action] += self.alpha * td_error * phi_sa
        return float(td_error)

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            weights=self.weights,
            reward_weights=self.reward_weights,
            weight_scales=self.weight_scales,
            alpha=self.alpha,
            gamma=self.gamma,
            epsilon=self.epsilon,
            epsilon_min=self.epsilon_min,
            epsilon_decay=self.epsilon_decay,
        )

    def load(self, path: Path) -> None:
        data = np.load(path)
        self.weights = data["weights"].astype(np.float64)
        if "reward_weights" in data:
            self.reward_weights = data["reward_weights"].astype(np.float64)
        if "weight_scales" in data:
            self.weight_scales = data["weight_scales"].astype(np.float64)
        else:
            self.weight_scales = np.ones(8, dtype=np.float64)
        self.alpha = float(data["alpha"])
        self.gamma = float(data["gamma"])
        self.epsilon = float(data["epsilon"])
        self.epsilon_min = float(data["epsilon_min"])
        self.epsilon_decay = float(data["epsilon_decay"])


def run_episode(env_instance: gym.Env, agent: LinearQAgent) -> TransitionStats:
    """Run one epsilon-greedy training episode and update agent online."""
    env_instance.reset()
    board = get_board_state(env_instance)

    done = False
    episode_env_reward = 0.0
    episode_shaped_reward = 0.0
    steps = 0
    highest = int(np.max(board))

    while not done:
        action = agent.select_action(board)
        _, env_reward, terminated, truncated, info = env_instance.step(action)
        next_board = get_board_state(env_instance)
        done = bool(terminated or truncated)

        shaped_reward, _, _ = agent.shaped_reward(board, action)
        agent.update(board, action, float(shaped_reward), next_board, done)

        board = next_board
        highest = max(highest, int(info.get("highest", np.max(board))))
        episode_env_reward += float(env_reward)
        episode_shaped_reward += float(shaped_reward)
        steps += 1

    agent.decay_epsilon()
    return TransitionStats(
        env_reward=episode_env_reward,
        shaped_reward=episode_shaped_reward,
        highest=highest,
        steps=steps,
    )


def evaluate_greedy(env_instance: gym.Env, agent: LinearQAgent, episodes: int) -> tuple[float, float]:
    """Evaluate the policy without exploration using the real game score."""
    old_epsilon = agent.epsilon
    agent.epsilon = 0.0

    rewards = []
    highs = []

    for _ in range(episodes):
        env_instance.reset()
        board = get_board_state(env_instance)
        done = False
        total_reward = 0.0
        highest = int(np.max(board))

        while not done:
            action = agent.greedy_action(board)
            _, reward, terminated, truncated, info = env_instance.step(action)
            board = get_board_state(env_instance)
            done = bool(terminated or truncated)
            total_reward += float(reward)
            highest = max(highest, int(info.get("highest", np.max(board))))

        rewards.append(total_reward)
        highs.append(highest)

    agent.epsilon = old_epsilon
    return float(np.mean(rewards)), float(np.mean(highs))


def _format_seconds(total_seconds: float) -> str:
    """Format seconds to HH:MM:SS for readable training logs."""
    total_seconds = max(0, int(total_seconds))
    hours, rem = divmod(total_seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def train(args: argparse.Namespace) -> None:
    """Train the reward-shaped Q-learning agent."""
    if getattr(args, "run_dir", None):
        run_dir = Path(args.run_dir)
    else:
        run_timestamp = time.strftime("%Y%m%d_%H%M%S")
        run_dir = Path(args.output_dir) / f"reward_q_run_{run_timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    config_path = run_dir / "config.json"
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    train_csv_path = run_dir / "train_metrics.csv"
    eval_csv_path = run_dir / "eval_metrics.csv"

    with train_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "episode",
                "env_reward",
                "shaped_reward",
                "episode_highest",
                "episode_steps",
                "epsilon",
                "avg_env_reward_window",
                "avg_shaped_reward_window",
                "avg_high_window",
            ]
        )

    with eval_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "eval_episodes", "mean_reward", "mean_high"])

    print(f"Run directory: {run_dir}")
    print(f"Config saved: {config_path}")
    print(f"Training metrics CSV: {train_csv_path}")
    print(f"Eval metrics CSV: {eval_csv_path}")
    print("Adaptive reward schedule enabled:")
    print("  w1_env_score=2.0")
    print("  w2_empty=6.0 when empty_cells<=4 else 1.5")
    print("  w3_max_tile=1.0 * log2(max_tile)")
    print("  w4_monotonicity=4.0 when max_tile>=512 else 1.5")
    print("  w5_smoothness=1.0")
    print("  w6_merge_potential=2.0")
    print("  w7_corner_max=8.0 when max_tile>=256 else 3.0")
    print("  w8_illegal=-500.0")
    print(f"Weight scales (w1..w8): {args.weight_scales}")

    train_env = gym.make("2048-v0")
    eval_env = gym.make("2048-v0")

    shaper = RewardShaper()
    agent = LinearQAgent(
        feature_builder=shaper,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon=args.epsilon,
        epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay,
        seed=args.seed,
        reward_weights=np.ones(8, dtype=np.float64),
        weight_scales=np.array(args.weight_scales, dtype=np.float64),
    )

    if args.load is not None:
        load_path = Path(args.load)
        agent.load(load_path)
        print(f"Loaded checkpoint: {load_path}")

    env_window: list[float] = []
    shaped_window: list[float] = []
    high_window: list[int] = []
    best_reward = float("-inf")
    best_episode = 0
    best_model_path = run_dir / "reward_q_best_reward.npz"
    training_start_time = time.time()
    last_log_time = training_start_time

    for episode in range(1, args.episodes + 1):
        stats = run_episode(train_env, agent)
        env_window.append(stats.env_reward)
        shaped_window.append(stats.shaped_reward)
        high_window.append(stats.highest)

        if stats.env_reward > best_reward:
            best_reward = stats.env_reward
            best_episode = episode
            agent.save(best_model_path)
            print(
                f"  [best] new best env_reward={best_reward:.2f} "
                f"at ep={best_episode}, saved: {best_model_path}"
            )

        if len(env_window) > args.log_window:
            env_window.pop(0)
            shaped_window.pop(0)
            high_window.pop(0)

        avg_env_reward = float(np.mean(env_window))
        avg_shaped_reward = float(np.mean(shaped_window))
        avg_high = float(np.mean(high_window))

        with train_csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    episode,
                    stats.env_reward,
                    stats.shaped_reward,
                    stats.highest,
                    stats.steps,
                    agent.epsilon,
                    avg_env_reward,
                    avg_shaped_reward,
                    avg_high,
                ]
            )

        if episode % args.log_every == 0:
            now = time.time()
            interval_elapsed = now - last_log_time
            total_elapsed = now - training_start_time
            avg_episode_time = total_elapsed / episode
            eta_seconds = avg_episode_time * max(0, args.episodes - episode)
            last_log_time = now

            print(
                f"ep={episode:6d}  "
                f"avg_env_reward({len(env_window)})={avg_env_reward:8.2f}  "
                f"avg_shaped_reward={avg_shaped_reward:8.2f}  "
                f"avg_high={avg_high:7.2f}  "
                f"eps={agent.epsilon:0.4f}  "
                f"dt={_format_seconds(interval_elapsed)}  "
                f"elapsed={_format_seconds(total_elapsed)}  "
                f"eta={_format_seconds(eta_seconds)}"
            )

        if args.eval_every > 0 and episode % args.eval_every == 0:
            eval_reward, eval_high = evaluate_greedy(eval_env, agent, args.eval_episodes)
            with eval_csv_path.open("a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([episode, args.eval_episodes, eval_reward, eval_high])
            print(
                f"  [eval] episodes={args.eval_episodes} "
                f"mean_reward={eval_reward:.2f} mean_high={eval_high:.2f}"
            )

        if args.save_every > 0 and episode % args.save_every == 0:
            ckpt = run_dir / f"reward_q_ep{episode}.npz"
            agent.save(ckpt)
            print(f"  saved: {ckpt}")

    final_name = args.output_name or f"reward_q_final_{int(time.time())}.npz"
    final_path = run_dir / final_name
    agent.save(final_path)
    print(f"Training complete. Final model saved: {final_path}")
    if best_episode > 0:
        print(
            f"Best-reward model: {best_model_path} "
            f"(episode={best_episode}, reward={best_reward:.2f})"
        )

    train_env.close()
    eval_env.close()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Reward-shaped linear Q-learning for 2048-v0")
    parser.add_argument("--episodes", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--gamma", type=float, default=0.99)

    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--epsilon-min", type=float, default=0.05)
    parser.add_argument("--epsilon-decay", type=float, default=0.9995)

    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--log-window", type=int, default=100)

    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--eval-episodes", type=int, default=20)

    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--output-dir", type=str, default="./output")
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="Final checkpoint filename (default: timestamped)",
    )
    parser.add_argument(
        "--load",
        type=str,
        default=None,
        help="Load existing .npz checkpoint before training",
    )

    parser.add_argument(
        "--weight-scales",
        type=float,
        nargs=8,
        metavar=(
            "S_ENV",
            "S_EMPTY",
            "S_MAX",
            "S_MONO",
            "S_SMOOTH",
            "S_MERGE",
            "S_CORNER",
            "S_ILLEGAL",
        ),
        default=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
        help="Multiplicative scales applied to adaptive weights [w1..w8]",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_dir = output_dir / f"reward_q_run_{log_timestamp}"
    log_dir.mkdir(parents=True, exist_ok=True)
    args.run_dir = str(log_dir)
    log_path = log_dir / "train.log"

    with log_path.open("w", encoding="utf-8") as log_file:
        tee_out = TeeStream(sys.stdout, log_file)
        tee_err = TeeStream(sys.stderr, log_file)
        with redirect_stdout(tee_out), redirect_stderr(tee_err):
            print(f"Logging to: {log_path}")
            train(args)