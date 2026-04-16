"""Feature-based linear Q-learning for 2048.

This script trains a linear action-value function:
	Q(s, a) = w_a^T phi(s, a)

The feature vector phi(s, a) is extracted from the deterministic after-state
produced by applying action a (without random tile spawn), plus one immediate
score feature. This avoids tabular state explosion while retaining useful
structure about board quality.
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

import gymnasium as gym
import numpy as np

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
	"""One transition statistics for episodic logging."""

	reward: float
	highest: int
	steps: int


class FeatureExtractor:
	"""Builds action-conditional features phi(s, a) for 2048 boards."""

	def __init__(self) -> None:
		# Small internal model env used only for deterministic action simulation.
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
		# Typical practical cap in this project is near 2^15.
		return float(np.log2(max_tile) / 15.0)

	@staticmethod
	def _monotonicity(board: np.ndarray) -> float:
		"""Higher is better. Uses negative sum of violations in rows/cols."""
		log_board = FeatureExtractor._log2_board(board)
		row_diff = np.diff(log_board, axis=1)
		col_diff = np.diff(log_board, axis=0)

		row_dec_viol = np.sum(np.maximum(0.0, row_diff))
		row_inc_viol = np.sum(np.maximum(0.0, -row_diff))
		col_dec_viol = np.sum(np.maximum(0.0, col_diff))
		col_inc_viol = np.sum(np.maximum(0.0, -col_diff))

		row_best = min(row_dec_viol, row_inc_viol)
		col_best = min(col_dec_viol, col_inc_viol)
		# Negated and lightly scaled to roughly [-1, 0].
		return float(-(row_best + col_best) / 24.0)

	@staticmethod
	def _smoothness(board: np.ndarray) -> float:
		"""Higher is smoother. Penalizes log-value differences of neighbors."""
		log_board = FeatureExtractor._log2_board(board)

		right = np.abs(log_board[:, 1:] - log_board[:, :-1])
		down = np.abs(log_board[1:, :] - log_board[:-1, :])

		# Only compare non-empty neighbors.
		right_mask = (board[:, 1:] > 0) & (board[:, :-1] > 0)
		down_mask = (board[1:, :] > 0) & (board[:-1, :] > 0)

		penalty = np.sum(right[right_mask]) + np.sum(down[down_mask])
		return float(-penalty / 32.0)

	@staticmethod
	def _merge_potential(board: np.ndarray) -> float:
		"""Counts adjacent equal non-zero pairs (horizontal + vertical)."""
		horizontal = (board[:, :-1] == board[:, 1:]) & (board[:, :-1] > 0)
		vertical = (board[:-1, :] == board[1:, :]) & (board[:-1, :] > 0)
		pairs = int(np.sum(horizontal) + np.sum(vertical))
		# Max neighboring pairs on 4x4 grid: 24.
		return float(pairs / 24.0)

	@staticmethod
	def _corner_max(board: np.ndarray) -> float:
		"""1 if max tile is in any corner, else 0."""
		max_tile = np.max(board)
		corners = [board[0, 0], board[0, 3], board[3, 0], board[3, 3]]
		return 1.0 if max_tile > 0 and max_tile in corners else 0.0

	def simulate(self, board: np.ndarray, action: int) -> tuple[bool, np.ndarray, float]:
		"""Apply action on a copied board without random tile spawn."""
		self._model_env.set_board(np.array(board, copy=True))
		try:
			merge_score = float(self._model_env.move(action))
		except IllegalMove:
			return False, np.array(board, copy=True), 0.0
		return True, np.array(self._model_env.get_board(), copy=True), merge_score

	def phi(self, board: np.ndarray, action: int) -> tuple[np.ndarray, bool]:
		"""Return (feature_vector, is_legal) for a state-action pair."""
		legal, after_board, merge_score = self.simulate(board, action)
		if not legal:
			# Keep illegal-action values low and stable.
			features = np.array([
				1.0,   # bias
				-1.0,  # illegal indicator (negative)
				0.0,   # empty ratio
				0.0,   # max tile
				-1.0,  # monotonicity
				-1.0,  # smoothness
				0.0,   # merge potential
				0.0,   # immediate merge score
				0.0,   # corner max
			], dtype=np.float64)
			return features, False

		features = np.array([
			1.0,
			0.0,
			self._count_empty(after_board),
			self._max_tile_feature(after_board),
			self._monotonicity(after_board),
			self._smoothness(after_board),
			self._merge_potential(after_board),
			merge_score / 2048.0,
			self._corner_max(after_board),
		], dtype=np.float64)
		return features, True


class LinearQAgent:
	"""Linear function approximation Q-learning agent."""

	def __init__(
		self,
		feature_extractor: FeatureExtractor,
		alpha: float,
		gamma: float,
		epsilon: float,
		epsilon_min: float,
		epsilon_decay: float,
		seed: int,
	) -> None:
		self.feature_extractor = feature_extractor
		self.alpha = alpha
		self.gamma = gamma
		self.epsilon = epsilon
		self.epsilon_min = epsilon_min
		self.epsilon_decay = epsilon_decay

		self.num_actions = 4
		phi_dim = len(self.feature_extractor.phi(np.zeros((4, 4), dtype=int), 0)[0])
		self.weights = np.zeros((self.num_actions, phi_dim), dtype=np.float64)
		self.rng = np.random.default_rng(seed)

	def q_value(self, board: np.ndarray, action: int) -> float:
		phi, _ = self.feature_extractor.phi(board, action)
		return float(np.dot(self.weights[action], phi))

	def legal_actions(self, board: np.ndarray) -> list[int]:
		legal = []
		for action in range(self.num_actions):
			_, is_legal = self.feature_extractor.phi(board, action)
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

	def update(
		self,
		board: np.ndarray,
		action: int,
		reward: float,
		next_board: np.ndarray,
		done: bool,
	) -> float:
		phi_sa, _ = self.feature_extractor.phi(board, action)
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
			alpha=self.alpha,
			gamma=self.gamma,
			epsilon=self.epsilon,
			epsilon_min=self.epsilon_min,
			epsilon_decay=self.epsilon_decay,
		)

	def load(self, path: Path) -> None:
		data = np.load(path)
		self.weights = data["weights"].astype(np.float64)
		self.alpha = float(data["alpha"])
		self.gamma = float(data["gamma"])
		self.epsilon = float(data["epsilon"])
		self.epsilon_min = float(data["epsilon_min"])
		self.epsilon_decay = float(data["epsilon_decay"])


def run_episode(env_instance: gym.Env, agent: LinearQAgent) -> TransitionStats:
	"""Run one epsilon-greedy training episode and update agent online."""
	env_instance.reset()
	board = np.array(env_instance.unwrapped.get_board(), copy=True)

	done = False
	episode_reward = 0.0
	steps = 0
	highest = int(np.max(board))

	while not done:
		action = agent.select_action(board)
		_, reward, terminated, truncated, info = env_instance.step(action)
		next_board = np.array(env_instance.unwrapped.get_board(), copy=True)
		done = bool(terminated or truncated)

		agent.update(board, action, float(reward), next_board, done)

		board = next_board
		highest = max(highest, int(info.get("highest", np.max(board))))
		episode_reward += float(reward)
		steps += 1

	agent.decay_epsilon()
	return TransitionStats(reward=episode_reward, highest=highest, steps=steps)


def evaluate_greedy(env_instance: gym.Env, agent: LinearQAgent, episodes: int) -> tuple[float, float]:
	"""Evaluate current policy without exploration."""
	old_epsilon = agent.epsilon
	agent.epsilon = 0.0

	rewards = []
	highs = []

	for _ in range(episodes):
		env_instance.reset()
		board = np.array(env_instance.unwrapped.get_board(), copy=True)
		done = False
		total_reward = 0.0
		highest = int(np.max(board))

		while not done:
			action = agent.greedy_action(board)
			_, reward, terminated, truncated, info = env_instance.step(action)
			board = np.array(env_instance.unwrapped.get_board(), copy=True)
			done = bool(terminated or truncated)
			total_reward += float(reward)
			highest = max(highest, int(info.get("highest", np.max(board))))

		rewards.append(total_reward)
		highs.append(highest)

	agent.epsilon = old_epsilon
	return float(np.mean(rewards)), float(np.mean(highs))


def train(args: argparse.Namespace) -> None:
	"""Train feature-based Q-learning and periodically report progress."""
	if getattr(args, "run_dir", None):
		run_dir = Path(args.run_dir)
	else:
		run_timestamp = time.strftime("%Y%m%d_%H%M%S")
		run_dir = Path(args.output_dir) / f"feature_q_run_{run_timestamp}"
	run_dir.mkdir(parents=True, exist_ok=True)

	config_path = run_dir / "config.json"
	with config_path.open("w", encoding="utf-8") as f:
		json.dump(vars(args), f, indent=2, ensure_ascii=False)

	train_csv_path = run_dir / "train_metrics.csv"
	eval_csv_path = run_dir / "eval_metrics.csv"

	with train_csv_path.open("w", newline="", encoding="utf-8") as f:
		writer = csv.writer(f)
		writer.writerow([
			"episode",
			"episode_reward",
			"episode_highest",
			"episode_steps",
			"epsilon",
			"avg_reward_window",
			"avg_high_window",
		])

	with eval_csv_path.open("w", newline="", encoding="utf-8") as f:
		writer = csv.writer(f)
		writer.writerow(["episode", "eval_episodes", "mean_reward", "mean_high"])

	print(f"Run directory: {run_dir}")
	print(f"Config saved: {config_path}")
	print(f"Training metrics CSV: {train_csv_path}")
	print(f"Eval metrics CSV: {eval_csv_path}")

	train_env = gym.make("2048-v0")
	eval_env = gym.make("2048-v0")

	extractor = FeatureExtractor()
	agent = LinearQAgent(
		feature_extractor=extractor,
		alpha=args.alpha,
		gamma=args.gamma,
		epsilon=args.epsilon,
		epsilon_min=args.epsilon_min,
		epsilon_decay=args.epsilon_decay,
		seed=args.seed,
	)

	if args.load is not None:
		load_path = Path(args.load)
		agent.load(load_path)
		print(f"Loaded checkpoint: {load_path}")

	reward_window: list[float] = []
	high_window: list[int] = []

	for episode in range(1, args.episodes + 1):
		stats = run_episode(train_env, agent)
		reward_window.append(stats.reward)
		high_window.append(stats.highest)

		if len(reward_window) > args.log_window:
			reward_window.pop(0)
			high_window.pop(0)

		avg_reward = float(np.mean(reward_window))
		avg_high = float(np.mean(high_window))

		with train_csv_path.open("a", newline="", encoding="utf-8") as f:
			writer = csv.writer(f)
			writer.writerow([
				episode,
				stats.reward,
				stats.highest,
				stats.steps,
				agent.epsilon,
				avg_reward,
				avg_high,
			])

		if episode % args.log_every == 0:
			print(
				f"ep={episode:6d}  "
				f"avg_reward({len(reward_window)})={avg_reward:8.2f}  "
				f"avg_high={avg_high:7.2f}  "
				f"eps={agent.epsilon:0.4f}"
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
			ckpt = run_dir / f"feature_q_ep{episode}.npz"
			agent.save(ckpt)
			print(f"  saved: {ckpt}")

	final_name = args.output_name or f"feature_q_final_{int(time.time())}.npz"
	final_path = run_dir / final_name
	agent.save(final_path)
	print(f"Training complete. Final model saved: {final_path}")

	train_env.close()
	eval_env.close()


def parse_args() -> argparse.Namespace:
	"""Parse command-line arguments."""
	parser = argparse.ArgumentParser(description="Feature-based linear Q-learning for 2048-v0")
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
	parser.add_argument("--output-name", type=str, default=None,
						help="Final checkpoint filename (default: timestamped)")
	parser.add_argument("--load", type=str, default=None,
						help="Load existing .npz checkpoint before training")
	return parser.parse_args()


if __name__ == "__main__":
	args = parse_args()
	output_dir = Path(args.output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)
	log_timestamp = time.strftime("%Y%m%d_%H%M%S")
	log_dir = output_dir / f"feature_q_run_{log_timestamp}"
	log_dir.mkdir(parents=True, exist_ok=True)
	args.run_dir = str(log_dir)
	log_path = log_dir / "train.log"

	with log_path.open("w", encoding="utf-8") as log_file:
		tee_out = TeeStream(sys.stdout, log_file)
		tee_err = TeeStream(sys.stderr, log_file)
		with redirect_stdout(tee_out), redirect_stderr(tee_err):
			print(f"Logging to: {log_path}")
			train(args)
