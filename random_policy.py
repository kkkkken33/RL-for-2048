import gymnasium as gym
import numpy as np
import os
import csv
import env  # noqa: F401 - register 2048-v0
from env.envs.game2048_env import IllegalMove

# Output directory for results
random_output_dir = "./output/random_policy"
os.makedirs(random_output_dir, exist_ok=True)
random_csv_path = os.path.join(random_output_dir, "random_policy_results.csv")

# Create the 2048 environment
eval_env = gym.make("2048-v0")
episodes = 1000
rewards = []
highests = []

# Only perform legal actions
def legal_actions(env_instance: gym.Env) -> list[int]:
    """Return currently legal actions using trial moves on the unwrapped env."""
    legal = []
    base_env = env_instance.unwrapped
    for action in range(base_env.action_space.n):
        try:
            base_env.move(action, trial=True)
            legal.append(action)
        except IllegalMove:
            continue
    return legal

# Save results
with open(random_csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([
			"episode",
			"episode_reward",
			"episode_highest",
            "average_reward",
            "average_highest",
		])

# Evaluate the random policy
for episode in range(episodes):
    eval_env.reset()
    board = np.array(eval_env.unwrapped.get_board(), copy=True)
    highest = int(np.max(board))
    done = False
    total_reward = 0.0

    while not done:
        # Take a random legal action to avoid immediate termination by illegal moves.
        legal = legal_actions(eval_env)
        if not legal:
            break
        action = int(np.random.choice(legal))

        # Get the next state and reward
        obs, reward, terminated, truncated, info = eval_env.step(action)
        board = np.array(eval_env.unwrapped.get_board(), copy=True)
        total_reward += float(reward)
        done = bool(terminated or truncated)
        highest = max(highest, int(np.max(board)))

    rewards.append(float(total_reward))
    highests.append(highest)

    # Save the results for this episode
    with open(random_csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            episode + 1,
            total_reward,
            highest,
            np.mean(rewards) if rewards else 0.0,
            np.mean(highests) if highests else 0.0,
        ])


    rewards.append(total_reward)
    highests.append(highest)


eval_env.close()