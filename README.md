# 2048 Reinforcement Learning Project (Gymnasium + PyTorch)

This project implements a 2048 game environment, along with various training and evaluation pipelines, including:

- 2048 Gymnasium environment (Env ID: `2048-v0`)
- Supervised learning data collection and processing (CSV)
- PPO training pipeline (Stable-Baselines3)
- Feature-based Q-learning training and visualization

Key environment information:

- Observation: `(16, 4, 4)`, channels-first one-hot
- Actions: 4 discrete actions, `0=Up, 1=Right, 2=Down, 3=Left`

## Project Structure

Core files:

- [env/envs/game2048_env.py](env/envs/game2048_env.py): 2048 environment implementation
- [env/envs/test_game2048_env.py](env/envs/test_game2048_env.py): Environment unit tests
- [training_data.py](training_data.py): Training data I/O and augmentation
- [feature_q_learning.py](feature_q_learning.py): Feature-based Q-learning training script
- [visualize_feature_q.py](visualize_feature_q.py): Feature-Q model evaluation/visualization script
- [run_feature_q.sh](run_feature_q.sh): Feature-Q training launcher (centralized config)
- [ppo_train.py](ppo_train.py): PPO training script
- [pretrain_bc.py](pretrain_bc.py): Behavior cloning pretraining script

## Visual Demo
<video controls src="2048_play.mp4" title="Play 2048"></video>

## Environment Setup

It is recommended to use a virtual environment.

```
python -m venv venv
```

Install dependencies:

```
pip install -r requirements.txt
```

## Testing

Test only the environment:

```
pytest env/envs/test_game2048_env.py
```

## Feature-based Q-learning

### Method Overview

Uses linear function approximation:

$$
Q(s, a) = w_a^T \phi(s, a)
$$

Features are extracted from the after-state (after taking the action), including empty tile ratio, max tile, monotonicity, smoothness, merge potential, corner max value, etc.

### Training Commands

Run directly:

```
python .\feature_q_learning.py
```

Specify parameters:

```
python .\feature_q_learning.py --episodes 20000 --alpha 0.01 --gamma 0.99 --eval-every 500
```

Use launcher script (Linux/WSL only):

```bash
sh run_feature_q.sh
```
Windows:
```
.\run_feature_q.ps1
```

Temporarily override script parameters:

```bash
sh run_feature_q.sh --episodes 5000 --alpha 0.005
```
Windows:
```
.\run_feature_q.ps1 --episodes 1 --log-every 1 --eval-every 0 --save-every 0
```

### Training Output

Each training run saves to `output/feature_q_run_<timestamp>/`:

- `train.log`: Full training log
- `config.json`: Parameters for this run
- `train_metrics.csv`: Per-episode training metrics
- `eval_metrics.csv`: Periodic evaluation metrics
- `feature_q_ep*.npz`: Intermediate checkpoints
- `feature_q_final_*.npz`: Final model

## Model Visualization & Evaluation

Use [visualize_feature_q.py](visualize_feature_q.py) to load `.npz` models for evaluation.

Examples:

```powershell
python .\visualize_feature_q.py --model .\output\feature_q_run_xxx\feature_q_final_xxx.npz --episodes 3 --mode human
```

```powershell
python .\visualize_feature_q.py --model .\output\feature_q_run_xxx\feature_q_final_xxx.npz --episodes 3 --mode ansi
```

```powershell
python .\visualize_feature_q.py --model .\output\feature_q_run_xxx\feature_q_final_xxx.npz --episodes 3 --mode video
```

### Visualization Modes

- `human`: Real-time image window rendering, suitable for manual observation
- `ansi`: Terminal text board output, suitable for SSH/headless environments
- `video`: Record mp4 video, suitable for playback and reporting

### Visualization Output

Each visualization run saves to `output/feature_q_visualize_<timestamp>/`:

- `config.json`: Evaluation parameters
- `episode_metrics.csv`: Per-episode reward/highest/steps
- `summary.json`: Mean and max summary
- `videos/*.mp4`: Only generated when `mode=video`

## Parameter Descriptions

### Training Parameters (feature_q_learning.py)

- `--episodes`: Number of training episodes
- `--alpha`: Learning rate
- `--gamma`: Discount factor
- `--epsilon`: Initial exploration rate
- `--epsilon-min`: Minimum exploration rate
- `--epsilon-decay`: Exploration rate decay
- `--log-every`: Log output interval
- `--log-window`: Moving average window
- `--eval-every`: Evaluation interval (0 to disable)
- `--eval-episodes`: Number of episodes per evaluation
- `--save-every`: Checkpoint interval (0 to disable)
- `--output-dir`: Output root directory
- `--output-name`: Final model filename (optional)
- `--load`: Continue training from existing `.npz` model

### Visualization Parameters (visualize_feature_q.py)

- `--model`: Model path (required)
- `--episodes`: Number of evaluation episodes
- `--mode`: `human | ansi | video`
- `--seed`: Random seed
- `--output-dir`: Output root directory

### Plotting Training Curves
```
python plot_reward_curves.py --run-dir=<path to training folder>
```

## PPO & BC (Optional Pipelines)

- Behavior cloning pretraining: [pretrain_bc.py](pretrain_bc.py)
- PPO training: [ppo_train.py](ppo_train.py)

Examples:

```powershell
.\venv\Scripts\python.exe .\pretrain_bc.py data\test_data.csv
.\venv\Scripts\python.exe .\ppo_train.py --total-timesteps 5000000
```

## FAQ

- `mode=video` fails to generate video: Please ensure `moviepy` is installed
- Font error causes rendering failure: Fallback fonts are built-in, no extra changes needed
- Import errors when running test files directly: Prefer running `pytest` from the project root

## License

MIT, see [LICENSE.txt](LICENSE.txt) for details.

