# 2048 强化学习项目（Gymnasium + PyTorch）

这个项目实现了一个 2048 游戏环境，以及多种训练与评估流程，包含：

- 2048 Gymnasium 环境（环境 ID：`2048-v0`）
- 监督学习数据采集与处理（CSV）
- PPO 训练流程（Stable-Baselines3）
- Feature-based Q-learning 训练与可视化

环境关键信息：

- 观测：`(16, 4, 4)`，channels-first one-hot
- 动作：4 个离散动作，`0=Up, 1=Right, 2=Down, 3=Left`

## 项目结构

核心文件：

- [env/envs/game2048_env.py](env/envs/game2048_env.py)：2048 环境实现
- [env/envs/test_game2048_env.py](env/envs/test_game2048_env.py)：环境单元测试
- [training_data.py](training_data.py)：训练数据读写与增强
- [feature_q_learning.py](feature_q_learning.py)：Feature-based Q-learning 训练脚本
- [visualize_feature_q.py](visualize_feature_q.py)：Feature-Q 模型评估/可视化脚本
- [run_feature_q.sh](run_feature_q.sh)：Feature-Q 训练启动脚本（集中配置参数）
- [ppo_train.py](ppo_train.py)：PPO 训练脚本
- [pretrain_bc.py](pretrain_bc.py)：行为克隆预训练脚本

## 环境安装

推荐使用虚拟环境。

```bash
python -m venv venv
```


安装依赖：

```bash
pip install -r requirements.txt
```

## 测试

只测环境：

```bash
pytest env/envs/test_game2048_env.py
```


## Feature-based Q-learning

### 方法简介

使用线性函数近似：

$$
Q(s, a) = w_a^T \phi(s, a)
$$

其中特征来自动作后的 after-state，包括空格比例、最大 tile、单调性、平滑度、合并潜力、角落最大值等。

### 训练命令

直接运行：

```powershell
python .\feature_q_learning.py
```

指定参数：

```powershell
python .\feature_q_learning.py --episodes 20000 --alpha 0.01 --gamma 0.99 --eval-every 500
```

使用启动脚本（仅在linux或wsl可用）：

```bash
sh run_feature_q.sh
```

临时覆盖脚本参数：

```bash
sh run_feature_q.sh --episodes 5000 --alpha 0.005
```

### 训练输出

每次训练会在 `output/feature_q_run_时间戳/` 下保存：

- `train.log`：完整训练日志
- `config.json`：本次参数
- `train_metrics.csv`：逐 episode 训练指标
- `eval_metrics.csv`：定期评估指标
- `feature_q_ep*.npz`：中间 checkpoint
- `feature_q_final_*.npz`：最终模型

## 模型可视化与评估

使用脚本 [visualize_feature_q.py](visualize_feature_q.py) 加载 `.npz` 模型进行评估。

示例：

```powershell
python .\visualize_feature_q.py --model .\output\feature_q_run_xxx\feature_q_final_xxx.npz --episodes 3 --mode human
```

```powershell
python .\visualize_feature_q.py --model .\output\feature_q_run_xxx\feature_q_final_xxx.npz --episodes 3 --mode ansi
```

```powershell
python .\visualize_feature_q.py --model .\output\feature_q_run_xxx\feature_q_final_xxx.npz --episodes 3 --mode video
```

### 可视化模式说明

- `human`：实时图像窗口渲染，适合人工观察
- `ansi`：终端文本棋盘输出，适合 SSH/无图形环境
- `video`：录制 mp4 视频，适合回放和汇报

### 可视化输出

每次可视化会在 `output/feature_q_visualize_时间戳/` 下保存：

- `config.json`：评估参数
- `episode_metrics.csv`：每局 reward/highest/steps
- `summary.json`：均值与最大值汇总
- `videos/*.mp4`：仅 `mode=video` 时生成

## 参数说明

### 训练参数（feature_q_learning.py）

- `--episodes`：训练回合数
- `--alpha`：学习率
- `--gamma`：折扣因子
- `--epsilon`：初始探索率
- `--epsilon-min`：最小探索率
- `--epsilon-decay`：探索率衰减
- `--log-every`：日志输出间隔
- `--log-window`：滑动平均窗口
- `--eval-every`：评估间隔（0 表示关闭）
- `--eval-episodes`：每次评估局数
- `--save-every`：checkpoint 间隔（0 表示关闭）
- `--output-dir`：输出根目录
- `--output-name`：最终模型文件名（可选）
- `--load`：从已有 `.npz` 模型继续训练

### 可视化参数（visualize_feature_q.py）

- `--model`：模型路径（必填）
- `--episodes`：评估局数
- `--mode`：`human | ansi | video`
- `--seed`：随机种子
- `--output-dir`：输出根目录

## PPO 与 BC（可选流程）

- 行为克隆预训练：[pretrain_bc.py](pretrain_bc.py)
- PPO 训练：[ppo_train.py](ppo_train.py)

示例：

```powershell
.\venv\Scripts\python.exe .\pretrain_bc.py data\test_data.csv
.\venv\Scripts\python.exe .\ppo_train.py --total-timesteps 5000000
```

## 常见问题

- `mode=video` 无法生成视频：请确认已安装 `moviepy`
- 字体报错导致渲染失败：环境已内置字体回退，不需要额外修改
- 直接运行测试文件出现导入问题：优先在项目根目录使用 `pytest`

## License

MIT，详见 [LICENSE.txt](LICENSE.txt)。

