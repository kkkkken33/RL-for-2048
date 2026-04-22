# RewardQ 权重影响报告（w1-w8）

## 1. 数据来源
- 扫描汇总数据: [output/sweep_single_weights_all/single_weight_sweep_summary.csv](output/sweep_single_weights_all/single_weight_sweep_summary.csv)
- 扫描文本摘要: [output/sweep_single_weights_all/single_weight_sweep_summary.md](output/sweep_single_weights_all/single_weight_sweep_summary.md)
- 可视化图: [output/sweep_single_weights_all/single_weight_sweep_impact.png](output/sweep_single_weights_all/single_weight_sweep_impact.png)
- 训练脚本: [my_work/reward_q_learning.py](my_work/reward_q_learning.py)
- 扫描脚本: [my_work/sweep_single_weights.py](my_work/sweep_single_weights.py)

## 2. 实验设置
- 每次试验 episode: 100
- 扫描方式: 单独调整一个权重，其他权重保持 scale=1.0
- 扫描 scale: 0.5, 0.8, 1.2, 1.5
- 指标:
  - 最大得分 max_score
  - 最大卡面 max_tile
  - 最大生存步数 max_survival_steps
  - 后100局均值（作为稳定性参考）

## 3. Baseline
- baseline max_score: 2752
- baseline max_tile: 256
- baseline max_survival_steps: 222
- baseline avg_score_last100: 1048.72
- baseline avg_tile_last100: 102.40
- baseline avg_steps_last100: 115.52

## 4. w1-w8 逐项影响（有数据支撑）

### w1_env_score
- 最优（按 max_score）: scale=1.5, max_score=3064, Δscore=+312, max_steps=254, Δsteps=+32
- 最差（按 max_score）: scale=0.5, max_score=2400, Δscore=-352, max_steps=201, Δsteps=-21
- 解读: 提高 w1 对冲高分和延长生存都有正面作用，且趋势较稳定。

### w2_empty
- 最优: scale=1.5, max_score=2860, Δscore=+108, max_steps=237, Δsteps=+15
- 最差: scale=0.5, max_score=2428, Δscore=-324, max_steps=204, Δsteps=-18
- 解读: w2 过低会明显损伤生存和上限分；适度提高可提升稳定性。

### w3_max_tile
- 最优: scale=0.8, max_score=2848, Δscore=+96, max_steps=234, Δsteps=+12
- 最差: scale=1.2, max_score=2384, Δscore=-368, max_steps=198, Δsteps=-24
- 解读: w3 在该阶段存在“过强反而变差”的现象，当前更适合中等偏低（约 0.8）。

### w4_monotonicity
- 最优: scale=0.8, max_score=3060, Δscore=+308, max_steps=254, Δsteps=+32
- 最差: scale=1.5, max_score=2400, Δscore=-352, max_steps=202, Δsteps=-20
- 解读: w4 对上限和生存非常敏感。过强会束缚策略，适度降低（0.8）明显更好。

### w5_smoothness
- 最优: scale=1.2, max_score=2964, Δscore=+212, max_steps=242, Δsteps=+20
- 最差: scale=0.8, max_score=2604, Δscore=-148, max_steps=210, Δsteps=-12
- 解读: w5 适当提高有利于质量和生存，但过低会退化。

### w6_merge_potential
- 最优: scale=0.5, max_score=2976, Δscore=+224, max_steps=242, Δsteps=+20
- 最差: scale=1.2, max_score=2332, Δscore=-420, max_steps=197, Δsteps=-25
- 解读: 当前配置下，w6 过高会明显伤害上限分；偏低反而更利于冲分。

### w7_corner_max
- 最优: scale=1.5, max_score=3076, Δscore=+324, max_steps=247, Δsteps=+25
- 最差: scale=0.5, max_score=2476, Δscore=-276, max_steps=212, Δsteps=-10
- 解读: w7 提高是有效方向，尤其对高分和生存有正向贡献。

### w8_illegal
- 最优: scale=1.5, max_score=2940, Δscore=+188, max_steps=247, Δsteps=+25
- 最差: scale=0.8, max_score=2448, Δscore=-304, max_steps=205, Δsteps=-17
- 解读: 非法动作惩罚偏强（1.5）比偏弱更稳，说明约束不足会导致浪费步数。

## 5. 关键观察
- 本轮所有 trial 的 max_tile 都是 256，说明 100 episode 预算下，卡面上限尚未拉开。
- 真正区分明显的是 max_score 和 max_survival_steps。
- 敏感权重（按绝对分数波动）主要是:
  - w6_merge_potential（最大负向波动大）
  - w4_monotonicity（正负都很敏感）
  - w7_corner_max（提升潜力高）
  - w1_env_score（稳态增益明显）

## 6. 面向高分目标（20000-25000）的建议
- 推荐先试一组偏冲分的 scale（基于本轮最优方向）:
  - w1=1.5, w2=1.5, w3=0.8, w4=0.8, w5=1.2, w6=0.5, w7=1.5, w8=1.5
- 对应命令:

```powershell
python .\my_work\reward_q_learning.py --episodes 8000 --eval-every 500 --eval-episodes 10 --weight-scales 1.5 1.5 0.8 0.8 1.2 0.5 1.5 1.5
```

- 建议做两阶段:
  - 阶段A（筛选）: 1000-2000 episode 比较候选 scale
  - 阶段B（冲刺）: 8000+ episode 只跑前2名配置

## 7. 结论
- 数据支持下，w1、w4、w7、w6 是当前最关键的四个旋钮。
- 方向上应当“提高 w1/w7，适度降低 w4，明显降低 w6 过强约束”，并保持较强 w8 约束。
- 目前短预算扫描已能给出明确方向，但要验证 20000+ 得分，必须用长训练（8k-20k episode）进行最终确认。
