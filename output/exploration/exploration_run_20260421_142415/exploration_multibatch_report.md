# Exploration Strategy Multi-Batch Report

This report compares exploration strategies only. It is independent from reward shaping experiments.

Total runs: 18
Total batches: 3
Strategies: 6

## Ranking by Mean Best Score
1. softmax_anneal (softmax): score_mean=8464.00, score_std=1620.26, score_max=10752.00, tile_mean=682.67, tile_max=1024, win_rate=33.3% (1/3)
2. softmax_hot (softmax): score_mean=8373.33, score_std=260.43, score_max=8640.00, tile_mean=512.00, tile_max=512, win_rate=66.7% (2/3)
3. ucb (ucb): score_mean=7329.33, score_std=279.05, score_max=7608.00, tile_mean=512.00, tile_max=512, win_rate=0.0% (0/3)
4. eps_fast (epsilon_greedy): score_mean=3137.33, score_std=262.18, score_max=3508.00, tile_mean=256.00, tile_max=256, win_rate=0.0% (0/3)
5. random (random): score_mean=3104.00, score_std=48.33, score_max=3144.00, tile_mean=256.00, tile_max=256, win_rate=0.0% (0/3)
6. eps_slow (epsilon_greedy): score_mean=2674.67, score_std=152.54, score_max=2840.00, tile_mean=256.00, tile_max=256, win_rate=0.0% (0/3)

## Batch Winners by Score
- batch 1: softmax_hot
- batch 2: softmax_hot
- batch 3: softmax_anneal

## Output Files
- Per-run details CSV: output\exploration\exploration_run_20260421_142415\exploration_batch_results.csv
- Aggregated strategy CSV: output\exploration\exploration_run_20260421_142415\exploration_strategy_aggregate.csv
- Plot: output\exploration\exploration_run_20260421_142415\exploration_multibatch_impact.png
