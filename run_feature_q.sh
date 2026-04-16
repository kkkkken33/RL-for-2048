#!/usr/bin/env sh
set -eu

# Feature Q-learning launcher
# Modify the values below instead of retyping long commands in terminal.

EPISODES=20000
SEED=42

ALPHA=0.01
GAMMA=0.99

EPSILON=1.0
EPSILON_MIN=0.05
EPSILON_DECAY=0.9995

LOG_EVERY=100
LOG_WINDOW=100

EVAL_EVERY=500
EVAL_EPISODES=20

SAVE_EVERY=1000
OUTPUT_DIR=./output

# Optional:
# OUTPUT_NAME="feature_q_custom_final.npz"
# LOAD_MODEL="./output/feature_q_run_xxx/feature_q_final_xxx.npz"
OUTPUT_NAME=""
LOAD_MODEL=""

# Choose python executable: prefer project venv, then fallback to python in PATH.
if [ -x "./venv/Scripts/python.exe" ]; then
  PYTHON_EXEC="./venv/Scripts/python.exe"
elif [ -x "./venv/bin/python" ]; then
  PYTHON_EXEC="./venv/bin/python"
else
  PYTHON_EXEC="python"
fi

CMD="$PYTHON_EXEC ./feature_q_learning.py \
  --episodes $EPISODES \
  --seed $SEED \
  --alpha $ALPHA \
  --gamma $GAMMA \
  --epsilon $EPSILON \
  --epsilon-min $EPSILON_MIN \
  --epsilon-decay $EPSILON_DECAY \
  --log-every $LOG_EVERY \
  --log-window $LOG_WINDOW \
  --eval-every $EVAL_EVERY \
  --eval-episodes $EVAL_EPISODES \
  --save-every $SAVE_EVERY \
  --output-dir $OUTPUT_DIR"

if [ -n "$OUTPUT_NAME" ]; then
  CMD="$CMD --output-name $OUTPUT_NAME"
fi

if [ -n "$LOAD_MODEL" ]; then
  CMD="$CMD --load $LOAD_MODEL"
fi

# Allow one-off overrides, e.g.:
# sh run_feature_q.sh --episodes 5000 --alpha 0.005
if [ "$#" -gt 0 ]; then
  CMD="$CMD $*"
fi

echo "Running command:"
echo "$CMD"

# shellcheck disable=SC2086
exec $CMD
