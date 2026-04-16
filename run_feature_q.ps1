param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExtraArgs
)

$ErrorActionPreference = "Stop"

# Feature Q-learning launcher (Windows PowerShell)
# Modify the values below instead of retyping long commands in terminal.

$EPISODES = 20000
$SEED = 42

$ALPHA = 0.01
$GAMMA = 0.99

$EPSILON = 1.0
$EPSILON_MIN = 0.05
$EPSILON_DECAY = 0.9995

$LOG_EVERY = 100
$LOG_WINDOW = 100

$EVAL_EVERY = 500
$EVAL_EPISODES = 20

$SAVE_EVERY = 1000
$OUTPUT_DIR = "./output"

# Optional:
# $OUTPUT_NAME = "feature_q_custom_final.npz"
# $LOAD_MODEL = "./output/feature_q_run_xxx/feature_q_final_xxx.npz"
$OUTPUT_NAME = ""
$LOAD_MODEL = "output\feature_q_run_20260416_192326\feature_q_ep3000.npz"

# Choose python executable: prefer project venv, then fallback to python in PATH.
if (Test-Path ".\venv\Scripts\python.exe") {
    $PYTHON_EXEC = ".\venv\Scripts\python.exe"
} elseif (Test-Path ".\venv\bin\python") {
    $PYTHON_EXEC = ".\venv\bin\python"
} else {
    $PYTHON_EXEC = "python"
}

$cmdArgs = @(
    ".\feature_q_learning.py",
    "--episodes", "$EPISODES",
    "--seed", "$SEED",
    "--alpha", "$ALPHA",
    "--gamma", "$GAMMA",
    "--epsilon", "$EPSILON",
    "--epsilon-min", "$EPSILON_MIN",
    "--epsilon-decay", "$EPSILON_DECAY",
    "--log-every", "$LOG_EVERY",
    "--log-window", "$LOG_WINDOW",
    "--eval-every", "$EVAL_EVERY",
    "--eval-episodes", "$EVAL_EPISODES",
    "--save-every", "$SAVE_EVERY",
    "--output-dir", "$OUTPUT_DIR"
)

if (-not [string]::IsNullOrWhiteSpace($OUTPUT_NAME)) {
    $cmdArgs += @("--output-name", $OUTPUT_NAME)
}

if (-not [string]::IsNullOrWhiteSpace($LOAD_MODEL)) {
    $cmdArgs += @("--load", $LOAD_MODEL)
}

# Allow one-off overrides, e.g.:
# .\run_feature_q.ps1 --episodes 5000 --alpha 0.005
if ($ExtraArgs -and $ExtraArgs.Count -gt 0) {
    $cmdArgs += $ExtraArgs
}

Write-Host "Running command:"
Write-Host "$PYTHON_EXEC $($cmdArgs -join ' ')"

& $PYTHON_EXEC @cmdArgs
exit $LASTEXITCODE
