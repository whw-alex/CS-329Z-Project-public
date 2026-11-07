#!/usr/bin/env bash
# B2/B3 baselines with GPT-6 Luna on the k4-30 full-history cases. Paid OpenRouter calls (~cents).
set -euo pipefail
cd "$(dirname "$0")"
source .venv/bin/activate
python -c "import numpy, sklearn, pandas, pyarrow" 2>/dev/null || pip install -q -r requirements-eval.txt -r requirements-data.txt
DIR=work/k4-30-full-luna
MODEL=openai/gpt-6-luna
if [ ! -d "$DIR" ]; then
  python -m preference_agent.relabel_cases --relabel-dir work/k4-30 --labels-file selection.csv \
    --history-labels work/k4-30/history/labels.csv --k 4 --output-dir "$DIR"
fi
echo "== Trial: 1 case =="
python -m preference_agent.baselines --cases-dir "$DIR" --env-file .env --model "$MODEL" --limit 1 > /dev/null
grep -h '"status"' "$DIR"/baseline-b2-0001.json "$DIR"/baseline-b3-0001.json
read -r -p "Trial OK? Run all 30 cases [y/N] " ok
[[ "$ok" == y* ]] || exit 0
python -m preference_agent.baselines --cases-dir "$DIR" --env-file .env --model "$MODEL"
echo "Results: $DIR/baselines-summary.json and $DIR/baselines-cases.csv"
