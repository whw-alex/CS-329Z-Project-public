#!/usr/bin/env bash
# B2, B3, no_memory and agent with GPT-6 Luna on the k4-30 full-history cases (binary, relabeling definition).
# Paid OpenRouter calls. Re-running resumes; results already saved are skipped.
set -euo pipefail
cd "$(dirname "$0")"
source .venv/bin/activate
python -c "import numpy, sklearn, pandas, pyarrow" 2>/dev/null || pip install -q -r requirements-eval.txt -r requirements-data.txt
DIR=work/k4-30-luna-v2
MODEL=openai/gpt-6-luna
if [ ! -d "$DIR" ]; then
  python -m preference_agent.relabel_cases --relabel-dir work/k4-30 --labels-file selection.csv \
    --history-labels work/k4-30/history/labels.csv --k 4 --output-dir "$DIR"
fi

echo "== Trial: case 1, all four arms =="
python -m preference_agent.baselines --cases-dir "$DIR" --env-file .env --model "$MODEL" --limit 1 > /dev/null
python -m preference_agent.batch --output-dir "$DIR" --backend openrouter --env-file .env --model "$MODEL" \
  --memory-mode session --limit 1 > /dev/null
grep -h '"status"' "$DIR"/baseline-b2-0001.json "$DIR"/baseline-b3-0001.json | head -2
python -c "import json;r=json.load(open('$DIR/result-0001.json'));print('  agent pair:',r['status'],r['arm_status'],'calls',r['batch']['stage_calls'])"
read -r -p "Trial OK? Run all 30 cases [y/N] " ok
[[ "$ok" == y* ]] || exit 0

python -m preference_agent.baselines --cases-dir "$DIR" --env-file .env --model "$MODEL" > /dev/null
python -m preference_agent.batch --output-dir "$DIR" --backend openrouter --env-file .env --model "$MODEL" \
  --memory-mode session > /dev/null
python -m preference_agent.baselines --cases-dir "$DIR" --summary-only
echo "Results: $DIR/baselines-summary.json (all four arms), $DIR/summary.json (agent pairs)"
