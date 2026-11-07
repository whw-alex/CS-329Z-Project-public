# Learning Developer Preferences from Coding-Agent Logs

CS 329Z project by Haowen Wang, Christy Yang and Zhen Jia. The system forecasts whether a developer will push back on a coding agent's next turn. It learns each developer's preferences from earlier sessions in the same repository. The data comes from [SWE-chat](https://huggingface.co/datasets/SALT-NLP/SWE-chat).

The agent code is in [preference-agent/](preference-agent/). Its [README](preference-agent/README.md) describes the pipeline, the configuration options and the evaluation commands. The data processing code and the exploration notebook are in this directory.

## Run the prototype without an API key

You need Python 3.10 or later. Run these commands from the repository root:

```sh
cd preference-agent
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt -r requirements-eval.txt
python -m unittest discover -s tests
python -m preference_agent example.json --backend demo --output demo-result.json
```

The tests (131 at the time of writing) use a dummy language model and make no network calls. The last command runs the full pipeline on a synthetic example. Its probabilities are hand-written, so they are not a result.

To run real forecasts, you need an OpenRouter or Gemini API key. Read "Local model configuration" in [preference-agent/README.md](preference-agent/README.md). These runs make paid API calls.

## Get the SWE-chat data

SWE-chat is a gated dataset on Hugging Face. Before you download it, accept its terms on the dataset page and log in with `hf auth login`. Then run these commands from the repository root:

```sh
pip install -r requirements.txt
python scripts/download_conversations.py   # writes data/raw/conversations.parquet (about 6.4 GB)
python scripts/build_turns.py              # writes data/turns_compact.parquet
jupyter lab eda.ipynb
```

The notebook downloads `sessions.parquet`, `repositories.parquet` and `conv_turns.parquet` into `data/` through `scripts/load_data.py`. Its last cell writes `data/targets.parquet` and `data/repo_split.parquet`.

## What this repository contains

This public repository contains no SWE-chat conversation text. It keeps the code, the tests, the synthetic examples and the results at the level of IDs, labels and metrics:

- `data/english-targets/targets_english.parquet` lists the English target turns by session ID and turn number, with SWE-chat's labels.
- Each `labels.csv` and `selection.csv` in `preference-agent/work/` holds our relabeled targets and history turns. The evidence quotes and the labeler's reasoning are removed.
- Each evaluation folder in `preference-agent/work/` keeps its report, `plan.json`, `summary.json` and per-case CSV files with probabilities and scores.

The prepared cases (`case-*.json`), the per-case predictions (`result-*.json`, `baseline-*.json`), the relabeling episodes and the run logs are not included, because they contain verbatim conversation text. Some reports refer to these files. To rebuild the cases, download the raw data and run `preference_agent.relabel_cases` as described in [preference-agent/README.md](preference-agent/README.md).

`demo/` contains the template and the build script for the midpoint demo page. The script reads case and result files, so it works only after you rebuild an evaluation folder.

## Data license

SWE-chat is released under the ODC-By license by Baumann et al. (2026), "SWE-chat: Coding agent interactions from real users in the wild" ([arXiv:2604.20779](https://arxiv.org/abs/2604.20779)). The labels in `preference-agent/work/` come from a language model. Except for the 30-turn pilot, no human verified them.
