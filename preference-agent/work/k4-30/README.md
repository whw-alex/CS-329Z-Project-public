# k=4 relabeled set: 15 pushback + 15 non-pushback

Targets from data/english-targets/targets_english.parquet that support a fixed k=4 history: the repo's first 4
sessions (by start time) all end before the target session starts, and the target is in session 5 or later.
Labels are model labels (azure/openai/gpt-6.1-sol via the NVIDIA endpoint, prompt v2 in
prompts/relabel_pushback.md), not yet human-verified. Drawn with seed 329 from all k=4-valid model-labeled
pushback/non-pushback cases in work/relabel-100 and work/relabel-k4-extra.

- selection.csv: the 30 cases with labels, history session ids, history end, target session start and gap.
- episodes.jsonl / labels.jsonl: the episode the labeler saw and its full output, renumbered by k4_id.
- verify.md: readable sheet for human verification.

- history/: the relabeled history the 30 cases are built from: all 449 labelled turns in the first 4 sessions of
  their 19 repos, labeled with the same prompt and model (123 pushback, 313 non-pushback, 13 not feedback).
  `plan.json` lists each repo's 4 history sessions. Made with
  `python -m preference_agent.relabel --history-of work/k4-30/selection.csv --k 4 --output-dir <dir>`.

Use the two parts differently:
- Target labels (`selection.csv`, `labels.jsonl`) are answers: use them only for scoring, never as model input.
- History labels (`history/`) are inputs. Join them to history events by `session_id:turn_number`, the event id the
  data adapter uses. Drop turns labeled not feedback.
- `stated_preference` in either part is the labeler's extraction, kept for analysis (e.g. how many preferences
  memory recovers). Do not give it to memory; for targets it comes from the reply being predicted.

One repo (blackgirlbytes/planetfall-seed-signalkit, 3 of the 30 cases) has no labelled history turns: each of its
4 history sessions has only the opening task message.

This public copy does not include `episodes.jsonl`, `labels.jsonl` or `verify.md`, because they contain verbatim SWE-chat text. The label CSV files have no evidence, reasoning or notes columns.
