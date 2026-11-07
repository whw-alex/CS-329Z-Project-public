# k=4 relabeled set: 100 random targets and their history

Targets from data/english-targets/targets_english.parquet that support a fixed k=4 history: the repo's first 4
sessions (by start time) all end before the target session starts, and the target is in session 5 or later.
Drawn at random (seed 329, at most one target per session) from all k=4-valid targets, excluding every episode
labeled in earlier runs (relabel-pilot, relabel-v2-pilot, relabel-100, relabel-k4-extra, k4-30). Labels are model
labels (azure/openai/gpt-6.1-sol via the NVIDIA endpoint, prompt v2 in prompts/relabel_pushback.md), not yet
human-verified. The split is random, not balanced: 20 pushback, 73 non-pushback, 7 not feedback.

- labels.csv / labels.jsonl: the label for each target, plus SWE-chat's label.
- episodes.jsonl: the episode the labeler saw.
- verify.md: readable sheet for human verification.

- history/: every labelled turn in the first 4 sessions of the targets' 55 repos (1089 turns), labeled with the same
  prompt and model. `plan.json` lists each repo's 4 history sessions. 242 turns from 12 repos shared with k4-30 reuse
  the k4-30/history label (identical episodes; `reused_from` in labels.jsonl). Made with
  `python -m preference_agent.relabel --history-of work/k4-100/labels.csv --k 4 --output-dir <dir>`.

Use the two parts differently:
- Target labels (`labels.csv`, `labels.jsonl`) are answers: use them only for scoring, never as model input.
- History labels (`history/`) are inputs. Join them to history events by `session_id:turn_number`, the event id the
  data adapter uses. Drop turns labeled not feedback.
- `stated_preference` in either part is the labeler's extraction, kept for analysis. Do not give it to memory.

Three repos (6 of the 100 targets) have no labelled history turns, because their history sessions have only the
opening task message: blackgirlbytes/planetfall-seed-signalkit, lgulliver/derrick, user1303836/spectraplex.

This public copy does not include `episodes.jsonl`, `labels.jsonl` or `verify.md`, because they contain verbatim SWE-chat text. The label CSV files have no evidence, reasoning or notes columns.
