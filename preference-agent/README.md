# Preference Agent: initial DSPy research framework

A prototype for forecasting the **next human reaction** to a coding-agent turn using repository-specific interaction history. It constructs candidate preference rules, checks them against independent evidence, and exposes scoped read-only history tools to a DSPy ReAct critic.

This forecasts feedback; it does not run a coding agent or edit a target repository. Prediction probabilities and rule confidence are uncalibrated. No model training or DSPy prompt optimization has been performed.

## Setup

Python 3.10+ is required; development and tests used Python 3.12 and DSPy 3.4.1.
From this directory:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
# Optional data adapters and offline evaluation, needed for the complete test suite:
python -m pip install -r requirements-eval.txt
python -m unittest discover -s tests -v
```

`requirements.lock.txt` records the installed DSPy runtime dependencies. Data/evaluation extras are listed separately. The current complete suite has 131 offline tests, including real DSPy JSONAdapter/ReAct execution with DummyLM, scope isolation, invalid evidence, citation feedback, shared budgets, sampling, paired checkpoints, session memory, the batch runner and the relabeling pipeline. No test needs a real API key.

Run the synthetic plumbing example without API access:

```sh
python -m preference_agent example.json --backend demo --output demo-result.json
```

Demo probabilities are hand-written and are not an effectiveness result.

## Local model configuration

```sh
cp .env.example .env
# Edit .env locally; leave it out of Git.
```

Set `OPENROUTER_API_KEY` and `OPENROUTER_MODEL` to an exact OpenRouter provider/model ID. The tested model was `google/gemini-3.5-flash-lite`; access depends on the account. `.env.example` contains empty fields, not credentials.

```sh
# This makes paid API calls.
python -m preference_agent example.json --backend openrouter --env-file .env \
  --critic-mode react --retrieval-policy autonomous --max-citation-repairs 1 \
  --max-react-iters 5 --max-tool-calls 4 --max-calls 60 --output prediction.json
```

`--env-file` loads only the specified file, does not interpolate variables, and does not override existing environment variables. Code reads credentials from the environment; never put credentials in code, examples, results, or notebooks. Git ignores local `.env`/`.env.*` files, virtual environments, generated cases and results. If a secret has already been committed, ignoring it is insufficient: revoke it and remove it from tracked content.

`--backend dspy` connects directly to Gemini using `GEMINI_API_KEY` and `GEMINI_MODEL`. Legacy `--backend gemini` uses the optional `requirements-gemini.txt` SDK. Both DSPy backends use JSONAdapter and disable LM caching and retries.

## Pipeline and memory

| Component | Implementation |
|---|---|
| Feedback miner | Validates and orders existing labeled events |
| Explainer | DSPy Predict extracts a narrow candidate rule |
| Verifier | DSPy Predict tests separate historical feedback |
| Memory curator | Exact-text merge and evidence-group confidence |
| Retrieval | Lexical rule/history search over scoped snapshots |
| Critic | DSPy ReAct autonomously selects tools and forecasts P(pushback) |
| Reviser | Optional experimental text rewriting; off by default |

`DSPyAgent` is the complete program; its nested predictors are visible to DSPy. Historical explanation/verification remain fixed stages. The critic is the autonomous agent. `core.Pipeline` also supports deterministic demo and fixed-critic baselines.

The task is binary. Every backend forecasts one P(pushback) under the relabeling standard: `pushback_definition.py` embeds the "What counts as pushback" definition of `prompts/relabel_pushback.md` in the agent, no_memory, B2 and B3 prompts, so editing that file changes labeling and forecasting together, and every label is `pushback` or `no_pushback`: SWE-chat's correction, rejection and failure_report load as `pushback` (`core.LEGACY_LABELS`), so older case files still work. All summaries report binary metrics only. The target is the next feedback after the supplied agent activity, not the already-visible user request. An assistant claim of success is not verified success; missing future feedback is not evidence of acceptance.

Event fields: `id`, `repo`, `order`, `context`, `agent_turn`, `follow_up`, `label`, optional `code_fate`, `evidence_group` and `user_actions` (rejected tool calls and interruptions inside `agent_turn`, extracted from raw rows by the data adapter). Current target inputs contain only `repo`, `target_order`, `context`, `response`; future feedback/labels must remain outside model inputs. Legacy labels (`non_pushback`, four-class names) are mapped on load.

A rule needs at least two independent support groups to become active. Same-session/task IDs and exact normalized duplicate contexts are conservatively grouped. This can undercount independent tasks within one session and cannot recognize all semantic duplicates across sessions. With k=1, most real rules remain candidates.

Verifier support/contradiction requires an exact substring of the event's current feedback, plus a reason. Invalid structured verdicts or quotes become `invalid_evidence` audit records and do not count as support. Other budget/network/program errors still stop the run. In event mode, no-pushback labels are conservatively excluded as positive preference support; session mode also permits explicitly stated reusable requirements. Exact quoting establishes traceability, not semantic correctness. Rule confidence is an evidence-count heuristic, not a calibrated probability.

### Memory modes

`--memory-mode event` (default) is the per-event pipeline above: one Explain call per historical pushback and one Verify call per rule and event from another session, so calls grow roughly quadratically with history.

`--memory-mode session` builds memory with a few calls per history session, so the cost depends on the number of sessions rather than pushbacks:

1. **Extract**, one call per session chunk (`--session-chunk-size`, default 20 turns): reusable preferences with polarity `prefers` or `avoids`, each with an exact quote from a turn's `follow_up` or `user_actions`.
2. **Merge**, one call per repository: equivalent same-polarity candidates from different sessions become one rule. Opposite polarities are never merged.
3. **Verify**, one call per session chunk: every rule against that chunk's turns, never against a session the rule came from.

With `--session-evidence stated` (default in session mode), evidence may also be an explicit statement in a no-pushback reply (e.g. "Python for now as it requires no setup") or a rejected tool call; the code rejects bare acceptance ("ok", "approved", "looks good") and quotes under 12 characters. `--session-evidence pushback` restricts evidence to pushback turns, like the event pipeline. Rules still need two independent sessions to become active. The ReAct critic additionally sees a free repository profile (historical label counts and pushback rate) and up to eight unconfirmed single-session candidate preferences, marked as such; it must read an episode before citing it.

## Scoped tools and citation feedback

Available tools:

- `search_preferences`: active rules and source IDs.
- `search_history`: event IDs and short feedback excerpts.
- `get_episode`: the allowed event's context, agent activity, and feedback.
- `find_counterexamples`: already-recorded contradictions, not exhaustive semantic search.
- `inspect_current_activity`: the supplied current agent activity.

Tools see only the supplied repository and events before target_order. They do not read files, execute code, access network, or reveal target feedback. Real data must also obey the fixed-k session protocol. Tool bindings are per request; inputs and quoted instructions are treated as untrusted data.

Default `retrieval_policy=autonomous` leaves search/read decisions to the model. Optional `search_first` spends one tool call on a harness-initiated historical search, explicitly marked in the trace. The critic is told how much history exists even when no active rule exists.

A cited event must have a readable `get_episode` result. Search summaries alone do not authorize citation. Tool-result truncation does not grant read access; already-truncated source events are marked partial_episode and omitted content must not be inferred.

For authorized but unread citations, the critic receives validation feedback and previous tool observations, then may autonomously read evidence or regenerate without unsupported historical claims. At most one retry is allowed (`--max-citation-repairs 0` disables it). The harness never reads events on the model's behalf. Unknown/future IDs, invalid probabilities, and truncated tool reads do not trigger this recovery. All attempts share the original decision, tool, and logical model-call budgets; each final extraction also consumes a model call.

Results retain the first rejected forecast and diagnostics in `attempts`, and distinguish `first_pass_success`, `read_and_cited`, and `regenerated_without_citations`. Internal model thoughts are not persisted in exported predictions. Natural-language grounding still needs manual review.

`max-calls` limits logical calls, not strict HTTP requests, tokens, or dollars. Adapters may make fallback requests. Paid experiments need separate per-request usage monitoring and a cost stop threshold. Long context is not automatically capped in the ordinary prediction CLI.

## SWE-chat data integration

Use the repository's data-processing instructions first. Required root-level files:

- `data/targets.parquet`, `data/repo_split.parquet`, `data/sessions.parquet`;
- `data/raw/conversations.parquet` and the existing `scripts/targets.py` rules.

The adapter joins target indices to full raw conversations, reuses the exact human-message/activity filters, and validates labels, duplicates, repository split, chronology and session overlap. It does not use EDA's 300-character compact text as model context. User IDs missing in the source limit developer-isolation guarantees.

From the repository root, with data extras installed in the chosen Python environment:

```sh
PYTHONPATH=preference-agent python -m preference_agent.data \
  --split holdout --k 4 --limit 4 --output-dir work/holdout-cases
```

Use a fresh output directory. `--skip-invalid` writes exclusion reasons to manifest.json; do not silently discard them. The first k sessions are frozen history for all later target sessions. `turn_number` indexes the feedback message; target input is the observable activity before it. `PreparedCase.model_inputs()` excludes expected labels. The CLI also supports `example-sessions.json` with explicit session times.

Optional pilot history budgets:

```sh
PYTHONPATH=preference-agent python -m preference_agent.data \
  --split holdout --k 4 --limit 4 --history-per-session 2 --history-seed 329 \
  --history-context-chars 1500 --history-agent-chars 2500 --history-feedback-chars 1000 \
  --output-dir work/holdout-budgeted-cases
```

Sampling uses only historical labels and temporal coverage, preserves positive/negative observations where capacity permits, and records source counts, selected IDs, label distributions and truncation in metadata.history_sampling. It changes historical label proportions. Character caps include explicit omission markers. Default export preserves full history; target context caps, if used, are the evaluation harness's responsibility.

## Paired and offline evaluation

From the agent directory:

```sh
# Paid: same ReAct controller, same target input; only historical access differs.
python -m preference_agent example.json --backend openrouter --env-file .env \
  --paired --retrieval-policy autonomous --max-citation-repairs 1 \
  --max-react-iters 5 --max-tool-calls 4 --max-calls 60 --output paired-result.json
```

`PairedEvaluator` supports batch use and reuses memory by repository plus a hash of the complete frozen history. It saves the no-memory result before memory work. Its on_update callback checkpoints independent arm results; validation failures yield partial/failed status without discarding the other arm. Fatal errors stop and retain a partial result. The CLI saves failed results and returns exit code 2 for an incomplete pair.

ForecastValidationError diagnostics distinguish summary-only citations, unknown IDs and truncated reads. Measure both first-pass and post-recovery success, actual episode reading/citation, and recovery costs. Only complete pairs belong in paired metrics; report all failures and coverage separately.

Balanced paired pilot over several cases, from the agent directory. Preparation makes no model calls; inspect `plan.json` and the case files before running:

```sh
python ../scripts/target_flags.py   # once: writes data/target_flags.parquet (language, tool rejections/interruptions)
python -m preference_agent.batch --output-dir work/batch-20 --prepare-only
# Paid: resumes from saved results; --limit 1 runs only the first case.
python -m preference_agent.batch --output-dir work/batch-20 --backend openrouter --env-file .env \
  --model google/gemini-3.5-flash-lite --limit 1
```

Session memory with the full history:

```sh
python -m preference_agent.batch --output-dir work/batch-20-session --history-per-session 1000 \
  --backend openrouter --env-file .env --model google/gemini-3.5-flash-lite --memory-mode session
```

**History budget.** `--history-per-session` defaults to **2**: each history session contributes two labelled turns, chosen deterministically (seeded hash of `--history-seed`) as one pushback and one no-pushback turn when both exist, then by label coverage and temporal spread. This uses history labels and changes historical label proportions. To give memory **every** labelled history turn, pass a value larger than any session, e.g. `--history-per-session 1000`. The per-field caps (`--history-context-chars` 1,500, `--history-agent-chars` 2,500, `--history-feedback-chars` 1,000) apply in both cases. The setting is part of the selection recorded in `plan.json`, so an existing output directory cannot be resumed with a different value.

**Case selection.** The runner picks its own targets: seeded random repositories, then per repository `--per-label` targets of each SWE-chat binary label. It does not read externally chosen case lists such as `work/k4-30/selection.csv`, and it scores against SWE-chat labels, not relabeled ones. Case files exported before `user_actions` was added carry an empty list; re-prepare them to use click evidence.

Defaults: holdout, k=4, 5 repositories × 2 pushback + 2 no-pushback targets, 2 history turns per session, target context/response capped at 8,000/12,000 characters, English feedback in mostly-English repositories, and no targets whose agent activity contains a user tool rejection or interruption (that activity already reveals part of the reaction). Each repository's history must end before the target session starts. `--per-case-calls` and `--max-total-calls` bound logical calls. Results go to `result-*.json`; `summary.json` and `cases.csv` report complete pairs only, with a repository bootstrap for the paired Brier difference.

LLM baselines B2/B3 on the same prepared cases, from the agent directory:

```sh
# Paid: one forecast call per case per arm; resumes from saved baseline-*.json.
python -m preference_agent.baselines --cases-dir work/batch-20 --backend openrouter --env-file .env \
  --model google/gemini-3.5-flash-lite --arms b2 b3 --top-k 5 --max-total-calls 100
python -m preference_agent.baselines --cases-dir work/batch-20 --summary-only   # no calls
```

- B2 (no-history LLM): one `dspy.Predict` call that sees only the target context and agent activity.
- B3 (BM25 + LLM): the same signature, plus the top-k BM25 episodes from the same repository before
  target_order (request, agent activity, user actions, follow-up, label; capped) placed in the prompt.
  No rules, no tools. The query is the tail of the target context and agent activity.

Both arms are binary: one signature outputs P(pushback) in [0, 1], with the pushback definition from
`prompts/relabel_pushback.md`; history labels in the prompt are pushback/no_pushback. Citations
must be prompt episode IDs. Validation failures are recorded as `failed`, not retried. `baselines-summary.json`
reports binary metrics only (AUROC, AUPRC, Brier, log loss, accuracy/F1 at a fixed 0.5) and, on cases where every
present arm succeeded, paired Brier differences with a repository bootstrap. `no_memory`/`agent` from
`result-*.json` use the same binary definition.
Cases from `relabel_cases.py` (`--relabel-dir work/relabel-100`) carry the relabeled binary target.

Offline B0/B1 baselines, from the repository root:

```sh
PYTHONPATH=preference-agent python -m preference_agent.evaluate \
  --k 1 --split holdout --bootstrap 200 --output-dir work/baseline-eval-k1
```

B0 uses the eligible training targets' pushback rate; B1 uses the same repository's first-k labeled history pushback rate, falling back to B0 if empty. Both use SWE-chat t1 labels, not relabeled targets. Outputs include coverage, binary AUROC/AUPRC/Brier/log loss, fixed-threshold F1 and repository bootstrap. This metadata-only baseline has not applied the raw adapter's strict session-overlap exclusions, so its coverage is not automatically identical to model evaluation.

## Pushback relabeling

SWE-chat's `prompt_pushback` labels are LLM-annotated from the typed prompt (Qwen3.5-9B; the paper reports 0.67 four-class and 0.79 binary accuracy against human labels). Its `correction` class also covers missing context, new requirements and answers to the agent's questions. `preference_agent.relabel` relabels a developer's reaction as:

- **pushback**: `failure_report`, `rejection` (including a rejected tool call), or `correction` of an avoidable agent error (misunderstood or ignored instruction, wrong file or environment, "no, do Y instead");
- **non-pushback**: `continue` (approval, next task, answers to the agent's questions, confirmation questions, complaints about earlier turns while accepting the current work) or `clarification` (missing context the agent could not know, new requirements, changes of mind, information-only takeovers);
- **not feedback**: orchestrator prompts, command templates, harness or hook text, pasted output, word-for-word repeats.

The rules live in `prompts/relabel_pushback.md` between the `prompt:start` and `prompt:end` markers; edit them there. Each episode shows the developer's last 10 earlier messages in the session, the agent's reply before the previous request, that request, all agent activity in order (replies, tool calls with 200-character result previews, interruptions), the developer's rejected tool calls, interruptions and typed-while-busy messages, and the next message. Nothing after that message is included.

```sh
# Exact episodes (e.g. a human-labeled CSV with session_id, turn_number[, human_label]):
python -m preference_agent.relabel --sample-csv work/relabel-pilot/labels.csv \
  --output-dir work/relabel-v2-pilot --backend openrouter --env-file .env --model <provider/model>
# Seeded random sample, one target per session, excluding already-labeled episodes;
# --k 4 keeps only targets whose repo's first 4 sessions end before the target session:
python -m preference_agent.relabel --targets ../data/english-targets/targets_english.parquet \
  --k 4 --n 60 --seed 330 --exclude-csv work/relabel-pilot/labels.csv \
  --exclude-csv work/relabel-100/labels.csv --output-dir work/relabel-k4-extra \
  --backend openrouter --env-file .env --model <provider/model>
```

`--prepare-only` builds episodes without model calls; `--report-only` rebuilds the outputs; `--limit N` labels only N episodes; `--workers` (default 4) sets parallel requests. Labeling resumes from `labels.jsonl` and retries failed episodes on the next run. The code checks every evidence quote against the reply, user actions and earlier messages and records mismatches. Outputs: `episodes.jsonl`, `labels.jsonl`, `labels.csv` (with empty `human_label`/`human_notes` columns), `verify.md` and `summary.json` (agreement with SWE-chat, and with human labels when the input CSV has them).

The k-history check (`relabel.history_valid`) mirrors the data adapter: sessions ordered by start time, the target in session k+1 or later, no missing timestamps on visible rows in the first k sessions, and the last of those timestamps before the target session starts. It reads every session's end time in one pass over two raw columns and takes about a second.

Labeled runs in `work/`. Every pipeline run used `nvidia_nim/azure/openai/gpt-6.1-sol` with an unset temperature, so reruns can differ. This public copy keeps `labels.csv`, `plan.json` and `summary.json` for each run. It removes the episodes, `labels.jsonl`, `verify.md`, and the evidence-quote, reasoning and notes columns, because they contain verbatim SWE-chat text:

| Folder | Contents |
|---|---|
| `relabel-pilot` | 30 English train targets: draft labels and human verification (draft vs human 24/27, SWE-chat vs human 15/27) |
| `relabel-v2-pilot` | The same 30 labeled by the pipeline: 24/27 agreement with the human labels. The prompt's rules were fitted on these cases, so this overstates accuracy. |
| `relabel-100` | 100 random English targets: 15 pushback, 69 non-pushback, 16 not feedback |
| `relabel-k4-extra` | 60 random k=4-valid English targets: 16 pushback, 40 non-pushback, 4 not feedback |
| `k4-30` | 15 pushback + 15 non-pushback k=4-valid targets drawn from the two previous runs, with history timing evidence in `selection.csv` |
| `k4-30/history` | The 449 labelled history turns of those 30 cases' repos (first 4 sessions, 19 repos): 123 pushback, 313 non-pushback, 13 not feedback. Made with `--history-of work/k4-30/selection.csv --k 4`. |
| `k4-100` | 100 random k=4-valid English targets, excluding every earlier run: 20 pushback, 73 non-pushback, 7 not feedback |
| `k4-100/history` | The 1089 labelled history turns of those 100 targets' repos (first 4 sessions, 55 repos): 269 pushback, 734 non-pushback, 86 not feedback; 242 turns shared with `k4-30/history` reuse its labels. |

Known issues: the labeler missed one plan-mode carry-over text (labeled correction), word-for-word repeats are treated as not feedback even when they may signal an ignored instruction, and it rarely reports low confidence. These labels are not yet human-verified except for `relabel-pilot`. Outside `k4-30/history` and `k4-100/history`, history events still carry SWE-chat labels. Target labels are for scoring only; history labels are model inputs; `stated_preference` is for analysis only and must not be given to memory. This public copy does not include it.

## Current research status

Real OpenRouter inference has been exercised. The latest development pilot used 20 repeatedly-inspected holdout targets from 5 repositories, balanced by binary label, with k=4 and budgeted history. There were 19 valid pairs; paired Brier was about 0.226 with the agent versus 0.279 without memory. Repository bootstrap uncertainty included no improvement. Agent first-pass validation passed 16/20, increasing to 19/20 after one bounded feedback attempt. These are preliminary development results, not evidence of generalization or a final benchmark.

Limitations include small selected samples, repeated holdout tuning, truncated context, lexical/multilingual retrieval, task-specific candidate rules, uncalibrated probabilities and incomplete historical reading. Test data has not been used for model evaluation. Freeze the configuration and evaluate new samples before claiming effectiveness. Reviser and DSPy optimization remain experimental/unrun.

This public repository contains no SWE-chat conversation text. The only runnable cases are the synthetic examples. For real runs, `work/` keeps the labels, plans, summaries, per-case score CSVs and reports. The prepared cases (`case-*.json`), predictions (`result-*.json`, `baseline-*.json`), memory files and logs are not included, although some reports refer to them. To rebuild the cases, download the raw data and run `relabel_cases` as shown below.

## Relabeled k4-30 evaluation

Prepare a fresh run from the committed target and history labels (requires the local raw dataset):

```sh
python -m preference_agent.relabel_cases --relabel-dir work/k4-30 --labels-file selection.csv \
  --history-labels work/k4-30/history/labels.csv --k 4 --output-dir work/k4-30-fresh
```

The preparation step makes no model calls. Use this directory with `baselines --cases-dir` and
`batch --output-dir --memory-mode session` to evaluate the same cases.
The completed Luna run is in [work/k4-30-eval-abad101/eval-report.md](work/k4-30-eval-abad101/eval-report.md).
Target and history labels are model-generated and await human verification.

## Experimental scoped memory (development candidate)

Session memory now records `kind`, `applies_when`, and `reusable`, preserving exact evidence quotes.
One-off task instructions stay in raw history rather than becoming standing preferences. Merging
keeps conditions narrow; verification must support the condition. Before forecasting, only active,
reusable scoped rules can pass a separate applicability check with an exact current-context quote.
`memory` is the full audit record, `memory_selection` explains decisions, and `used_memory` contains
only rules passed to the forecaster. Candidates and legacy unscoped caches are not injected by the
scoped session mode; regenerate memory rather than reusing old-format caches. Event/demo backends
retain their existing behavior.

When eligible active rules exist, applicability adds one model call, counted against the same budget.
Memory construction alone uses an 8,192-token output cap; forecasts and applicability retain the
configured LM cap (2,048 in the OpenRouter setup). Extraction requests at most four reusable rules
per batch with short evidence. A parse error is split-retried only when a fresh response explicitly
reports a length stop; each half is called once, without recursive retries or budget resets.

This is an experimental candidate, not a validated improvement. In the recovered k4-100 result,
no scoped rule actually reached the forecaster. Some probability metrics improved, but pushback
recall/F1 did not, and one memory build remains unavailable. The prediction prompt and historical
retrieval remain the original versions; prior prompt/retrieval experiments are not enabled.

Saved results (model labels await human verification; case and prediction files are not included):

- [List of the 93 prepared cases](work/k4-100-eval/plan.json) and [per-case baseline scores](work/k4-100-eval/baselines-cases.csv)
- [Original four-arm evaluation](work/k4-100-eval/eval-report.md)
- [Recovered scoped-memory evaluation](work/k4-100-scoped-memory-recovered/eval-report.md)
- [Five-method comparison on 91 common cases](work/k4-100-scoped-memory-recovered/baseline-comparison.json)
- [Memory-output reliability retest](work/k4-100-memory-output-retest/retest-report.md)

The recovered evaluation merges 84 existing successful predictions and 8 recovery predictions; it
is not a fresh homogeneous full run. Case-0052 remains a memory failure. `expected` and target-label
metadata are scoring/audit data, never predictor inputs. Do not send complete case JSON to an LLM.
