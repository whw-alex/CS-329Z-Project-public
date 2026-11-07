"""Write data/target_flags.parquet: per-target language and in-window user actions.

For each target (a labelled human prompt), the agent-activity window is everything after the previous human
prompt. A window is flagged when the developer rejected a tool call or interrupted the agent inside it: that is
part of the human reaction leaking into the model input. Automatic denials (permission classifiers, hooks) are
not user actions and are not flagged. Language comes from SWE-chat's `language` column on the feedback prompt;
repo_language is the most common target language in the repository.

Requires data/raw/conversations.parquet and data/targets.parquet.
"""
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds

from targets import is_non_human

ROOT = Path(__file__).resolve().parent.parent
REJECTED = ("The user doesn't want to proceed with this tool use", "The user doesn't want to take this action")
INTERRUPTED = "[Request interrupted"

scan = ds.dataset(ROOT / "data" / "raw" / "conversations.parquet").scanner(
    columns=["session_id", "turn_number", "turn_type", "language", "content"],
    filter=pc.field("turn_type").isin(["tool_result", "user_prompt"]), batch_size=200_000)
rejections, prompts = [], []
for b in scan.to_batches():
    head = pc.utf8_slice_codeunits(pc.fill_null(b["content"], ""), 0, 300)
    is_tool = pc.equal(b["turn_type"], "tool_result")
    rejected = pc.or_(*[pc.starts_with(head, p) for p in REJECTED])
    rejections.append(pa.table({"session_id": b["session_id"], "turn_number": b["turn_number"]})
                      .filter(pc.and_(is_tool, rejected)).to_pandas())
    prompts.append(pa.table({"session_id": b["session_id"], "turn_number": b["turn_number"], "head": head,
                             "language": b["language"]}).filter(pc.invert(is_tool)).to_pandas())
prompts = pd.concat(prompts)
prompts["human"] = ~is_non_human(prompts["head"])
events = pd.concat([pd.concat(rejections).assign(kind="tool_rejected"),
                    prompts[prompts["head"].str.lstrip().str.startswith(INTERRUPTED)]
                    [["session_id", "turn_number"]].assign(kind="interrupted")])

targets = pd.read_parquet(ROOT / "data" / "targets.parquet")
human = prompts[prompts.human].sort_values(["session_id", "turn_number"])
human["prev_turn"] = human.groupby("session_id").turn_number.shift()
out = targets[["session_id", "turn_number", "repo_id", "split"]].merge(
    human[["session_id", "turn_number", "prev_turn", "language"]], on=["session_id", "turn_number"], how="left")
if out.prev_turn.isna().any():
    raise ValueError("Some targets have no previous human prompt; targets.parquet and raw data disagree")
window = out[["session_id", "turn_number", "prev_turn"]].merge(events, on="session_id", suffixes=("", "_event"))
window = window[(window.turn_number_event > window.prev_turn) & (window.turn_number_event < window.turn_number)]
for kind in ["tool_rejected", "interrupted"]:
    hit = window[window.kind == kind][["session_id", "turn_number"]].drop_duplicates().assign(**{kind: True})
    out = out.merge(hit, on=["session_id", "turn_number"], how="left")
    out[kind] = out[kind].fillna(False).astype(bool)
out["user_action_in_window"] = out.tool_rejected | out.interrupted
out["repo_language"] = out.repo_id.map(out.groupby("repo_id").language.agg(
    lambda s: s.mode().iat[0] if s.notna().any() else None))
out.drop(columns="prev_turn").to_parquet(ROOT / "data" / "target_flags.parquet", index=False)
print("targets", len(out), "| user action in window", int(out.user_action_in_window.sum()),
      "| english targets", int(out.language.eq("ENGLISH").sum()),
      "| english-repo targets", int(out.repo_language.eq("ENGLISH").sum()))
