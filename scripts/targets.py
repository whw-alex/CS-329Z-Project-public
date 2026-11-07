"""Turn-level target construction for pushback forecasting.

A target is one labelled *human* user prompt that follows the previous human prompt of the same session and
has *substantive agent activity* in between (>= 1 tool call, or >= 1 non-empty, non-error text reply).
The label (prompt_pushback) of that prompt is the user's reaction to the agent activity before it.
"""
import re

import numpy as np
import pandas as pd

# Prompts the harness or another tool injected (not typed by the developer as feedback).
NON_HUMAN_PREFIXES = (
    "[Request interrupted", "<environment_context", "<command-", "<local-command", "<system-reminder",
    "<task-notification", "<subagent_notification", "<turn_aborted", "<codex_internal_context",
    "<recommended_plugins", "# AGENTS.md instructions", ">>> ", "The following is the Codex agent history",
    "The Codex agent has re", "Reviewed Codex session", "Assess the exact plann", "Planned action JSON",
    "Another Claude session", "[delegate worktree]", "You are a code reviewer", "## Output", "## Tools",
    "## Before You Flag", '{"', "{\n",
)
SLASH_COMMAND = re.compile(r"^/[A-Za-z][\w:.-]*(\s|$)")

# Agent "replies" that are service failures or placeholders, not real work.
API_ERROR = re.compile(
    r"^(API Error|Prompt is too long|You've hit your|You have hit your|Credit balance is too low|"
    r"Invalid API key|Please run /login|Claude AI usage limit|Request timed out|Unable to connect)", re.I)
EMPTY_REPLIES = {"", ".", "No response requested."}


def is_non_human(head: pd.Series) -> pd.Series:
    h = head.fillna("").str.lstrip()
    starts = np.zeros(len(h), dtype=bool)
    for p in NON_HUMAN_PREFIXES:
        starts |= h.str.startswith(p).to_numpy()
    return pd.Series(starts | h.str.match(SLASH_COMMAND).to_numpy(), index=head.index)


def build_targets(turns: pd.DataFrame):
    """turns: data/turns_compact.parquet. Returns (targets, funnel)."""
    t = turns.sort_values(["session_id", "turn_number"]).reset_index(drop=True)
    is_up = t.turn_type == "user_prompt"
    t["human"] = is_up & ~is_non_human(t["head"])
    # seg = how many human prompts have occurred so far in the session
    t["seg"] = t.groupby("session_id").human.cumsum()

    h = t["head"].fillna("").str.strip()
    is_text = t.turn_type == "assistant_response"
    bad_text = is_text & (h.isin(EMPTY_REPLIES) | h.str.match(API_ERROR))
    t["tool"] = t.turn_type == "tool_use"
    t["text_ok"] = is_text & ~bad_text
    t["text_bad"] = bad_text
    act = t.groupby(["session_id", "seg"]).agg(n_tool=("tool", "sum"), n_text=("text_ok", "sum"),
                                                n_bad_text=("text_bad", "sum"))

    p = t[is_up].copy()
    p["labelled"] = p.prompt_pushback.notna()
    funnel = [("all user prompts", len(p)), ("with a pushback label", int(p.labelled.sum()))]
    p = p[p.labelled]
    p = p[p.human]
    funnel.append(("... after removing non-human prompts", len(p)))
    # the human prompt that opened the previous segment is seg-1; the first human prompt has none
    p["prev_seg"] = p.seg - 1
    p = p[p.prev_seg >= 1]
    funnel.append(("... after removing each session's first human prompt", len(p)))
    p = p.join(act, on=["session_id", "prev_seg"]).fillna({"n_tool": 0, "n_text": 0, "n_bad_text": 0})
    p["substantive"] = (p.n_tool + p.n_text) > 0
    funnel.append(("... after requiring substantive agent activity", int(p.substantive.sum())))
    p = p[p.substantive].copy()
    p["t1"] = (p.prompt_pushback != "non_pushback").astype(int)
    p["agent_has_text"] = p.n_text > 0
    cols = ["session_id", "turn_number", "head", "prompt_pushback", "t1", "n_tool", "n_text", "n_bad_text",
            "agent_has_text"]
    return p[cols].reset_index(drop=True), pd.DataFrame(funnel, columns=["step", "prompts"])
