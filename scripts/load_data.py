"""Download (and cache to data/) only the SWE-chat columns needed for the proposal's data checks."""
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

REPO = "SALT-NLP/SWE-chat"
CACHE = Path(__file__).resolve().parent.parent / "data"
CACHE.mkdir(exist_ok=True)

SESSION_COLS = ["session_id", "repo_id", "user_id", "agent", "created_at", "session_started_at", "prompt_count"]
REPO_COLS = ["repo_id", "is_fork", "num_sessions", "num_contributors_in_dataset"]
CONV_COLS = ["session_id", "repo_id", "user_id", "turn_number", "conversation_turn_number", "turn_type",
             "is_first_turn", "is_continuation", "prompt_pushback", "prompt_intent", "timestamp"]


def read(table, columns, filters=None, name=None):
    out = CACHE / f"{name or table}.parquet"
    if out.exists():
        return pd.read_parquet(out)
    local = CACHE / "raw" / f"{table}.parquet"
    if local.exists():
        df = pq.read_table(local, columns=columns, filters=filters).to_pandas()
    else:
        with HfFileSystem().open(f"datasets/{REPO}/{table}.parquet") as f:
            df = pq.read_table(f, columns=columns, filters=filters).to_pandas()
    df.to_parquet(out)
    return df


def load():
    sessions = read("sessions", SESSION_COLS)
    repos = read("repositories", REPO_COLS)
    # conversational turns only: user prompts + assistant responses (needed to pair agent turn -> next user prompt)
    conv = read("conversations", CONV_COLS,
                filters=[("turn_type", "in", ["user_prompt", "assistant_response"])], name="conv_turns")
    return sessions, repos, conv


if __name__ == "__main__":
    s, r, c = load()
    print(s.shape, r.shape, c.shape)
