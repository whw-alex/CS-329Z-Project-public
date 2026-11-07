"""Build data/turns_compact.parquet from the raw conversations table.

Keeps user_prompt / assistant_response / tool_use rows. Text is truncated to its first 300 characters
(enough to classify non-human prompts and API-error replies); tool_use rows carry only the tool name.
"""
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds

ROOT = Path(__file__).resolve().parent.parent
COLS = ["session_id", "turn_number", "turn_type", "is_first_turn", "prompt_pushback", "tool_name", "content"]

scanner = ds.dataset(ROOT / "data" / "raw" / "conversations.parquet").scanner(
    columns=COLS, filter=pc.field("turn_type").isin(["user_prompt", "assistant_response", "tool_use"]),
    batch_size=100_000)
out = []
for i, b in enumerate(scanner.to_batches()):
    is_tool = pc.equal(b["turn_type"], "tool_use")
    head = pc.if_else(is_tool, pa.scalar(None, pa.string()), pc.utf8_slice_codeunits(b["content"], 0, 300))
    out.append(pa.table({**{c: b[c] for c in COLS if c != "content"}, "head": head,
                         "content_len": pc.utf8_length(b["content"])}))
    if i % 10 == 0:
        print("batch", i, flush=True)
t = pa.concat_tables(out)
import pyarrow.parquet as pq
pq.write_table(t, ROOT / "data" / "turns_compact.parquet")
print("rows", t.num_rows)
