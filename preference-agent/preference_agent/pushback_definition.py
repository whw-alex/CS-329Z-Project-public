"""The pushback standard shared by labeling and forecasting.

Forecast prompts (agent, no_memory, B2, B3) embed the "What counts as pushback" definition from
prompts/relabel_pushback.md, the definition used to relabel targets, so models predict the label under
the standard it was made with. The labeler's steps (not-feedback check, fine-label rules, recorded fields)
are not included.
"""
from functools import lru_cache
from pathlib import Path

PROMPT = Path(__file__).resolve().parents[1] / 'prompts' / 'relabel_pushback.md'
START, END = '## What counts as pushback', '### Step 1'

@lru_cache(maxsize=1)
def pushback_rules(path=PROMPT):
    text = Path(path).read_text(encoding='utf-8')
    if START not in text or END not in text:
        raise ValueError(f'{path} lacks the expected pushback section')
    return text[text.index(START):text.index(END)].strip()

def forecast_instructions(task):
    """task: what to forecast, e.g. the hidden-next-message framing; followed by the shared definition."""
    return (task.strip() + '\n\nThe label you forecast was assigned with the definition below, applied to the hidden '
            'next message. It is a label definition, not an observation about this case.\n\n' + pushback_rules())
