"""Fixed-k history selection, independent of raw parquet schema."""
from dataclasses import dataclass, replace
from datetime import datetime
from .core import Event

@dataclass
class Session:
    id: str
    repo: str
    started_at: str
    ended_at: str
    events: list[Event]

def timestamp(value):
    time = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if time.tzinfo is None: raise ValueError('Session times must include timezone')
    return time

def select_history(sessions: list[Session], repo: str, target_session: str, k: int):
    if k not in {1, 2, 4, 8}: raise ValueError('k must be 1, 2, 4 or 8')
    scoped = [s for s in sessions if s.repo == repo]
    if len({s.id for s in scoped}) != len(scoped): raise ValueError('Duplicate session ID')
    ordered = sorted(scoped, key=lambda s: (timestamp(s.started_at), s.id))
    if len(ordered) < k + 1: raise ValueError('Need at least k+1 sessions')
    target = next((s for s in ordered if s.id == target_session), None)
    if target is None or target in ordered[:k]: raise ValueError('Target must be after history sessions')
    history = ordered[:k]
    for session in history:
        if timestamp(session.ended_at) < timestamp(session.started_at): raise ValueError('Invalid session interval')
        if timestamp(session.ended_at) >= timestamp(target.started_at):
            raise ValueError('History overlaps target; feedback may not yet be observable')
        if any(e.repo != repo for e in session.events): raise ValueError('Event repository mismatch')
    return [replace(event, evidence_group=session.id) for session in history for event in session.events]
