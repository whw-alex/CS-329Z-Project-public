"""Deterministic history-only pilot budgets; never inspect target feedback/labels."""
from collections import Counter, defaultdict
from dataclasses import replace
import hashlib
from .core import FeedbackMiner, evidence_group

OMITTED = '[SMOKE TEST: MIDDLE OMITTED]'

def cap_text(text, limit):
    if limit is None or len(text) <= limit:
        return text
    if limit < len(OMITTED) + 4:
        raise ValueError('Character cap is too small for an omission marker')
    remaining = limit - len(OMITTED) - 2
    left = remaining // 2
    return text[:left] + '\n' + OMITTED + '\n' + text[-(remaining-left):]


def sample_history(history, *, per_session=4, seed=329, context_chars=None,
                   agent_chars=None, feedback_chars=None):
    """Spread over time and retain both binary label groups where capacity permits.

    This changes historical label proportions; report the manifest with pilot results.
    Does not create independent evidence: source group IDs remain unchanged.
    """
    if per_session < 1:
        raise ValueError('per_session must be positive')
    for limit in [context_chars, agent_chars, feedback_chars]:
        if limit is not None and limit < len(OMITTED) + 4:
            raise ValueError('Character cap is too small for an omission marker')
    events = FeedbackMiner().run(list(history))
    sessions = defaultdict(list)
    for event in events:
        sessions[(event.repo, evidence_group(event))].append(event)
    chosen = []
    manifest = {'policy': 'historical-label-stratified temporal coverage', 'seed': seed,
                'per_session': per_session, 'source_events': len(events), 'sessions': [],
                'caps': {'context': context_chars, 'agent_turn': agent_chars, 'follow_up': feedback_chars}}
    for (repo, group), members in sessions.items():
        picked = []
        priority = int(hashlib.sha256(f'{seed}:{repo}:{group}'.encode()).hexdigest(), 16)
        by_label = defaultdict(list)
        for i, event in enumerate(members):
            by_label[event.label].append(i)
        positive = [i for i, e in enumerate(members) if e.label != 'no_pushback']
        negative = by_label.get('no_pushback', [])
        # Spend first two slots on positive and negative observations if both exist.
        strata = [positive, negative] if positive and negative and per_session >= 2 else []
        for indices in strata:
            # Deterministic seeded pick across each stratum, instead of always its prefix.
            picked.append(indices[priority % len(indices)])
        while len(picked) < min(per_session, len(members)):
            available = [i for i in range(len(members)) if i not in picked]
            seen = {members[i].label for i in picked}
            unseen = [i for i in available if members[i].label not in seen]
            pool = unseen or available
            # Favor labels not yet covered, then temporal separation from prior choices.
            index = max(pool, key=lambda i: (min((abs(i-j) for j in picked), default=len(members)),
                                            -((i-priority) % len(members))))
            picked.append(index)
        selected = [members[i] for i in sorted(picked)]
        for event in selected:
            chosen.append(replace(event, context=cap_text(event.context, context_chars),
                                  agent_turn=cap_text(event.agent_turn, agent_chars),
                                  follow_up=cap_text(event.follow_up, feedback_chars)))
        manifest['sessions'].append({'repo': repo, 'group': group, 'source_events': len(members),
                                    'source_labels': dict(Counter(e.label for e in members)),
                                    'selected_ids': [e.id for e in selected],
                                    'selected_labels': dict(Counter(e.label for e in selected))})
    chosen = sorted(chosen, key=lambda e: (e.order, e.id))
    manifest['selected_events'] = len(chosen)
    manifest['truncated_fields'] = sum(OMITTED in value for e in chosen
                                     for value in [e.context, e.agent_turn, e.follow_up])
    return chosen, manifest
