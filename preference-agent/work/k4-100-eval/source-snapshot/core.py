"""Chronological preference-memory pipeline; no external dependencies."""
from dataclasses import asdict, dataclass, field
from typing import Protocol
import json
import math
import re
import hashlib

LABELS = {'pushback', 'no_pushback'}
# Older SWE-chat style labels in existing case files are read as binary.
LEGACY_LABELS = {'correction': 'pushback', 'rejection': 'pushback', 'failure_report': 'pushback',
                 'non_pushback': 'no_pushback'}

def binary_label(label):
    """pushback or no_pushback; legacy four-class names map onto them, anything else is returned unchanged."""
    return LEGACY_LABELS.get(label, label)

@dataclass
class Event:
    id: str
    repo: str
    order: int  # Global chronological turn index within a repository.
    context: str
    agent_turn: str
    follow_up: str
    label: str
    code_fate: str = 'unknown'
    evidence_group: str = ''  # Same session/task is one conservative evidence unit.
    # Non-typed reactions inside agent_turn, e.g. rejected tool calls and interruptions.
    user_actions: list[str] = field(default_factory=list)

    def __post_init__(self):
        self.label = binary_label(self.label)

@dataclass
class Rule:
    text: str
    repo: str
    evidence: list[str] = field(default_factory=list)
    contradictions: list[str] = field(default_factory=list)
    confidence: float = 0.0
    status: str = 'candidate'
    verification: list[dict] = field(default_factory=list)
    independent_supports: int = 0
    polarity: str = ''  # Session memory: 'prefers' or 'avoids'; empty for per-event rules.

def model_event(event):
    """Event as a model sees it."""
    record = asdict(event)
    record['label'] = binary_label(event.label)
    return record

class Model(Protocol):
    def explain(self, event: Event) -> str: ...
    def verify(self, rule: Rule, event: Event) -> dict:
        """Return verdict, reason, and exact feedback_excerpt."""
        ...
    def predict(self, context: str, response: str, rules: list[Rule]) -> dict: ...
    def revise(self, context: str, response: str, prediction: dict) -> str: ...
    # Session memory mode only (memory_mode == 'session').
    def extract_session(self, events: list[Event]) -> list[dict]:
        """Return [{'text', 'polarity': 'prefers'|'avoids', 'evidence': [{'event_id', 'quote'}]}]."""
        ...
    def merge_rules(self, candidates: list[dict], max_rules: int) -> list[dict]:
        """Return [{'text', 'sources': [candidate ids]}]."""
        ...
    def verify_session(self, rules: list[dict], events: list[Event]) -> list[dict]:
        """Return [{'rule_id', 'event_id', 'verdict', 'reason', 'feedback_excerpt'}]."""
        ...

class FeedbackMiner:
    def run(self, events: list[Event]) -> list[Event]:
        if len({e.id for e in events}) != len(events):
            raise ValueError('Event IDs must be unique')
        if any(e.label not in LABELS for e in events):
            raise ValueError('Unknown pushback label')
        return sorted(events, key=lambda e: (e.order, e.id))

class Explainer:
    def __init__(self, model: Model): self.model = model
    def run(self, event: Event) -> Rule:
        return Rule(self.model.explain(event), event.repo, [event.id])

def evidence_group(event):
    if event.evidence_group:
        return event.evidence_group
    # Real SWE-chat IDs are session_id:turn_number. A session may contain
    # several tasks; treating it as one unit deliberately errs conservatively.
    session, separator, turn = event.id.rpartition(':')
    if separator and turn.isdigit():
        return session
    context = ' '.join(event.context.split()).casefold()
    return hashlib.sha256(context.encode()).hexdigest() if context else event.id



def evidence_groups(events):
    """Collapse connected same-session/task and exact duplicate-context events."""
    parent = {event.id: event.id for event in events}
    def find(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key
    seen = {}
    for event in events:
        keys = [('group', evidence_group(event))]
        context = ' '.join(event.context.split()).casefold()
        if context: keys.append(('context', context))
        for key in keys:
            if key in seen: parent[find(event.id)] = find(seen[key])
            else: seen[key] = event.id
    return {event.id: find(event.id) for event in events}


class VerificationEvidenceError(ValueError):
    """An invalid model verdict; safe to reject locally, never count as evidence."""
    def __init__(self, message, result):
        super().__init__(message)
        self.result = result


ACCEPTANCE = {'ok', 'okay', 'k', 'yes', 'yep', 'yeah', 'sure', 'approved', 'approve', 'lgtm', 'looks good',
              'looks good to me', 'sounds good', 'good', 'great', 'perfect', 'nice', 'thanks', 'thank you', 'done',
              'go ahead', 'yes please', 'continue', 'proceed', 'all', 'do it'}

def stated_preference_quote(excerpt, event):
    """A quote that states something: a user action, or a non-trivial phrase that is not bare acceptance."""
    if any(excerpt in action for action in event.user_actions):
        return True
    text = re.sub(r'[^\w\s]', ' ', excerpt).casefold().split()
    return len(excerpt.strip()) >= 12 and ' '.join(text) not in ACCEPTANCE

def validate_verification(result, event, allow_stated=False):
    """Exact quotes only. allow_stated lets an explicit statement in a no-pushback reply support a rule."""
    if not isinstance(result, dict) or result.get('verdict') not in {'support', 'contradict', 'irrelevant'}:
        raise VerificationEvidenceError('Verifier must return a structured verdict', result)
    reason, excerpt = result.get('reason'), result.get('feedback_excerpt')
    if not isinstance(reason, str) or not reason.strip() or not isinstance(excerpt, str):
        raise VerificationEvidenceError('Verifier requires reason and feedback_excerpt', result)
    sources = [event.follow_up] + (list(event.user_actions) if allow_stated else [])
    if result['verdict'] != 'irrelevant' and (not excerpt.strip() or not any(excerpt in s for s in sources)):
        raise VerificationEvidenceError('Verifier evidence must be an exact current feedback excerpt', result)
    if result['verdict'] == 'support' and event.label == 'no_pushback' and not (
            allow_stated and stated_preference_quote(excerpt, event)):
        # Silence/acceptance labels alone cannot establish a preference.
        result = {**result, 'verdict': 'irrelevant', 'reason': 'Conservative exclusion of no-pushback support: ' + reason}
    return result


class Verifier:
    def __init__(self, model: Model): self.model = model
    def run(self, rule: Rule, events: list[Event]) -> Rule:
        by_id = {e.id: e for e in events if e.repo == rule.repo}
        grouped = evidence_groups(list(by_id.values()))
        groups = {grouped[e] for e in rule.evidence if e in by_id}
        contra_groups = {grouped[e] for e in rule.contradictions if e in by_id}
        for event in events:
            if event.repo != rule.repo or event.id in rule.evidence + rule.contradictions:
                continue
            group = grouped[event.id]
            if group in groups | contra_groups:
                rule.verification.append({'event_id': event.id, 'group': group,
                    'verdict': 'skipped_dependent', 'reason': 'Same session/task or duplicate context', 'feedback_excerpt': ''})
                continue
            try:
                result = validate_verification(self.model.verify(rule, event), event)
            except VerificationEvidenceError as error:
                rejected = error.result if isinstance(error.result, dict) else {}
                rule.verification.append({'event_id': event.id, 'group': group,
                    'verdict': 'invalid_evidence', 'reason': str(error),
                    'proposed_verdict': str(rejected.get('verdict', ''))[:64],
                    'proposed_reason': str(rejected.get('reason', ''))[:1000],
                    'feedback_excerpt': str(rejected.get('feedback_excerpt', ''))[:1000]})
                continue
            rule.verification.append({'event_id': event.id, 'group': group, **result})
            if result['verdict'] == 'support':
                rule.evidence.append(event.id); groups.add(group)
            elif result['verdict'] == 'contradict':
                rule.contradictions.append(event.id); contra_groups.add(group)
        rule.independent_supports = len(groups)
        rule.confidence = (len(groups) + 1) / (len(groups) + len(contra_groups) + 2)
        rule.status = ('retired' if rule.confidence < .5 else
                       'active' if len(groups) >= 2 else 'candidate')
        return rule

class MemoryCurator:
    def run(self, rules: list[Rule]) -> list[Rule]:
        merged = {}
        for rule in rules:
            key = (rule.repo, rule.text.strip().casefold())
            if key not in merged: merged[key] = rule
            else:
                current = merged[key]
                current.evidence = sorted(set(current.evidence + rule.evidence))
                current.contradictions = sorted(set(current.contradictions + rule.contradictions))
        return list(merged.values())

class SessionMemory:
    """Session-level memory: few calls per session instead of one per event or rule-event pair.

    1. extract: one call per session chunk lists reusable preferences ('prefers' or 'avoids'), each quoting
       a pushback reply, a rejected tool call/interruption, or an explicit statement in any reply;
    2. merge: one call groups equivalent same-polarity candidates from different sessions into general rules;
    3. verify: one call per session chunk tests every rule against that chunk's turns.
    Quotes must be exact. With stated_preferences=False, extraction is pushback-only like the per-event
    pipeline. Only distinct evidence groups (sessions/duplicate contexts) count as independent support.
    """
    def __init__(self, model: Model, chunk_size=20, max_rules=10, stated_preferences=True):
        if chunk_size < 1 or max_rules < 1: raise ValueError('Invalid session memory limits')
        self.model, self.chunk_size, self.max_rules = model, chunk_size, max_rules
        self.stated_preferences = stated_preferences

    def _chunks(self, events):
        grouped = evidence_groups(events)
        sessions = {}
        for event in events:
            sessions.setdefault(grouped[event.id], []).append(event)
        for group, members in sessions.items():
            for start in range(0, len(members), self.chunk_size):
                yield group, members[start:start + self.chunk_size]

    def run(self, past: list[Event], repo: str) -> list[Rule]:
        events = [e for e in past if e.repo == repo]
        grouped = evidence_groups(events)
        by_id = {e.id: e for e in events}
        chunks = list(self._chunks(events))
        candidates = []
        for group, chunk in chunks:
            if not self.stated_preferences and all(e.label == 'no_pushback' for e in chunk): continue
            in_chunk = {e.id for e in chunk}
            for item in self.model.extract_session(chunk):
                text = item.get('text') if isinstance(item, dict) else None
                if not isinstance(text, str) or not text.strip(): continue
                records = []
                for evidence in item.get('evidence') or []:
                    event_id, quote = evidence.get('event_id'), evidence.get('quote')
                    if event_id not in in_chunk: continue
                    try:
                        result = validate_verification({'verdict': 'support', 'reason': 'Session extraction evidence',
                                                        'feedback_excerpt': quote}, by_id[event_id],
                                                       allow_stated=self.stated_preferences)
                    except VerificationEvidenceError:
                        continue
                    if result['verdict'] == 'support':
                        records.append({'event_id': event_id, 'group': grouped[event_id], 'stage': 'extract', **result})
                if records:
                    polarity = item.get('polarity') if item.get('polarity') in {'prefers', 'avoids'} else 'prefers'
                    candidates.append({'text': text.strip(), 'polarity': polarity, 'records': records})
        if not candidates:
            return []
        rules, assigned = [], set()
        merged = self.model.merge_rules([{'id': i, 'text': c['text'], 'polarity': c['polarity']}
                                         for i, c in enumerate(candidates)], self.max_rules)
        for item in merged:
            sources = [i for i in (item.get('sources') or []) if isinstance(i, int) and 0 <= i < len(candidates)
                       and i not in assigned] if isinstance(item, dict) else []
            if not sources: continue
            # Never merge opposite polarities: keep the first source's polarity; others stay separate.
            polarity = candidates[sources[0]]['polarity']
            sources = [i for i in sources if candidates[i]['polarity'] == polarity]
            assigned.update(sources)
            text = item.get('text') if isinstance(item.get('text'), str) and item['text'].strip() else candidates[sources[0]]['text']
            rules.append((text.strip(), polarity, [r for i in sources for r in candidates[i]['records']]))
        # Unmerged candidates stay separate rather than silently disappearing.
        rules += [(c['text'], c['polarity'], c['records']) for i, c in enumerate(candidates) if i not in assigned]
        memory = []
        for text, polarity, records in rules:
            rule = Rule(text, repo, sorted({r['event_id'] for r in records}), verification=list(records), polarity=polarity)
            memory.append(rule)
        sources = [{grouped[e] for e in rule.evidence} for rule in memory]
        support = [set(s) for s in sources]
        contra = [set() for _ in memory]
        for group, chunk in chunks:
            testable = [i for i, s in enumerate(sources) if group not in s]
            if not testable: continue
            in_chunk = {e.id: e for e in chunk}
            ids = {f'm{i}': i for i in testable}
            seen = set()
            rules_view = [{'rule_id': key, 'text': memory[i].text, 'polarity': memory[i].polarity} for key, i in ids.items()]
            for result in self.model.verify_session(rules_view, chunk):
                if not isinstance(result, dict) or result.get('rule_id') not in ids: continue
                index, event_id = ids[result['rule_id']], result.get('event_id')
                rule = memory[index]
                if event_id not in in_chunk:
                    rule.verification.append({'event_id': None, 'proposed_event_id': str(event_id)[:128], 'group': group,
                                              'stage': 'verify', 'verdict': 'invalid_reference',
                                              'reason': 'Verdict names an event outside this session chunk'})
                    continue
                if (index, event_id) in seen: continue
                seen.add((index, event_id))
                try:
                    checked = validate_verification({key: result.get(key) for key in ['verdict', 'reason', 'feedback_excerpt']},
                                                    in_chunk[event_id], allow_stated=self.stated_preferences)
                except VerificationEvidenceError as error:
                    rejected = error.result if isinstance(error.result, dict) else {}
                    rule.verification.append({'event_id': event_id, 'group': group, 'stage': 'verify',
                        'verdict': 'invalid_evidence', 'reason': str(error),
                        'proposed_verdict': str(rejected.get('verdict', ''))[:64],
                        'proposed_reason': str(rejected.get('reason', ''))[:1000],
                        'feedback_excerpt': str(rejected.get('feedback_excerpt', ''))[:1000]})
                    continue
                rule.verification.append({'event_id': event_id, 'group': group, 'stage': 'verify', **checked})
                if checked['verdict'] == 'support':
                    if event_id not in rule.evidence: rule.evidence.append(event_id)
                    support[index].add(group)
                elif checked['verdict'] == 'contradict':
                    if event_id not in rule.contradictions: rule.contradictions.append(event_id)
                    contra[index].add(group)
        for rule, groups, contra_groups in zip(memory, support, contra):
            rule.independent_supports = len(groups)
            rule.confidence = (len(groups) + 1) / (len(groups) + len(contra_groups) + 2)
            rule.status = ('retired' if rule.confidence < .5 else
                           'active' if len(groups) >= 2 else 'candidate')
        return memory

class Retriever:
    def run(self, context: str, response: str, rules: list[Rule], limit=5) -> list[Rule]:
        tokens = lambda text: set(re.findall(r'\w+', text.casefold()))
        query = tokens(context + ' ' + response)
        ranked = sorted((r for r in rules if r.status == 'active'),
                        key=lambda r: len(query & tokens(r.text)) * r.confidence, reverse=True)
        return [r for r in ranked if query & tokens(r.text)][:limit]

class Critic:
    def __init__(self, model: Model): self.model = model
    def run(self, context, response, rules):
        return self.validate(self.model.predict(context, response, rules))

    @staticmethod
    def validate(result):
        """Every backend forecasts one binary pushback_probability in [0, 1]."""
        p = result['pushback_probability']
        if type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1:
            raise ValueError('Expected a pushback probability in [0, 1]')
        result['pushback_probability'] = float(p)
        result['calibrated'] = False
        return result

class Reviser:
    def __init__(self, model: Model): self.model = model
    def run(self, context, response, prediction):
        return self.model.revise(context, response, prediction)

class Pipeline:
    def __init__(self, model: Model): self.model = model
    def build_memory(self, history: list[Event], repo: str, target_order: int) -> list[Rule]:
        past = FeedbackMiner().run([e for e in history if e.repo == repo and e.order < target_order])
        if getattr(self.model, 'memory_mode', 'event') == 'session':
            return SessionMemory(self.model, getattr(self.model, 'session_chunk_size', 20),
                                 getattr(self.model, 'max_session_rules', 10),
                                 getattr(self.model, 'stated_preferences', True)).run(past, repo)
        rules = [Explainer(self.model).run(e) for e in past if e.label != 'no_pushback']
        rules = MemoryCurator().run(rules)
        return [Verifier(self.model).run(r, past) for r in rules]

    def run(self, history: list[Event], repo: str, target_order: int,
            context: str, response: str, revision_threshold=.5, revise=False) -> dict:
        rules = self.build_memory(history, repo, target_order)
        return self.predict_with_memory(history, rules, repo, target_order, context, response,
                                        revision_threshold, revise)

    def predict_with_memory(self, history: list[Event], rules: list[Rule], repo: str,
                            target_order: int, context: str, response: str,
                            revision_threshold=.5, revise=False) -> dict:
        from copy import deepcopy
        past = FeedbackMiner().run([e for e in history if e.repo == repo and e.order < target_order])
        allowed = {e.id for e in past}
        if any(r.repo != repo or not set(r.evidence + r.contradictions) <= allowed
               or any(v.get('event_id') not in allowed for v in r.verification
                      if v.get('verdict') != 'invalid_reference') for r in rules):
            raise ValueError('Cached memory contains out-of-scope evidence')
        rules = deepcopy(rules)
        retrieved = Retriever().run(context, response, rules)
        if getattr(self.model, 'critic_mode', 'fixed') == 'react':
            prediction = self.model.predict_with_history(context, response, rules, past, repo, target_order)
        else:
            prediction = Critic(self.model).run(context, response, retrieved)
        revised = (Reviser(self.model).run(context, response, prediction)
                   if revise and prediction['pushback_probability'] >= revision_threshold else response)
        return {'memory': [asdict(r) for r in rules],
                'retrieved_rules': [asdict(r) for r in retrieved],
                'prediction': prediction, 'revised_response': revised,
                'history_ids': [e.id for e in past]}

def save_result(result: dict, path: str):
    with open(path, 'w', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
