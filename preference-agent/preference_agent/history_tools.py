"""Read-only tools over a snapshot of already-authorized history only."""
from copy import deepcopy
from dataclasses import asdict
import json
import re
from .core import binary_label, model_event

class HistoryTools:
    def __init__(self, history, rules, repo, target_order, context, response,
                 max_calls=6, output_chars=12000):
        if max_calls < 1 or output_chars < 256: raise ValueError('Invalid tool budget')
        self.events = {e.id: deepcopy(e) for e in history if e.repo == repo and e.order < target_order}
        self.rules = {f'r{i}': deepcopy(r) for i, r in enumerate(rules)
                      if r.repo == repo and r.status == 'active'
                      and set(r.evidence + r.contradictions) <= self.events.keys()}
        self.context, self.response = context, response
        self.max_calls, self.output_chars = max_calls, output_chars
        self.calls, self.trace, self.exposed_ids = 0, [], set()
        self.read_ids, self.access = set(), {}

    @staticmethod
    def _score(query, text):
        words = lambda x: set(re.findall(r'\w+', x.casefold()))
        return len(words(query) & words(text))

    def _begin(self, name, args):
        if self.calls >= self.max_calls:
            return {'error': 'Tool-call budget exhausted; finish using existing observations.'}
        self.calls += 1
        self.trace.append({'tool': name, 'arguments': args, 'status': 'started'})
        return None

    def _result(self, payload, ids=()):
        text = json.dumps(payload, ensure_ascii=False)
        self.exposed_ids.update(ids)
        truncated = len(text) > self.output_chars
        self.trace[-1].update(status='ok', evidence_ids=list(ids), truncated=truncated)
        return {'content': text[:self.output_chars], 'truncated': truncated,
                'evidence_ids': list(ids), 'remaining_calls': self.max_calls - self.calls}

    def search_preferences(self, query: str) -> dict:
        """Find relevant active preference rules. Returns rule IDs and source event IDs."""
        if error := self._begin('search_preferences', {'query': query[:512]}): return error
        ranked = sorted(self.rules.items(), key=lambda item: self._score(query, item[1].text), reverse=True)
        found = [{'rule_id': key, **asdict(rule)} for key, rule in ranked
                 if self._score(query, rule.text) > 0][:5]
        # IDs in rule summaries are visible citations, but full episodes require get_episode.
        return self._result(found, [e for r in found for e in r['evidence'] + r['contradictions']])

    def search_history(self, query: str) -> dict:
        """Search authorized historical episodes; returns IDs and short feedback excerpts only; read get_episode before citing historical behavior."""
        if error := self._begin('search_history', {'query': query[:512]}): return error
        ranked = sorted(self.events.values(), key=lambda e: self._score(query, e.context + e.agent_turn + e.follow_up), reverse=True)
        found = [{'event_id': e.id, 'label': binary_label(e.label), 'follow_up_excerpt': e.follow_up[:500]}
                 for e in ranked if self._score(query, e.context + e.agent_turn + e.follow_up) > 0][:5]
        return self._result(found, [e['event_id'] for e in found])

    def get_episode(self, event_id: str) -> dict:
        """Read one historical episode by ID. Unknown, other-repository and future IDs are inaccessible."""
        if error := self._begin('get_episode', {'event_id': event_id}): return error
        if event_id not in self.events:
            self.trace[-1]['status'] = 'denied'
            return {'error': 'Episode is outside authorized history or does not exist.'}
        result = self._result(model_event(self.events[event_id]), [event_id])
        if not result['truncated']:
            self.read_ids.add(event_id)
            partial = any('[SMOKE TEST: MIDDLE OMITTED]' in value for value in
                          (self.events[event_id].context, self.events[event_id].agent_turn, self.events[event_id].follow_up))
            self.access[event_id] = 'partial_episode' if partial else 'episode'
        else:
            self.access[event_id] = 'truncated_tool_result'
        return result

    def find_counterexamples(self, rule_id: str) -> dict:
        """Return recorded historical contradictions for a rule; not an exhaustive semantic search."""
        if error := self._begin('find_counterexamples', {'rule_id': rule_id}): return error
        if rule_id not in self.rules:
            self.trace[-1]['status'] = 'denied'
            return {'error': 'Unknown authorized rule ID.'}
        ids = self.rules[rule_id].contradictions
        return self._result([{'event_id': e, 'follow_up_excerpt': self.events[e].follow_up[:500]} for e in ids], ids)

    def inspect_current_activity(self) -> dict:
        """Inspect the supplied current agent activity. No current user feedback or outcome is available."""
        if error := self._begin('inspect_current_activity', {}): return error
        return self._result({'agent_turn': self.response})

    def functions(self):
        return [self.search_preferences, self.search_history, self.get_episode,
                self.find_counterexamples, self.inspect_current_activity]
