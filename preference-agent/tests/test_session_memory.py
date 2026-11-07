import unittest
from preference_agent.core import Event, Pipeline, SessionMemory

def history():
    return [Event('A:1', 'r', 1, 'fix parser', 'done', 'please run tests first', 'correction'),
            Event('A:2', 'r', 2, 'fix parser', 'tests pass', 'ok', 'no_pushback'),
            Event('B:1', 'r', 3, 'fix serializer', 'done', 'run the tests before saying done', 'correction'),
            Event('B:2', 'r', 4, 'fix serializer', 'done', 'thanks, looks good', 'no_pushback'),
            Event('C:1', 'r', 5, 'rename', 'renamed', 'stop editing unrelated files', 'correction')]

class FakeSessionModel:
    memory_mode, critic_mode = 'session', 'fixed'
    def __init__(self): self.calls, self.verify_rules = [], {}
    def extract_session(self, events):
        self.calls.append('extract'); ids = {e.id for e in events}
        if 'A:1' in ids:
            return [{'text': 'Run tests before claiming done',
                     'evidence': [{'event_id': 'A:1', 'quote': 'run tests first'}, {'event_id': 'A:2', 'quote': 'ok'}]},
                    {'text': 'Invented quote', 'evidence': [{'event_id': 'A:1', 'quote': 'not in the feedback'}]},
                    {'text': 'Foreign event', 'evidence': [{'event_id': 'B:1', 'quote': 'run the tests'}]}]
        if 'B:1' in ids:
            return [{'text': 'Run tests before reporting', 'evidence': [{'event_id': 'B:1', 'quote': 'run the tests before saying done'}]}]
        return [{'text': 'Only edit related files', 'evidence': [{'event_id': 'C:1', 'quote': 'stop editing unrelated files'}]}]
    def merge_rules(self, candidates, max_rules):
        self.calls.append('merge')
        tests = [c['id'] for c in candidates if 'tests' in c['text']]
        return [{'text': 'Run tests before reporting completion', 'sources': tests + [99]}]
    def verify_session(self, rules, events):
        self.calls.append('verify'); ids = {e.id for e in events}
        texts = {r['rule_id']: r['text'] for r in rules}
        self.verify_rules[min(ids)] = sorted(texts.values())
        edit = next((k for k, t in texts.items() if 'edit' in t), None)
        tests = next((k for k, t in texts.items() if 'tests' in t), None)
        if 'A:1' in ids:
            return [{'rule_id': edit, 'event_id': 'A:1', 'verdict': 'support', 'reason': 'x', 'feedback_excerpt': 'invented'},
                    {'rule_id': edit, 'event_id': 'Z:9', 'verdict': 'support', 'reason': 'x', 'feedback_excerpt': 'ok'},
                    {'rule_id': 'unknown', 'event_id': 'A:1', 'verdict': 'support', 'reason': 'x', 'feedback_excerpt': 'run tests'}]
        if 'B:1' in ids:
            return [{'rule_id': edit, 'event_id': 'B:1', 'verdict': 'support', 'reason': 'x', 'feedback_excerpt': 'run the tests'},
                    {'rule_id': edit, 'event_id': 'B:2', 'verdict': 'support', 'reason': 'x', 'feedback_excerpt': 'looks good'}]
        return [{'rule_id': tests, 'event_id': 'C:1', 'verdict': 'contradict', 'reason': 'x', 'feedback_excerpt': 'stop editing'}]
    def predict(self, context, response, rules):
        return {'pushback_probability': .6,
                'rationale': 'fixed', 'evidence_ids': []}

class SessionMemoryTests(unittest.TestCase):
    def test_merge_verification_and_evidence_rules(self):
        model = FakeSessionModel()
        rules = {r.text: r for r in SessionMemory(model).run(history(), 'r')}
        self.assertEqual(model.calls, ['extract'] * 3 + ['merge'] + ['verify'] * 3)
        tests, edit = rules['Run tests before reporting completion'], rules['Only edit related files']
        self.assertEqual(sorted(rules), sorted([tests.text, edit.text]))
        # Cross-session merge is independent support; a contradiction from a third session lowers confidence.
        self.assertEqual((tests.evidence, tests.contradictions, tests.independent_supports), (['A:1', 'B:1'], ['C:1'], 2))
        self.assertEqual((tests.status, round(tests.confidence, 2)), ('active', .6))
        # Rules are never tested against their own source session.
        self.assertEqual(model.verify_rules['C:1'], [tests.text])
        self.assertEqual(model.verify_rules['A:1'], [edit.text])
        self.assertEqual((edit.evidence, edit.status), (['C:1', 'B:1'], 'active'))
        verdicts = [v['verdict'] for v in edit.verification]
        self.assertIn('invalid_evidence', verdicts)
        self.assertIn('invalid_reference', verdicts)
        # No-pushback turns cannot support a preference.
        self.assertNotIn('B:2', edit.evidence)
        self.assertNotIn('A:2', tests.evidence)

    def test_pipeline_dispatch_and_scope_check_accept_session_memory(self):
        pipeline = Pipeline(FakeSessionModel())
        result = pipeline.run(history(), 'r', 6, 'task', 'response')
        self.assertEqual(len(result['memory']), 2)
        self.assertEqual(result['history_ids'], [e.id for e in history()])

    def test_future_events_and_all_no_pushback_sessions_are_not_read(self):
        model = FakeSessionModel()
        events = history()[:2] + [Event('D:1', 'r', 9, 'x', 'y', 'fine', 'no_pushback')]
        Pipeline(model).build_memory(events, 'r', 3)
        self.assertEqual(model.calls, ['extract', 'merge'])

    def test_no_candidates_means_no_merge_call(self):
        class Empty(FakeSessionModel):
            def extract_session(self, events): self.calls.append('extract'); return []
        model = Empty()
        self.assertEqual(SessionMemory(model).run(history(), 'r'), [])
        self.assertEqual(model.calls, ['extract'] * 3)

class StatedPreferenceTests(unittest.TestCase):
    class Model(FakeSessionModel):
        def __init__(self, items, merged=None):
            super().__init__(); self.items, self.merged = items, merged
        def extract_session(self, events):
            self.calls.append('extract'); ids = {e.id for e in events}
            return [i for i in self.items if i['evidence'][0]['event_id'] in ids]
        def merge_rules(self, candidates, max_rules):
            self.calls.append('merge')
            return self.merged if self.merged is not None else [{'text': c['text'], 'sources': [c['id']]} for c in candidates]
        def verify_session(self, rules, events): self.calls.append('verify'); return []

    def test_explicit_statement_in_no_pushback_reply_counts_but_bare_acceptance_does_not(self):
        events = [Event('E:1', 'r', 1, 'write script', 'Which language?', '1. Yes 2. Python for now as it requires no setup', 'no_pushback'),
                  Event('F:1', 'r', 2, 'other task', 'done', 'looks good', 'no_pushback')]
        items = [{'text': 'Prefer Python scripts that need no setup', 'polarity': 'prefers',
                  'evidence': [{'event_id': 'E:1', 'quote': 'Python for now as it requires no setup'}]},
                 {'text': 'Likes the result', 'evidence': [{'event_id': 'F:1', 'quote': 'looks good'}]}]
        model = self.Model(items)
        rules = SessionMemory(model).run(events, 'r')
        self.assertEqual([(r.text, r.polarity, r.evidence) for r in rules],
                         [('Prefer Python scripts that need no setup', 'prefers', ['E:1'])])
        # Pushback-only mode skips all-no-pushback sessions entirely.
        model = self.Model(items)
        self.assertEqual(SessionMemory(model, stated_preferences=False).run(events, 'r'), [])
        self.assertEqual(model.calls, [])

    def test_rejected_tool_call_is_quotable_evidence(self):
        event = Event('G:1', 'r', 1, 'deploy', '{"tool_use": "git push --force"}', 'what now?', 'no_pushback',
                      user_actions=['Rejected tool call Bash: git push --force origin main'])
        items = [{'text': 'Never force-push', 'polarity': 'avoids',
                  'evidence': [{'event_id': 'G:1', 'quote': 'Rejected tool call Bash: git push --force'}]}]
        rules = SessionMemory(self.Model(items)).run([event], 'r')
        self.assertEqual([(r.polarity, r.evidence) for r in rules], [('avoids', ['G:1'])])
        # The per-event pipeline keeps its stricter rule: follow_up quotes only, no no-pushback support.
        from preference_agent.core import VerificationEvidenceError, validate_verification
        with self.assertRaises(VerificationEvidenceError):
            validate_verification({'verdict': 'support', 'reason': 'x', 'feedback_excerpt': 'git push --force'}, event)

    def test_opposite_polarities_are_never_merged(self):
        events = [Event('A:1', 'r', 1, 'one', 'x', 'always add unit tests to new modules', 'correction'),
                  Event('B:1', 'r', 2, 'two', 'x', 'do not add unit tests for prototypes', 'correction')]
        items = [{'text': 'Add unit tests', 'polarity': 'prefers', 'evidence': [{'event_id': 'A:1', 'quote': 'always add unit tests'}]},
                 {'text': 'Skip tests for prototypes', 'polarity': 'avoids', 'evidence': [{'event_id': 'B:1', 'quote': 'do not add unit tests'}]}]
        rules = SessionMemory(self.Model(items, merged=[{'text': 'Tests', 'sources': [0, 1]}])).run(events, 'r')
        self.assertEqual(sorted((r.text, r.polarity) for r in rules), [('Skip tests for prototypes', 'avoids'), ('Tests', 'prefers')])

class SessionDSPyTests(unittest.TestCase):
    def test_real_dspy_adapter_with_offline_lm(self):
        import dspy
        from dspy.utils import DummyLM
        from preference_agent.dspy_backend import DSPyAgent
        adapter = dspy.JSONAdapter()
        answers = [
            {'preferences': [{'text': 'Run tests', 'polarity': 'prefers', 'kind': 'stated_preference', 'applies_when': 'when changing code', 'reusable': True, 'evidence': [{'event_id': 'A:1', 'quote': 'run tests'}]}]},
            {'preferences': [{'text': 'Run tests first', 'polarity': 'prefers', 'kind': 'stated_preference', 'applies_when': 'when changing code', 'reusable': True, 'evidence': [{'event_id': 'B:1', 'quote': 'run tests'}]}]},
            {'rules': [{'text': 'Run tests before claiming completion', 'polarity': 'prefers', 'sources': [0, 1]}]},
            {'decisions': [{'rule_id': 'm0', 'applies': True, 'context_quote': 'fix tests', 'reason': 'Current task changes code'}]},
            {'pushback_probability': .9,
             'rationale': 'Earlier test requests', 'evidence_ids': ['A:1']},
        ]
        agent = DSPyAgent(critic_mode='fixed', memory_mode='session')
        self.assertEqual(len(agent.predictors()), 8)
        # Distinct contexts: identical contexts would be one conservative evidence group.
        events = [Event('A:1', 'r', 1, 'fix parser', 'done', 'please run tests', 'correction'),
                  Event('B:1', 'r', 2, 'fix serializer', 'done', 'run tests now', 'correction')]
        with dspy.context(lm=DummyLM(answers, adapter=adapter), adapter=adapter):
            result = agent(history=events, repo='r', target_order=3, context='fix tests', response='tests not run')
        self.assertEqual(agent.stages.calls, 5)
        self.assertEqual(result.memory[0]['status'], 'active')
        self.assertEqual(result.prediction['evidence_ids'], ['A:1'])

if __name__ == '__main__': unittest.main()
