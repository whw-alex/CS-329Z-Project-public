import unittest
try:
    import pandas as pd
    from preference_agent.relabel import build_episode, check, instructions, render, sample, binary
except ModuleNotFoundError as error:
    if error.name not in {'pandas', 'pyarrow'}: raise
    pd = None

def session():
    rows = [(0, 'user_prompt', 'Please always use uv, never pip.', {}),
            (1, 'assistant_response', 'Noted.', {}),
            (2, 'user_prompt', 'install requests', {}),
            (3, 'assistant_response', 'Installing with pip.', {}),
            (4, 'tool_use', '{}', dict(tool_name='Bash', tool_call_id='c1', command='pip install requests')),
            (5, 'tool_result', "The user doesn't want to proceed with this tool use. The tool use was rejected", dict(tool_name='Bash', tool_call_id='c1')),
            (6, 'user_prompt', '[Request interrupted by user for tool use]', {}),
            (7, 'tool_use', '{}', dict(tool_name='Read', tool_call_id='c2', file_path='a.py')),
            (8, 'tool_result', '', dict(tool_name='Read', tool_call_id='c2')),
            (9, 'queue_operation', '{"task_id": "x"}', dict(queue_op_subtype='user_prompt_enqueued')),
            (10, 'queue_operation', 'I said uv!', dict(queue_op_subtype='user_prompt_enqueued')),
            (11, 'user_prompt', 'I said uv!', {}),
            (12, 'assistant_response', 'FUTURE reply', {})]
    return pd.DataFrame([dict(session_id='s', turn_number=t, turn_type=k, content=c, **x) for t, k, c, x in rows])

@unittest.skipIf(pd is None, 'Install requirements-data.txt to run relabel tests')
class RelabelTests(unittest.TestCase):
    def test_episode_window_actions_and_no_future(self):
        e = build_episode(session(), 11)
        self.assertEqual(e['earlier_messages'], ['Please always use uv, never pip.'])
        self.assertEqual((e['previous_agent_reply'], e['previous_request'], e['reply']), ('Noted.', 'install requests', 'I said uv!'))
        tools = [s for s in e['steps'] if s['kind'] == 'tool']
        self.assertEqual([(t['tool'], t['input'], t['result']) for t in tools][1], ('Read', 'a.py', '(no output)'))
        self.assertIn({'kind': 'interrupted'}, e['steps'])
        self.assertEqual(e['user_actions'][:2], ['Rejected tool call Bash: pip install requests', 'Interrupted the agent during a tool call'])
        # JSON queue entries are task notifications, not typed text; typed ones are kept and linked to the reply.
        self.assertEqual(e['user_actions'][2:], ['Typed while the agent was busy (delivered): I said uv! (this became the next message)'])
        text = render(e)
        self.assertNotIn('FUTURE', text)
        self.assertIn("### Developer's next message (label this)\nI said uv!", text)

    def test_non_human_feedback_is_not_a_target(self):
        rows = session(); rows.loc[rows.turn_number == 11, 'content'] = '<command-name>/clear</command-name>'
        with self.assertRaises(ValueError): build_episode(rows, 11)

    def test_quote_checks_and_binary_mapping(self):
        e = build_episode(session(), 11)
        ok = check({'evidence_quote': 'always use uv', 'fine_label': 'correction', 'is_human_feedback': True}, e)
        self.assertEqual((ok['binary'], ok['quote_found_in'], ok['issues']), ('P', ['earlier'], []))
        bad = check({'evidence_quote': 'invented', 'fine_label': 'continue', 'is_human_feedback': True}, e)
        self.assertEqual((bad['binary'], bad['issues']), ('N', ['quote_not_found']))
        self.assertEqual([binary('clarification', True), binary('rejection', True), binary('none', False)], ['N', 'P', 'NF'])

    def test_sample_is_seeded_one_per_session_and_excludes(self):
        t = pd.DataFrame({'session_id': ['a', 'a', 'b', 'c', 'd'], 'turn_number': [1, 2, 1, 1, 1]})
        a, b = sample(t, 2, 0, [('b', 1)]), sample(t, 2, 0, [('b', 1)])
        self.assertTrue(a.equals(b))
        self.assertEqual(a.session_id.nunique(), 2)
        self.assertNotIn('b', a.session_id.tolist())
        with self.assertRaises(ValueError): sample(t, 4, 0, [('b', 1)])

    def test_history_targets_are_first_k_sessions_by_start_once_per_repo(self):
        import tempfile
        from pathlib import Path
        from preference_agent.relabel import history_targets
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Session ids sort differently from start times; created_at fills a missing session_started_at.
            pd.DataFrame([dict(session_id='z', repo_id='r', session_started_at='2026-01-01T00:00:00Z', created_at=None),
                          dict(session_id='a', repo_id='r', session_started_at=None, created_at='2026-01-02T00:00:00Z'),
                          dict(session_id='m', repo_id='r', session_started_at='2026-01-03T00:00:00Z', created_at=None),
                          dict(session_id='o', repo_id='other', session_started_at='2026-01-01T00:00:00Z', created_at=None)]
                         ).to_parquet(root / 'sessions.parquet')
            pd.DataFrame([dict(session_id=s, turn_number=t, repo_id=r, prompt_pushback='correction')
                          for s, t, r in [('z', 2, 'r'), ('z', 5, 'r'), ('a', 1, 'r'), ('m', 1, 'r'), ('o', 1, 'other')]]
                         ).to_parquet(root / 'targets.parquet')
            cases = pd.DataFrame({'session_id': ['m', 'm']})  # two cases in one repo
            rows, history = history_targets(cases, root, 2)
        self.assertEqual(history.session_id.tolist(), ['z', 'a'])
        self.assertEqual(list(zip(rows.session_id, rows.turn_number)), [('z', 2), ('z', 5), ('a', 1)])

    def test_prompt_markers_and_offline_dspy_label(self):
        import dspy
        from dspy.utils import DummyLM
        from preference_agent.relabel import label_signature
        self.assertIn('What counts as pushback', instructions())
        self.assertNotIn('Worked examples', instructions())
        adapter = dspy.JSONAdapter()
        answer = {'reasoning': 'Repeats an earlier instruction the agent ignored.', 'is_human_feedback': True,
                  'not_feedback_reason': '', 'fine_label': 'correction', 'pushback_source': 'typed+rejected_tool_call',
                  'evidence_quote': 'I said uv!', 'confidence': 'high', 'stated_preference': 'Use uv instead of pip.'}
        with dspy.context(lm=DummyLM([answer], adapter=adapter), adapter=adapter):
            result = dspy.Predict(label_signature())(episode=render(build_episode(session(), 11)))
        self.assertEqual((result.fine_label, result.confidence), ('correction', 'high'))

if __name__ == '__main__': unittest.main()
