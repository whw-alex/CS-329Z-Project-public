import json
import tempfile
import unittest
from pathlib import Path
try:
    import pandas as pd
    from preference_agent.batch import select_balanced, summarize
except ModuleNotFoundError as error:
    if error.name not in {'numpy', 'pandas', 'pyarrow', 'sklearn'}: raise
    pd = None

def prediction(no_pushback):
    return {'prediction': {'pushback_probability': 1 - no_pushback, 'tool_trace': [{'tool': 'get_episode'}],
                           'evidence_ids': ['e1'], 'first_pass_success': True}, 'memory': []}

@unittest.skipIf(pd is None, 'Install evaluation/data dependencies')
class BatchTests(unittest.TestCase):
    def targets(self):
        rows = []
        for repo in 'abc':
            for i, t1 in enumerate([1, 1, 0, 0, 0]):
                rows.append(dict(repo_id=repo, session_id=f'{repo}{i}', turn_number=i, rank=i + 1, n_sess=6,
                                 split='holdout', t1=t1))
        rows.append(dict(repo_id='d', session_id='d0', turn_number=0, rank=1, n_sess=6, split='holdout', t1=1))
        rows.append(dict(repo_id='a', session_id='h', turn_number=0, rank=0, n_sess=6, split='holdout', t1=1))
        return pd.DataFrame(rows)

    def test_selection_is_seeded_balanced_and_excludes_history_sessions(self):
        a = select_balanced(self.targets(), split='holdout', k=1, repos=2, per_label=2, seed=1)
        b = select_balanced(self.targets(), split='holdout', k=1, repos=2, per_label=2, seed=1)
        self.assertEqual(a[0], b[0])
        self.assertNotIn('d', a[0])
        for (repo, label), rows in a[1].items():
            self.assertTrue(all(r['t1'] == label and r['rank'] >= 1 for r in rows))
            self.assertNotIn('h', [r['session_id'] for r in rows])

    def test_too_few_balanced_repositories_raise(self):
        with self.assertRaisesRegex(ValueError, 'repositories'):
            select_balanced(self.targets(), split='holdout', k=1, repos=4, per_label=2, seed=1)

    def test_summary_uses_only_complete_pairs(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp); names = []
            for i, (label, status) in enumerate([('correction', 'ok'), ('no_pushback', 'ok'), ('correction', 'partial')]):
                name = f'case-{i + 1:04}.json'; names.append(name)
                (out / name).write_text(json.dumps({'metadata': {'repo_id': 'r'},
                    'expected': {'label': label, 't1': int(label != 'no_pushback')}}))
                result = {'status': status, 'no_memory': prediction(.5), 'agent': prediction(.2 if label != 'no_pushback' else .8),
                          'batch': {'stage_calls': 3, 'seconds': 1}}
                if status != 'ok': result['agent'] = None
                (out / name.replace('case-', 'result-')).write_text(json.dumps(result))
            summary, rows = summarize(out, {'cases': names}, bootstrap=10)
            self.assertEqual(summary['complete_pairs'], 2)
            self.assertAlmostEqual(summary['paired_brier']['no_memory_minus_agent'], .25 - .04)
            self.assertEqual(summary['total_stage_calls'], 9)
            self.assertTrue((out / 'cases.csv').exists())

if __name__ == '__main__': unittest.main()
