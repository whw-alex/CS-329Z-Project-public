import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import dspy
from dspy.utils import DummyLM
from preference_agent.baselines import BM25, BaselineForecaster, RetrievalConfig, retrieve, run_case
from preference_agent.core import Event

try:
    import numpy, sklearn  # noqa: F401
    from preference_agent.baselines import summarize
except ModuleNotFoundError as error:
    if error.name not in {'numpy', 'sklearn'}: raise
    summarize = None

P = .8

def history():
    return [Event('s1:1', 'r', 1, 'fix the parser', 'parser fixed, tests not run', 'run the parser tests', 'correction'),
            Event('s2:1', 'r', 2, 'update the readme', 'readme updated', 'thanks', 'no_pushback'),
            Event('x:1', 'other', 3, 'fix the parser', 'parser fixed', 'run tests', 'correction'),
            Event('s3:9', 'r', 9, 'fix the parser', 'parser fixed', 'future feedback', 'correction')]

class BM25Tests(unittest.TestCase):
    def test_ranks_matching_documents_first(self):
        index = BM25(['parser tests failing', 'readme typo', 'parser parser'])
        top = index.top('parser tests', 3)
        self.assertEqual(top[0][1], 0)
        self.assertNotIn(1, [i for _, i in top])

    def test_empty_corpus_and_query(self):
        self.assertEqual(BM25([]).top('x', 3), [])
        self.assertEqual(BM25(['a b']).top('', 3), [])

    def test_retrieval_is_scoped_to_repo_and_past(self):
        episodes, trace = retrieve(history(), 'r', 5, 'fix the parser', 'parser fixed', RetrievalConfig(top_k=5))
        ids = [e['event_id'] for e in episodes]
        self.assertEqual(trace['history_events'], 2)
        self.assertIn('s1:1', ids)
        self.assertNotIn('x:1', ids); self.assertNotIn('s3:9', ids)
        self.assertEqual(ids[0], 's1:1')

    def test_top_k_and_caps(self):
        long = [Event(f's{i}:1', 'r', i, 'parser ' * 500, 'parser', 'fb ' * 500, 'correction') for i in range(10)]
        episodes, _ = retrieve(long, 'r', 100, 'parser', '', RetrievalConfig(top_k=3, request_chars=100, feedback_chars=100))
        self.assertEqual(len(episodes), 3)
        self.assertTrue(all(len(e['request']) <= 100 and len(e['follow_up']) <= 100 for e in episodes))

class ForecasterTests(unittest.TestCase):
    def target(self):
        return {'repo': 'r', 'target_order': 5, 'context': 'fix the parser', 'response': 'parser fixed, untested'}

    def test_b2_sees_no_history_and_b3_sees_retrieved(self):
        model = BaselineForecaster()
        reply = dspy.Prediction(pushback_probability=P, rationale='x', evidence_ids=[])
        with patch.object(model.forecaster, 'forward', return_value=reply) as forecast:
            b2 = model('b2', history(), **self.target())
            b3 = model('b3', history(), **self.target())
        first, second = forecast.call_args_list
        self.assertEqual(first.kwargs['past_episodes'], [])
        self.assertIn('s1:1', [e['event_id'] for e in second.kwargs['past_episodes']])
        self.assertEqual({e['label'] for e in second.kwargs['past_episodes']}, {'pushback', 'no_pushback'})
        self.assertNotIn('follow_up', first.kwargs)
        self.assertAlmostEqual(b2['pushback_probability'], .8)
        self.assertEqual(b3['arm'], 'b3')
        self.assertEqual(model.calls, 2)

    def test_citation_outside_prompt_fails(self):
        model = BaselineForecaster()
        reply = dspy.Prediction(pushback_probability=P, rationale='x', evidence_ids=['s3:9'])
        with patch.object(model.forecaster, 'forward', return_value=reply):
            result = run_case(model, 'b3', {'history': [asdict(e) for e in history()], 'target': self.target()})
        self.assertEqual(result['status'], 'failed')

    def test_out_of_range_probability_fails(self):
        model = BaselineForecaster()
        reply = dspy.Prediction(pushback_probability=1.3, rationale='x', evidence_ids=[])
        with patch.object(model.forecaster, 'forward', return_value=reply):
            self.assertEqual(run_case(model, 'b2', {'history': [], 'target': self.target()})['status'], 'failed')

    def test_budget(self):
        model = BaselineForecaster(max_calls=1)
        reply = dspy.Prediction(pushback_probability=P, rationale='x', evidence_ids=[])
        with patch.object(model.forecaster, 'forward', return_value=reply):
            model('b2', [], **self.target())
            with self.assertRaisesRegex(RuntimeError, 'budget'): model('b2', [], **self.target())

    def test_real_json_adapter_with_offline_lm(self):
        adapter = dspy.JSONAdapter()
        answers = [{'pushback_probability': P, 'rationale': 'Earlier test request', 'evidence_ids': ['s1:1']}]
        model = BaselineForecaster()
        with dspy.context(lm=DummyLM(answers, adapter=adapter), adapter=adapter):
            result = model('b3', history(), **self.target())
        self.assertEqual(result['evidence_ids'], ['s1:1'])

@unittest.skipIf(summarize is None, 'Install evaluation dependencies')
class SummaryTests(unittest.TestCase):
    def test_common_cases_and_paired_brier(self):
        def pred(p):
            return {'pushback_probability': p, 'retrieval': {'retrieved': 2}, 'evidence_ids': []}
        with tempfile.TemporaryDirectory() as tmp:
            out, names = Path(tmp), []
            for i, (label, b2, b3) in enumerate([('correction', .5, .8), ('no_pushback', .5, .2), ('correction', .5, None)]):
                name = f'case-{i + 1:04}.json'; names.append(name)
                (out / name).write_text(json.dumps({'metadata': {'repo_id': f'r{i}'}, 'target': {'repo': f'r{i}'},
                    'expected': {'label': label, 't1': int(label != 'no_pushback')}}))
                (out / f'baseline-b2-{i + 1:04}.json').write_text(json.dumps({'status': 'ok', 'prediction': pred(b2)}))
                b3_result = {'status': 'ok', 'prediction': pred(b3)} if b3 is not None else {'status': 'failed'}
                (out / f'baseline-b3-{i + 1:04}.json').write_text(json.dumps(b3_result))
            summary, _ = summarize(out, names, bootstrap=10)
            self.assertEqual(summary['common_cases']['cases'], 2)
            self.assertEqual(summary['metrics_per_arm']['b2']['samples'], 3)
            self.assertAlmostEqual(summary['common_cases']['paired_brier']['b2_minus_b3']['mean'], .25 - .04)
            self.assertNotIn('agent', summary['status_counts'])
            self.assertEqual(summary['metrics_per_arm']['b3']['auroc'], 1.0)
            self.assertNotIn('four_class_accuracy', summary['metrics_per_arm']['b2'])
            self.assertTrue((out / 'baselines-cases.csv').exists())

if __name__ == '__main__': unittest.main()
