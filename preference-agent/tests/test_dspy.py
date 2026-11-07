import unittest
from unittest.mock import patch
import dspy
from preference_agent.core import Event, Pipeline, Critic, Rule
from preference_agent.dspy_backend import DSPyModel, DSPyAgent

class DSPyTests(unittest.TestCase):
    def test_stages_and_evidence(self):
        model = DSPyModel(max_calls=10)
        self.assertEqual(len(model.predictors()), 4)
        history = [Event(str(i), 'r', i, f'fix tests task {i}', 'done', 'run tests', 'correction') for i in (1, 2)]
        with patch.object(model.explainer, 'forward', return_value=dspy.Prediction(rule='Run tests')) as explain, \
             patch.object(model.verifier, 'forward', return_value=dspy.Prediction(verdict='support', reason='Explicit request', feedback_excerpt='run tests')), \
             patch.object(model.critic, 'forward', return_value=dspy.Prediction(
                 pushback_probability=.9,
                 rationale='Prior test requests', evidence_ids=['1'])) as critic:
            result = Pipeline(model).run(history, 'r', 3, 'fix tests', 'untested')
        self.assertEqual(len(result['memory']), 1)
        self.assertEqual(result['prediction']['evidence_ids'], ['1'])
        self.assertEqual(result['revised_response'], 'untested')
        self.assertEqual(explain.call_count, 2)
        self.assertNotIn('follow_up', critic.call_args.kwargs)
        self.assertEqual(model.calls, 3)

    def test_unknown_evidence_rejected(self):
        model = DSPyModel()
        with patch.object(model.critic, 'forward', return_value=dspy.Prediction(
            pushback_probability=.5, rationale='x', evidence_ids=['future'])):
            with self.assertRaisesRegex(ValueError, 'Unknown evidence'):
                model.predict('x', 'y', [])

    def test_budget(self):
        model = DSPyModel(max_calls=1)
        with patch.object(model.explainer, 'forward', return_value=dspy.Prediction(rule='Run tests')):
            event = Event('1', 'r', 1, '', '', '', 'correction')
            model.explain(event)
            with self.assertRaisesRegex(RuntimeError, 'budget'): model.explain(event)

    def test_nonfinite_probability_rejected(self):
        model = DSPyModel()
        with patch.object(model.critic, 'forward', return_value=dspy.Prediction(
            pushback_probability=float('nan'),
            rationale='x', evidence_ids=[])):
            with self.assertRaises(ValueError): Critic(model).run('x', 'y', [])

    def test_verifier_dispatch(self):
        model = DSPyModel()
        rule = Rule('Run tests', 'r', ['1'])
        event = Event('2', 'r', 2, '', '', 'run tests', 'correction')
        with patch.object(model.verifier, 'forward', return_value=dspy.Prediction(verdict='support', reason='Explicit request', feedback_excerpt='run tests')) as verify:
            self.assertEqual(model.verify(rule, event)['verdict'], 'support')
        self.assertEqual(verify.call_args.kwargs['event']['id'], '2')

    def test_program_exposes_predictors(self):
        agent = DSPyAgent(critic_mode='fixed')
        self.assertEqual(len(agent.predictors()), 4)
        with patch.object(agent.stages.critic, 'forward', return_value=dspy.Prediction(
            pushback_probability=.3,
            rationale='No history', evidence_ids=[])):
            result = agent(history=[], repo='r', target_order=1, context='fix', response='done')
        self.assertEqual(result.history_ids, [])
        self.assertEqual(result.revised_response, 'done')

    def test_real_dspy_adapter_with_offline_lm(self):
        from dspy.utils import DummyLM
        adapter = dspy.JSONAdapter()
        answers = [
            {'rule': 'Run tests'}, {'rule': 'Run relevant tests'},
            {'verdict': 'support', 'reason': 'Explicit request', 'feedback_excerpt': 'run tests'}, {'verdict': 'support', 'reason': 'Explicit request', 'feedback_excerpt': 'run tests'},
            {'pushback_probability': .9,
             'rationale': 'Earlier test requests', 'evidence_ids': ['1', '2']},
            {'revised_response': 'I will run the relevant tests.'},
        ]
        agent = DSPyAgent(critic_mode='fixed')
        history = [Event(str(i), 'r', i, f'fix tests task {i}', 'done', 'run tests', 'correction') for i in (1, 2)]
        with dspy.context(lm=DummyLM(answers, adapter=adapter), adapter=adapter):
            result = agent(history=history, repo='r', target_order=3,
                           context='fix tests', response='untested', revise=True)
        self.assertEqual(agent.stages.calls, 6)
        self.assertEqual(result.revised_response, 'I will run the relevant tests.')
