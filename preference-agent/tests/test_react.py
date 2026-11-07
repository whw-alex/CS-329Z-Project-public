import unittest
import dspy
from dspy.utils import DummyLM
from preference_agent.core import Event, Rule
from preference_agent.dspy_backend import DSPyAgent, DSPyModel
from preference_agent.history_tools import HistoryTools

PUSHBACK = .9

def action(name, args=None):
    return {'next_thought': 'Inspect allowed evidence.', 'next_tool_name': name, 'next_tool_args': args or {}}

def forecast(ids):
    return {'reasoning': 'Available evidence supports this estimate.', 'pushback_probability': PUSHBACK,
            'rationale': 'Earlier requests about tests.', 'evidence_ids': ids}

def event(id='e1', repo='r', order=1):
    return Event(id, repo, order, 'Run tests', 'Untested change', 'Please run tests', 'correction')

class HistoryToolTests(unittest.TestCase):
    def test_scope_and_unknown_id(self):
        tools=HistoryTools([event(),event('foreign','other'),event('future',order=5)], [], 'r',3,'','')
        self.assertIn('error',tools.get_episode('foreign'))
        self.assertIn('error',tools.get_episode('future'))
        self.assertIn('error',tools.get_episode('missing'))
        self.assertIn('Please run tests',tools.get_episode('e1')['content'])
        self.assertEqual(tools.exposed_ids, {'e1'})

    def test_budget_and_truncation(self):
        e=event();e.follow_up='x'*1000
        tools=HistoryTools([e],[],'r',3,'','',max_calls=1,output_chars=256)
        self.assertTrue(tools.get_episode('e1')['truncated'])
        self.assertIn('error',tools.inspect_current_activity())
        self.assertEqual(tools.calls,1)

    def test_rule_scope_and_counterexamples(self):
        rules=[Rule('Run tests','r',['e1'],['e2'],.7,'active'),
               Rule('Run tests','r',['future'],[],.7,'active')]
        tools=HistoryTools([event(),event('e2',order=2)],rules,'r',3,'','')
        self.assertEqual(len(tools.rules),1)
        self.assertIn('r0',tools.search_preferences('tests')['content'])
        self.assertEqual(tools.find_counterexamples('r0')['evidence_ids'],['e2'])

    def test_snapshot_not_modified_by_caller(self):
        e=event();tools=HistoryTools([e],[],'r',3,'','')
        e.follow_up='changed'
        self.assertNotIn('changed',tools.get_episode('e1')['content'])

class ReActTests(unittest.TestCase):
    def test_model_selects_tools_and_finishes(self):
        adapter=dspy.JSONAdapter()
        answers=[{'rule':'Run tests'},{'rule':'Run tests'},
                 action('search_history',{'query':'tests'}),action('get_episode',{'event_id':'e1'}),
                 action('finish'),forecast(['e1'])]
        agent=DSPyAgent(max_calls=10)
        with dspy.context(lm=DummyLM(answers,adapter=adapter),adapter=adapter):
            result=agent(history=[event(),event('e2',order=2)],repo='r',target_order=3,
                         context='Fix tests',response='Untested')
        p=result.prediction
        self.assertEqual(p['critic_mode'],'react')
        self.assertEqual([t['tool'] for t in p['tool_trace']],['search_history','get_episode'])
        self.assertEqual(p['tool_calls'],2)
        self.assertEqual(agent.stages.calls,6)
        self.assertEqual(p['evidence_ids'],['e1'])
        self.assertFalse(p['calibrated'])

    def test_denied_future_cannot_be_cited(self):
        adapter=dspy.JSONAdapter();model=DSPyModel(critic_mode='react')
        answers=[action('get_episode',{'event_id':'future'}),action('finish'),forecast(['future'])]
        with dspy.context(lm=DummyLM(answers,adapter=adapter),adapter=adapter):
            with self.assertRaisesRegex(ValueError,'not given'):
                model.predict_with_history('x','y',[],[event('future',order=5)],'r',3)

    def test_lm_budget_counts_inner_decisions(self):
        adapter=dspy.JSONAdapter();model=DSPyModel(critic_mode='react',max_calls=1)
        with dspy.context(lm=DummyLM([action('finish')],adapter=adapter),adapter=adapter):
            with self.assertRaisesRegex(RuntimeError,'budget'):
                model.predict_with_history('x','y',[],[],'r',3)
        self.assertEqual(model.calls,1)

    def test_iteration_limit_still_extracts_forecast(self):
        adapter=dspy.JSONAdapter();model=DSPyModel(critic_mode='react',max_react_iters=1)
        with dspy.context(lm=DummyLM([action('inspect_current_activity'),forecast([])],adapter=adapter),adapter=adapter):
            result=model.predict_with_history('x','y',[],[],'r',3)
        self.assertEqual(result['tool_calls'],1)
        self.assertEqual(model.calls,2)

    def test_request_scopes_do_not_leak_between_predictions(self):
        adapter=dspy.JSONAdapter();model=DSPyModel(critic_mode='react')
        answers=[action('get_episode',{'event_id':'e1'}),action('finish'),forecast(['e1']),
                 action('get_episode',{'event_id':'e1'}),action('finish'),forecast([])]
        with dspy.context(lm=DummyLM(answers,adapter=adapter),adapter=adapter):
            model.predict_with_history('x','y',[],[event()],'r',3)
            result=model.predict_with_history('x','y',[],[event('e2',repo='other')],'other',3)
        self.assertEqual(result['tool_trace'][0]['status'],'denied')
