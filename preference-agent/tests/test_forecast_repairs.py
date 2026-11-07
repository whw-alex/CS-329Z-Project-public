import unittest
from unittest.mock import Mock
import dspy
from dspy.utils import DummyLM
from preference_agent.core import Event, Rule, Verifier
from preference_agent.dspy_backend import DSPyModel, Forecast
from preference_agent.sampling import sample_history,cap_text,OMITTED
from preference_agent.paired import PairedEvaluator
from test_react import action, forecast

class ForecastRepairTests(unittest.TestCase):
    def event(self,id,label='correction',order=1):
        return Event(id,'r',order,'task '+id,'change','Please run tests',label)

    def test_bad_quote_is_audited_not_counted_then_valid_event_can_support(self):
        model=Mock();model.verify.side_effect=[{'verdict':'support','reason':'wrong quote','feedback_excerpt':'invented'},
            {'verdict':'support','reason':'explicit request','feedback_excerpt':'run tests'}]
        rule=Verifier(model).run(Rule('Run tests','r',['a:1']),[self.event('a:1'),self.event('b:1'),self.event('c:1')])
        self.assertEqual(rule.evidence,['a:1','c:1']);self.assertEqual(rule.status,'active')
        self.assertEqual(rule.verification[0]['verdict'],'invalid_evidence')
        self.assertEqual(rule.verification[0]['feedback_excerpt'],'invented')

    def test_runtime_budget_and_unrelated_value_errors_are_not_swallowed(self):
        for error in [RuntimeError('budget'),ValueError('unrelated internal error')]:
            model=Mock();model.verify.side_effect=error
            with self.assertRaises(type(error)):
                Verifier(model).run(Rule('Run tests','r',['a:1']),[self.event('a:1'),self.event('b:1')])

    def test_history_sampling_keeps_late_positive_and_is_deterministic(self):
        events=[self.event('session:'+str(i),'no_pushback' if i<5 else 'failure_report',i) for i in range(1,7)]
        sampled,audit=sample_history(events,per_session=2,seed=329)
        self.assertEqual({e.label for e in sampled},{'no_pushback','pushback'})
        self.assertEqual(sample_history(events,per_session=2,seed=329)[0],sampled)
        self.assertEqual(len(events),6);self.assertEqual(audit['source_events'],6)
        self.assertEqual([e.order for e in sampled],sorted(e.order for e in sampled))

    def test_character_caps_are_real_limits_and_preserve_source(self):
        event=self.event('a:1');event.context='a'*200
        selected,_=sample_history([event],context_chars=80)
        self.assertLessEqual(len(selected[0].context),80);self.assertIn(OMITTED,selected[0].context)
        self.assertEqual(event.context,'a'*200)
        with self.assertRaises(ValueError):cap_text('abcde',2)

    def test_search_first_runs_even_with_empty_active_rules(self):
        model=DSPyModel(critic_mode='react',retrieval_policy='search_first');adapter=dspy.JSONAdapter()
        answers=[action('get_episode',{'event_id':'a:1'}),action('finish'),forecast(['a:1'])]
        with dspy.context(lm=DummyLM(answers,adapter=adapter),adapter=adapter):
            result=model.predict_with_history('tests','change',[],[self.event('a:1')],'r',3)
        self.assertEqual([t['tool'] for t in result['tool_trace']],['search_history','get_episode'])
        self.assertEqual(result['tool_trace'][0]['initiator'],'harness')
        self.assertEqual(result['history_event_count'],1);self.assertEqual(result['tool_calls'],2)
        self.assertEqual(model.calls,3)

    def test_empty_history_same_policy_skips_bootstrap(self):
        model=DSPyModel(critic_mode='react',retrieval_policy='search_first');adapter=dspy.JSONAdapter()
        with dspy.context(lm=DummyLM([action('finish'),forecast([])],adapter=adapter),adapter=adapter):
            result=model.predict_with_history('tests','change',[],[],'r',3)
        self.assertEqual(result['tool_calls'],0);self.assertEqual(result['history_event_count'],0)

    def test_bootstrap_uses_existing_tool_budget(self):
        model=DSPyModel(critic_mode='react',retrieval_policy='search_first',max_tool_calls=1);adapter=dspy.JSONAdapter()
        with dspy.context(lm=DummyLM([action('get_episode',{'event_id':'a:1'}),action('finish'),forecast([])],adapter=adapter),adapter=adapter):
            result=model.predict_with_history('tests','change',[],[self.event('a:1')],'r',3)
        self.assertEqual(result['tool_calls'],1);self.assertEqual(result['evidence_ids'],[])

    def test_paired_controller_and_content_addressed_cache(self):
        model=Mock();model.critic_mode='react'
        model.predict_with_history.return_value={'pushback_probability':.3}
        evaluator=PairedEvaluator(model);events=[self.event('a:1','no_pushback')]
        first=evaluator.run(events,'r',3,'task','response');second=evaluator.run(events,'r',4,'task','response')
        self.assertFalse(first['protocol']['memory_reused']);self.assertTrue(second['protocol']['memory_reused'])
        calls=model.predict_with_history.call_args_list
        self.assertEqual(calls[0].args[3],[]);self.assertEqual(calls[1].args[3],events)
        self.assertEqual(calls[0].args[:2],calls[1].args[:2])
        events[0].context='changed';third=evaluator.run(events,'r',4,'task','response')
        self.assertFalse(third['protocol']['memory_reused'])

    def test_signature_defines_hidden_future_not_visible_request(self):
        self.assertIn('NEXT message AFTER',Forecast.instructions);self.assertIn('future message is hidden',Forecast.instructions)
        self.assertIn('Absence of feedback',Forecast.instructions)

class PairedCLITests(unittest.TestCase):
    def test_paired_cli_uses_two_react_forecasts_without_network(self):
        import json
        import sys
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from preference_agent.__main__ import main
        adapter=dspy.JSONAdapter()
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/'case.json';output=Path(directory)/'output.json'
            source.write_text(json.dumps({'history':[],'target':{'repo':'r','target_order':1,'context':'task','response':'done'},'expected':{'label':'correction','t1':1}}))
            answers=[action('finish'),forecast([]),action('finish'),forecast([])]
            with dspy.context(lm=DummyLM(answers,adapter=adapter),adapter=adapter), patch('preference_agent.dspy_backend.configure_openrouter',return_value='openrouter/test/model'), patch.object(sys,'argv',['agent',str(source),'--backend','openrouter','--paired','--output',str(output)]):
                main()
            result=json.loads(output.read_text())
            self.assertEqual(result['protocol']['controller'],'react')
            self.assertEqual(result['stage_calls'],4)
            self.assertEqual(result['agent']['prediction']['retrieval_policy'],'autonomous')
            self.assertNotIn('expected',result)
