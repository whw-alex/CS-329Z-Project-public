import unittest
import dspy
from dspy.utils import DummyLM
from preference_agent.core import Event
from preference_agent.dspy_backend import DSPyModel
from preference_agent.react_critic import ForecastValidationError
from test_react import action,forecast

class CitationFeedbackTests(unittest.TestCase):
    def event(self):return Event('s:1','r',1,'tests','change','run tests','no_pushback')
    def run_model(self,answers,**kwargs):
        model=DSPyModel(critic_mode='react',**kwargs);adapter=dspy.JSONAdapter()
        with dspy.context(lm=DummyLM(answers,adapter=adapter),adapter=adapter):
            result=model.predict_with_history('tests','change',[],[self.event()],'r',3)
        return model,result

    def test_agent_chooses_to_read_after_feedback_shared_budget(self):
        answers=[action('search_history',{'query':'tests'}),action('finish'),forecast(['s:1']),
                 action('get_episode',{'event_id':'s:1'}),action('finish'),forecast(['s:1'])]
        model,result=self.run_model(answers,max_react_iters=4,max_tool_calls=2,max_calls=6)
        self.assertFalse(result['first_pass_success']);self.assertEqual(result['citation_repair_attempts'],1)
        self.assertEqual(result['repair_outcome'],'read_and_cited')
        self.assertEqual([t['tool'] for t in result['tool_trace']],['search_history','get_episode'])
        self.assertFalse(any(t.get('initiator')=='harness' for t in result['tool_trace']))
        self.assertEqual(result['decision_calls'],4);self.assertEqual(model.calls,6)
        self.assertEqual(result['stage_calls_in_critic'],6);self.assertEqual(result['repair_stage_calls'],3)
        self.assertEqual(result['attempts'][0]['rejected_prediction']['evidence_ids'],['s:1'])
        self.assertEqual(result['attempts'][0]['diagnostics']['read_ids'],[])
        self.assertNotIn('next_thought',str(result['attempts']))

    def test_model_may_regenerate_without_history_claims(self):
        _,result=self.run_model([action('finish'),forecast(['s:1']),action('finish'),forecast([])],max_react_iters=2)
        self.assertEqual(result['repair_outcome'],'regenerated_without_citations')
        self.assertEqual(result['tool_calls'],0);self.assertEqual(result['evidence_ids'],[])

    def test_second_invalid_forecast_is_not_retried_again(self):
        model=DSPyModel(critic_mode='react',max_react_iters=4);adapter=dspy.JSONAdapter()
        with dspy.context(lm=DummyLM([action('finish'),forecast(['s:1']),action('finish'),forecast(['s:1'])],adapter=adapter),adapter=adapter):
            with self.assertRaises(ForecastValidationError) as caught:
                model.predict_with_history('tests','change',[],[self.event()],'r',3)
        self.assertEqual(len(caught.exception.diagnostics['attempts']),2)
        self.assertEqual(model.calls,4)

    def test_unknown_or_future_reference_never_triggers_repair(self):
        model=DSPyModel(critic_mode='react');adapter=dspy.JSONAdapter()
        with dspy.context(lm=DummyLM([action('finish'),forecast(['future:9'])],adapter=adapter),adapter=adapter):
            with self.assertRaises(ForecastValidationError) as caught:
                model.predict_with_history('tests','change',[],[self.event()],'r',3)
        self.assertFalse(caught.exception.diagnostics['repair_eligible']);self.assertEqual(model.calls,2)

    def test_no_decisions_remaining_does_not_restart_loop(self):
        model=DSPyModel(critic_mode='react',max_react_iters=1);adapter=dspy.JSONAdapter()
        with dspy.context(lm=DummyLM([action('finish'),forecast(['s:1'])],adapter=adapter),adapter=adapter):
            with self.assertRaises(ForecastValidationError) as caught:
                model.predict_with_history('tests','change',[],[self.event()],'r',3)
        self.assertFalse(caught.exception.diagnostics['repair_budget_available']);self.assertEqual(model.calls,2)

    def test_tool_budget_is_not_reset_for_repair(self):
        model=DSPyModel(critic_mode='react',max_tool_calls=1);adapter=dspy.JSONAdapter()
        with dspy.context(lm=DummyLM([action('search_history',{'query':'tests'}),action('finish'),forecast(['s:1'])],adapter=adapter),adapter=adapter):
            with self.assertRaises(ForecastValidationError) as caught:
                model.predict_with_history('tests','change',[],[self.event()],'r',3)
        self.assertEqual(caught.exception.diagnostics['remaining_tool_calls'],0)
        self.assertEqual(len(caught.exception.diagnostics['attempts']),1)

    def test_global_lm_budget_exhaustion_preserves_first_failure(self):
        model=DSPyModel(critic_mode='react',max_calls=2);adapter=dspy.JSONAdapter()
        with dspy.context(lm=DummyLM([action('finish'),forecast(['s:1'])],adapter=adapter),adapter=adapter):
            with self.assertRaisesRegex(RuntimeError,'budget') as caught:
                model.predict_with_history('tests','change',[],[self.event()],'r',3)
        self.assertEqual(model.calls,2)
        self.assertEqual(len(caught.exception.forecast_diagnostics['attempts']),1)

    def test_previous_tool_observations_but_not_thoughts_reach_repair(self):
        from unittest.mock import patch
        model=DSPyModel(critic_mode='react',max_react_iters=4);adapter=dspy.JSONAdapter();captured=[]
        predictor=model.react_critic.runner.react.module
        original=predictor.forward
        def capture(**kwargs):
            captured.append(kwargs.get('validation_feedback'));return original(**kwargs)
        answers=[action('search_history',{'query':'tests'}),action('finish'),forecast(['s:1']),action('get_episode',{'event_id':'s:1'}),action('finish'),forecast(['s:1'])]
        with patch.object(predictor,'forward',side_effect=capture),dspy.context(lm=DummyLM(answers,adapter=adapter),adapter=adapter):
            model.predict_with_history('tests','change',[],[self.event()],'r',3)
        feedback=next(x for x in captured if x['status']=='invalid_citations')
        self.assertEqual(feedback['previous_tool_observations'][0]['tool'],'search_history')
        self.assertNotIn('thought',str(feedback))
