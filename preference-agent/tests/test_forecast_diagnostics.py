import unittest
from unittest.mock import patch
import dspy
from dspy.utils import DummyLM
from preference_agent.core import Event
from preference_agent.dspy_backend import DSPyModel
from preference_agent.paired import PairedEvaluator, PairedExecutionError
from preference_agent.react_critic import ForecastValidationError
from test_react import action,forecast

class DiagnosticsTests(unittest.TestCase):
    def event(self):return Event('s:1','r',1,'tests','change','run tests','no_pushback')

    def reject(self, actions, history=None, **kwargs):
        adapter=dspy.JSONAdapter();model=DSPyModel(critic_mode='react',max_citation_repairs=0,**kwargs)
        with dspy.context(lm=DummyLM(actions+[action('finish'),forecast(['s:1'])],adapter=adapter),adapter=adapter):
            with self.assertRaises(ForecastValidationError) as caught:
                model.predict_with_history('tests','change',[],history or [self.event()],'r',3)
        return caught.exception.diagnostics

    def test_search_only_is_distinguished_without_thoughts(self):
        diagnostic=self.reject([action('search_history',{'query':'tests'})])
        self.assertEqual(diagnostic['citation_status'][0]['status'],'summary_only_not_read')
        self.assertEqual(diagnostic['read_ids'],[])
        self.assertEqual(diagnostic['tool_trace'][0]['tool'],'search_history')
        self.assertNotIn('next_thought',str(diagnostic));self.assertNotIn('reasoning',str(diagnostic))

    def test_unknown_and_truncated_citations_are_distinguished(self):
        adapter=dspy.JSONAdapter();model=DSPyModel(critic_mode='react')
        with dspy.context(lm=DummyLM([action('finish'),forecast(['foreign'])],adapter=adapter),adapter=adapter):
            with self.assertRaises(ForecastValidationError) as caught:
                model.predict_with_history('tests','change',[],[self.event()],'r',3)
        self.assertEqual(caught.exception.diagnostics['citation_status'][0]['status'],'out_of_scope_or_unknown')
        event=self.event();event.agent_turn='x'*2000
        diagnostic=self.reject([action('get_episode',{'event_id':'s:1'})],history=[event],tool_output_chars=256)
        self.assertEqual(diagnostic['citation_status'][0]['status'],'truncated_tool_result')

    def test_agent_failure_preserves_baseline_and_checkpoints(self):
        model=DSPyModel(critic_mode='react');evaluator=PairedEvaluator(model);updates=[]
        baseline={'prediction':{'evidence_ids':[]}}
        error=ForecastValidationError('bad cite',{'cited_ids':['s:1'],'read_ids':[]})
        with patch.object(evaluator.pipeline,'predict_with_memory',side_effect=[baseline,error]):
            result=evaluator.run([],'r',3,'tests','change',on_update=updates.append)
        self.assertEqual(result['status'],'partial');self.assertEqual(result['no_memory'],baseline)
        self.assertIsNone(result['agent']);self.assertEqual(result['errors']['agent']['diagnostics']['cited_ids'],['s:1'])
        self.assertEqual(updates[0]['arm_status'],{'no_memory':'ok','agent':'pending'})

    def test_invalid_baseline_does_not_discard_valid_agent(self):
        evaluator=PairedEvaluator(DSPyModel(critic_mode='react'))
        with patch.object(evaluator.pipeline,'predict_with_memory',side_effect=[ForecastValidationError('bad',{}),{'prediction':{}}]):
            result=evaluator.run([],'r',3,'tests','change')
        self.assertEqual(result['arm_status'],{'no_memory':'failed','agent':'ok'})

    def test_budget_or_network_errors_stop_without_losing_baseline(self):
        evaluator=PairedEvaluator(DSPyModel(critic_mode='react'));baseline={'prediction':{}};updates=[]
        with patch.object(evaluator.pipeline,'predict_with_memory',return_value=baseline),patch.object(evaluator.pipeline,'build_memory',side_effect=RuntimeError('budget')):
            with self.assertRaises(PairedExecutionError) as caught:
                evaluator.run([],'r',3,'tests','change',on_update=updates.append)
        self.assertEqual(caught.exception.partial_result['no_memory'],baseline)
        self.assertEqual(caught.exception.stage,'memory');self.assertEqual(updates[-1]['status'],'stopped')
