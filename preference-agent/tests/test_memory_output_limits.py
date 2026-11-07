import unittest
from types import SimpleNamespace
from unittest.mock import patch
import dspy
from dspy.utils.exceptions import AdapterParseError
from preference_agent.dspy_backend import DSPyModel,memory_response_truncated
from preference_agent.core import Event

def event(i):return Event(str(i),'r',i,'task','done','Always run tests','pushback')

class MemoryOutputTests(unittest.TestCase):
    def setup_model(self,max_calls=10):
        model=DSPyModel(memory_mode='session',max_calls=max_calls)
        lm=SimpleNamespace(history=[])
        model.session_extractor.lm=lm
        return model,lm
    def failure(self,model,lm,reason='length'):
        lm.history.append({'response':{'choices':[{'finish_reason':reason}]}})
        return AdapterParseError('JSONAdapter',model.session_extractor.signature,'incomplete')

    def test_memory_config_does_not_raise_forecast_or_global_limit(self):
        model,lm=self.setup_model()
        self.assertEqual([getattr(model,k).config.get('max_tokens') for k in ['session_extractor','rule_merger','session_verifier']],[8192]*3)
        self.assertNotIn('max_tokens',model.critic.config)
        self.assertNotIn('max_tokens',model.memory_selector.config)

    def test_finish_metadata_handles_current_and_legacy_shapes(self):
        self.assertTrue(memory_response_truncated({'choices':[{'finish_reason':'length'}]}))
        self.assertTrue(memory_response_truncated((SimpleNamespace(finish_reason='length'),)))
        self.assertFalse(memory_response_truncated({'choices':[{'finish_reason':'stop'}]}))
        self.assertFalse(memory_response_truncated({'text':'length truncated unfinished JSON'}))

    def test_explicit_truncation_splits_once_and_preserves_all_input_ids(self):
        model,lm=self.setup_model();seen=[]
        def invoke(**inputs):
            seen.append([t['event_id'] for t in inputs['turns']])
            if len(seen)==1:raise self.failure(model,lm)
            return dspy.Prediction(preferences=[{'ids':seen[-1]}])
        with patch.object(model.session_extractor,'forward',side_effect=invoke):
            result=model.extract_session([event(i) for i in range(4)])
        self.assertEqual(seen,[['0','1','2','3'],['0','1'],['2','3']])
        self.assertEqual(result,[{'ids':['0','1']},{'ids':['2','3']}])
        self.assertEqual(model.calls,3);self.assertEqual(model.memory_recoveries[0]['status'],'ok')

    def test_non_truncation_parse_error_is_not_retried(self):
        model,lm=self.setup_model()
        def invoke(**inputs):raise self.failure(model,lm,'stop')
        with patch.object(model.session_extractor,'forward',side_effect=invoke) as mock:
            with self.assertRaises(AdapterParseError):model.extract_session([event(1),event(2)])
        self.assertEqual(mock.call_count,1);self.assertEqual(model.memory_recoveries,[])

    def test_stale_length_metadata_cannot_trigger_retry(self):
        model,lm=self.setup_model();lm.history.append({'response':{'finish_reason':'length'}})
        error=AdapterParseError('JSONAdapter',model.session_extractor.signature,'bad')
        with patch.object(model.session_extractor,'forward',side_effect=error) as mock:
            with self.assertRaises(AdapterParseError):model.extract_session([event(1),event(2)])
        self.assertEqual(mock.call_count,1)

    def test_split_child_failure_has_no_recursive_retry(self):
        model,lm=self.setup_model()
        def invoke(**inputs):raise self.failure(model,lm)
        with patch.object(model.session_extractor,'forward',side_effect=invoke) as mock:
            with self.assertRaises(AdapterParseError):model.extract_session([event(i) for i in range(4)])
        self.assertEqual(mock.call_count,2);self.assertEqual(model.memory_recoveries[0]['status'],'failed')

    def test_recovery_respects_shared_model_budget(self):
        model,lm=self.setup_model(max_calls=2);count=0
        def invoke(**inputs):
            nonlocal count
            count+=1
            if count==1:raise self.failure(model,lm)
            return dspy.Prediction(preferences=[])
        with patch.object(model.session_extractor,'forward',side_effect=invoke) as mock:
            with self.assertRaisesRegex(RuntimeError,'budget'):model.extract_session([event(i) for i in range(4)])
        self.assertEqual(model.calls,2);self.assertEqual(mock.call_count,2)

    def test_single_turn_is_not_retried_without_a_smaller_batch(self):
        model,lm=self.setup_model()
        def invoke(**inputs):raise self.failure(model,lm)
        with patch.object(model.session_extractor,'forward',side_effect=invoke) as mock:
            with self.assertRaises(AdapterParseError):model.extract_session([event(1)])
        self.assertEqual(mock.call_count,1);self.assertEqual(model.memory_recoveries,[])
