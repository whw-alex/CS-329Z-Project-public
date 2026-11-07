import unittest
from unittest.mock import Mock
import dspy
from dspy.utils import DummyLM
from preference_agent.core import Event, Rule, Verifier, validate_verification
from preference_agent.dspy_backend import DSPyModel
from preference_agent.history_tools import HistoryTools

class EvidenceQualityTests(unittest.TestCase):
    def event(self, id, context='task'):
        return Event(id,'r',1,context,'change','Please update README','correction')

    def test_same_session_cannot_cross_verify(self):
        events=[self.event('session:1'),self.event('session:2','later context')]
        model=Mock()
        rule=Verifier(model).run(Rule('Update README','r',['session:1']),events)
        model.verify.assert_not_called()
        self.assertEqual(rule.status,'candidate')
        self.assertEqual(rule.independent_supports,1)
        self.assertEqual(rule.verification[0]['verdict'],'skipped_dependent')

    def test_merged_origins_same_session_still_one_support(self):
        rule=Verifier(Mock()).run(Rule('Update README','r',['session:1','session:2']),[self.event('session:1'),self.event('session:2')])
        self.assertEqual(rule.status,'candidate')
        self.assertEqual(rule.independent_supports,1)

    def test_duplicate_context_not_independent(self):
        model=Mock()
        rule=Verifier(model).run(Rule('Update README','r',['one']),[self.event('one'),self.event('two')])
        self.assertEqual(rule.status,'candidate');model.verify.assert_not_called()

    def test_independent_explicit_feedback_activates_with_audit(self):
        model=Mock();model.verify.return_value={'verdict':'support','reason':'Explicit README request','feedback_excerpt':'update README'}
        rule=Verifier(model).run(Rule('Update README','r',['a:1']),[self.event('a:1'),self.event('b:1','another task')])
        self.assertEqual(rule.status,'active');self.assertEqual(rule.independent_supports,2)
        self.assertEqual(rule.verification[0]['feedback_excerpt'],'update README')

    def test_duplicate_context_across_sessions_not_independent(self):
        model=Mock()
        rule=Verifier(model).run(Rule('Update README','r',['a:1']),[self.event('a:1'),self.event('b:1')])
        self.assertEqual(rule.status,'candidate');model.verify.assert_not_called()

    def test_fabricated_excerpt_rejected(self):
        with self.assertRaisesRegex(ValueError,'exact'):
            validate_verification({'verdict':'support','reason':'claim','feedback_excerpt':'invented'},self.event('a:1'))

    def test_acceptance_cannot_establish_preference(self):
        e=self.event('a:1');e.label='no_pushback';e.follow_up='yes'
        result=validate_verification({'verdict':'support','reason':'accepted','feedback_excerpt':'yes'},e)
        self.assertEqual(result['verdict'],'irrelevant')

    def test_search_and_truncated_read_do_not_grant_episode_access(self):
        e=self.event('a:1');e.agent_turn='x'*1000
        tools=HistoryTools([e],[],'r',2,'','',output_chars=256)
        tools.search_history('README');self.assertEqual(tools.read_ids,set())
        self.assertTrue(tools.get_episode('a:1')['truncated']);self.assertEqual(tools.read_ids,set())

    def test_partial_input_is_labeled(self):
        e=self.event('a:1');e.context='[SMOKE TEST: MIDDLE OMITTED]'
        tools=HistoryTools([e],[],'r',2,'','')
        tools.get_episode('a:1');self.assertEqual(tools.access['a:1'],'partial_episode')

    def test_react_cannot_cite_search_only(self):
        model=DSPyModel(critic_mode='react',max_citation_repairs=0);adapter=dspy.JSONAdapter()
        answers=[{'next_thought':'Search','next_tool_name':'search_history','next_tool_args':{'query':'README'}},
                 {'next_thought':'Done','next_tool_name':'finish','next_tool_args':{}},
                 {'reasoning':'Summary','pushback_probability':.3,'rationale':'Prior behavior','evidence_ids':['a:1']}]
        with dspy.context(lm=DummyLM(answers,adapter=adapter),adapter=adapter):
            with self.assertRaisesRegex(ValueError,'get_episode'):
                model.predict_with_history('task','change',[],[self.event('a:1')],'r',2)
