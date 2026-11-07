import unittest
from dataclasses import asdict
from preference_agent.core import Event, Rule, SessionMemory, Pipeline, select_memory_for_target

def event(id, order, reply='Always run tests for code changes'):
    return Event(id,'r',order,'task '+id,'done',reply,'pushback')

def item(id,scope='when changing code',kind='stated_preference',text='Run tests'):
    return {'text':text,'polarity':'prefers','kind':kind,'applies_when':scope,
            'reusable':kind!='task_instruction','evidence':[{'event_id':id,'quote':'run tests'}]}

class MemoryModel:
    scoped_memory=True
    def __init__(self,items,merged=None,verdicts=None):
        self.items=items;self.merged=merged;self.verdicts=verdicts or [];self.merge_inputs=[]
    def extract_session(self,events):
        ids={e.id for e in events}
        return [i for i in self.items if i.get('evidence',[{}])[0].get('event_id') in ids]
    def merge_rules(self,candidates,max_rules):
        self.merge_inputs=candidates
        return self.merged if self.merged is not None else [{'text':c['text'],'sources':[c['id']]} for c in candidates]
    def verify_session(self,rules,events):return self.verdicts

class ScopedExtractionTests(unittest.TestCase):
    def test_task_instruction_stays_in_history_not_rules(self):
        e=event('A:1',1,'make a PR');i=item('A:1',kind='task_instruction');i['evidence'][0]['quote']='make a PR'
        model=MemoryModel([i])
        self.assertEqual(SessionMemory(model).run([e],'r'),[])
        self.assertEqual(e.follow_up,'make a PR')
        self.assertEqual(model.merge_inputs,[])

    def test_missing_scope_or_unclassified_items_fail_closed(self):
        bad=[item('A:1',scope=''),{'text':'Run tests','evidence':[{'event_id':'A:1','quote':'run tests'}]},item('A:1',kind='unknown')]
        self.assertEqual(SessionMemory(MemoryModel(bad)).run([event('A:1',1)],'r'),[])

    def test_merger_cannot_combine_distinct_conditions_into_active_rule(self):
        model=MemoryModel([item('A:1','when changing production code'),item('B:1','when changing prototypes')],
                          merged=[{'text':'Always run every test','sources':[0,1]}])
        rules=SessionMemory(model).run([event('A:1',1),event('B:1',2)],'r')
        self.assertEqual(len(rules),2)
        self.assertEqual({r.applies_when for r in rules},{'when changing production code','when changing prototypes'})
        self.assertTrue(all(r.status=='candidate' for r in rules))

    def test_merge_preserves_condition_text_and_original_quotes(self):
        model=MemoryModel([item('A:1'),item('B:1')],merged=[{'text':'Always do everything','sources':[0,1]}])
        rules=SessionMemory(model).run([event('A:1',1),event('B:1',2)],'r');self.assertEqual(len(rules),1)
        r=rules[0];self.assertEqual((r.text,r.applies_when,r.status),('Run tests','when changing code','active'))
        self.assertEqual([v['feedback_excerpt'] for v in r.verification],['run tests','run tests'])
        self.assertIn('evidence',model.merge_inputs[0])
        self.assertEqual(Rule(**asdict(r)),r)

    def test_true_quote_outside_scope_does_not_activate(self):
        model=MemoryModel([item('A:1')],verdicts=[{'rule_id':'m0','event_id':'B:1','verdict':'support',
            'reason':'This only asks for this task','feedback_excerpt':'run tests','scope_supported':False}])
        r=SessionMemory(model).run([event('A:1',1),event('B:1',2)],'r')[0]
        self.assertEqual((r.independent_supports,r.status),(1,'candidate'))
        self.assertIn('out_of_scope',[v['verdict'] for v in r.verification])

    def test_verified_same_scope_can_activate(self):
        model=MemoryModel([item('A:1')],verdicts=[{'rule_id':'m0','event_id':'B:1','verdict':'support',
            'reason':'Same general condition','feedback_excerpt':'run tests','scope_supported':True}])
        r=SessionMemory(model).run([event('A:1',1),event('B:1',2)],'r')[0]
        self.assertEqual((r.independent_supports,r.status),(2,'active'))

class Selector:
    scoped_memory=True
    critic_mode='react'
    def __init__(self,decisions):self.decisions=decisions;self.views=[];self.received=[]
    def select_memory(self,rules,context):self.views.append((rules,context));return self.decisions
    def predict_with_history(self,context,response,rules,history,repo,target_order):
        self.received=rules
        return {'pushback_probability':.3,'rationale':'test','evidence_ids':[]}

def rule(**overrides):
    args=dict(text='Open the requested PR',repo='r',evidence=['A:1'],status='active',
              kind='stated_preference',applies_when='when explicitly asked to open a PR',reusable=True)
    args.update(overrides);return Rule(**args)

class ApplicabilityTests(unittest.TestCase):
    def test_only_active_scoped_reusable_rules_are_sent_to_selector(self):
        model=Selector([{'rule_id':'m0','applies':True,'context_quote':'open a PR','reason':'Requested'}])
        selected,audit=select_memory_for_target(model,[rule(),rule(status='candidate'),rule(kind='task_instruction',reusable=False),rule(kind='legacy')],'Please open a PR')
        self.assertEqual(len(selected),1);self.assertEqual(len(model.views[0][0]),1)
        self.assertEqual(sum(a['status']=='applicable' for a in audit),1)

    def test_no_current_request_match_does_not_inject_historical_instruction(self):
        model=Selector([{'rule_id':'m0','applies':False,'context_quote':'','reason':'Task is README editing'}])
        selected,_=select_memory_for_target(model,[rule()],'Update the README')
        self.assertEqual(selected,[])

    def test_invented_or_duplicate_current_quotes_fail_closed(self):
        for decisions in [[{'rule_id':'m0','applies':True,'context_quote':'make a PR'}],
                          [{'rule_id':'m0','applies':True,'context_quote':'README'}]*2,
                          [{'rule_id':'foreign','applies':True,'context_quote':'README'}]]:
            with self.subTest(decisions=decisions):
                self.assertEqual(select_memory_for_target(Selector(decisions),[rule()],'Update README')[0],[])

    def test_pipeline_preserves_audit_memory_but_hides_rejected_rules(self):
        model=Selector([{'rule_id':'m0','applies':False,'context_quote':'','reason':'No PR request'}])
        r=Pipeline(model).predict_with_memory([event('A:1',1)],[rule()],repo='r',target_order=3,context='Update README',response='README updated')
        self.assertEqual(model.received,[]);self.assertEqual(r['used_memory'],[])
        self.assertEqual(len(r['memory']),1);self.assertEqual(len(r['memory_selection']),1)

    def test_foreign_evidence_is_rejected_before_selection(self):
        model=Selector([])
        with self.assertRaisesRegex(ValueError,'out-of-scope'):
            Pipeline(model).predict_with_memory([event('A:1',1)],[rule(evidence=['future'])],repo='r',target_order=3,context='Open a PR',response='done')
        self.assertEqual(model.views,[])

    def test_legacy_cache_needs_regeneration_and_costs_no_selection_call(self):
        model=Selector([])
        selected,audit=select_memory_for_target(model,[rule(kind='legacy')],'Open a PR')
        self.assertEqual(selected,[]);self.assertEqual(model.views,[])
        self.assertEqual(audit[0]['status'],'not_reusable_or_legacy')
