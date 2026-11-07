import json
from pathlib import Path
import tempfile
import unittest

try:
    import pandas as pd
    from preference_agent.data import SWEChatData, DataAlignmentError
except ModuleNotFoundError as error:
    if error.name not in {'pandas', 'pyarrow'}: raise
    pd = None

@unittest.skipIf(pd is None, 'Install requirements-data.txt to run parquet tests')
class UserActionTests(unittest.TestCase):
    def test_rejections_and_interruptions_but_not_automatic_denials(self):
        from preference_agent.data import user_actions
        rows = pd.DataFrame([
            dict(turn_type='tool_use', content='{}', tool_name='Bash', tool_call_id='c1', command='git push --force', file_path=None, tool_input_json=None),
            dict(turn_type='tool_result', content="The user doesn't want to proceed with this tool use. The tool use was rejected", tool_name='Bash', tool_call_id='c1', command=None, file_path=None, tool_input_json=None),
            dict(turn_type='user_prompt', content='[Request interrupted by user for tool use]', tool_name=None, tool_call_id=None, command=None, file_path=None, tool_input_json=None),
            dict(turn_type='tool_use', content='{}', tool_name='Edit', tool_call_id='c2', command=None, file_path='a.py', tool_input_json=None),
            dict(turn_type='tool_result', content='Permission for this action was denied by the Claude Code auto mode classifier', tool_name='Edit', tool_call_id='c2', command=None, file_path=None, tool_input_json=None),
        ])
        self.assertEqual(user_actions(rows), ['Rejected tool call Bash: git push --force',
                                              'Interrupted the agent during a tool call'])

@unittest.skipIf(pd is None, 'Install requirements-data.txt to run parquet tests')
class DataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'raw').mkdir()
        sessions, targets, raw = [], [], []
        for i in range(1, 4):
            sid = f's{i}'
            start = f'2026-01-0{i}T00:00:00Z'
            sessions.append(dict(session_id=sid, repo_id='r', session_started_at=start, created_at=start))
            targets.append(dict(session_id=sid, repo_id='r', turn_number=4, rank=i-1,
                                n_sess=3, prompt_pushback='correction', t1=1, split='train'))
            for turn, kind, content in [
                (0, 'user_prompt', 'request-' + sid + 'x'*500),
                (1, 'user_prompt', '<environment_context> injected'),
                (2, 'assistant_response', 'done-' + sid),
                (3, 'tool_use', None),
                (4, 'user_prompt', 'SECRET_FEEDBACK-' + sid),
                (5, 'assistant_response', 'FUTURE-' + sid),
            ]:
                raw.append(dict(session_id=sid, repo_id='r', turn_number=turn, turn_type=kind,
                                content=content, timestamp=f'2026-01-0{i}T00:0{turn}:00Z',
                                prompt_pushback='correction' if turn==4 else 'non_pushback' if kind=='user_prompt' else None,
                                tool_name='Bash' if turn==3 else None,
                                tool_input_json='{"command":"pytest"}' if turn==3 else None,
                                command='pytest' if turn==3 else None, file_path=None))
        pd.DataFrame(sessions).to_parquet(self.root/'sessions.parquet')
        pd.DataFrame(targets).to_parquet(self.root/'targets.parquet')
        pd.DataFrame([dict(repo_id='r', split='train')]).to_parquet(self.root/'repo_split.parquet')
        pd.DataFrame(raw).to_parquet(self.root/'raw/conversations.parquet')
        self.data = SWEChatData(self.root)

    def case(self, index=0):
        row = self.data.select_targets(split='train', k=1, limit=2).iloc[index]
        return self.data.build_case(row, k=1)

    def test_complete_content_tools_and_no_feedback_leak(self):
        case = self.case()
        text=json.dumps(case.target)
        self.assertIn('x'*500, text)
        self.assertIn('pytest', text)
        self.assertNotIn('SECRET_FEEDBACK-s2', text)
        self.assertNotIn('FUTURE-s2', text)
        self.assertNotIn('prompt_pushback', text)
        self.assertEqual(case.history[0].follow_up, 'SECRET_FEEDBACK-s1')
        self.assertEqual(case.expected, {'label':'pushback','t1':1})
        self.assertNotIn('expected', case.model_inputs())
        self.assertLess(case.history[0].order, case.target['target_order'])

    def test_history_fixed_for_later_target(self):
        a,b=self.case(),self.case(1)
        self.assertEqual([e.id for e in a.history], [e.id for e in b.history])
        self.assertEqual(b.metadata['history_session_ids'], ['s1'])

    def test_mutating_current_feedback_does_not_change_inputs(self):
        a=self.case()
        rows=self.data._raw['s2']
        rows.loc[rows.turn_number.eq(4),'content']='different private feedback'
        b=self.case()
        self.assertEqual(a.model_inputs(), b.model_inputs())

    def test_label_mismatch_rejected(self):
        self.case()
        rows=self.data._raw['s2']
        rows.loc[rows.turn_number.eq(4),'prompt_pushback']='rejection'
        with self.assertRaises(DataAlignmentError): self.case()

    def test_overlap_rejected(self):
        self.case()
        self.data._raw['s1']['timestamp']=pd.Timestamp('2026-01-03',tz='UTC')
        with self.assertRaisesRegex(DataAlignmentError,'overlaps'): self.case()

    def test_missing_history_timestamp_rejected(self):
        self.case()
        self.data._raw['s1'].loc[0,'timestamp']=pd.NaT
        with self.assertRaises(DataAlignmentError): self.case()

    def test_split_and_k_filter(self):
        with self.assertRaises(DataAlignmentError):self.data.select_targets(split='test',k=1)
        with self.assertRaises(DataAlignmentError):self.data.select_targets(split='train',k=4)
        with self.assertRaises(ValueError):self.data.select_targets(split='train',k=3)

    def test_split_mapping_mismatch_rejected(self):
        pd.DataFrame([dict(repo_id='r',split='test')]).to_parquet(self.root/'repo_split.parquet')
        with self.assertRaisesRegex(DataAlignmentError,'split'):SWEChatData(self.root)

    def test_label_mapping(self):
        t=pd.read_parquet(self.root/'targets.parquet')
        t['prompt_pushback']='non_pushback';t['t1']=0;t.to_parquet(self.root/'targets.parquet')
        r=pd.read_parquet(self.root/'raw/conversations.parquet')
        r.loc[r.turn_number.eq(4),'prompt_pushback']='non_pushback';r.to_parquet(self.root/'raw/conversations.parquet')
        self.data=SWEChatData(self.root)
        self.assertEqual(self.case().expected, {'label':'no_pushback','t1':0})


    def test_unused_harness_metadata_does_not_require_timestamp(self):
        raw=pd.read_parquet(self.root/'raw/conversations.parquet')
        extra=raw.iloc[0].to_dict()
        extra.update(turn_number=6, turn_type='last_prompt', timestamp=None, content='UNUSED_METADATA')
        pd.concat([raw,pd.DataFrame([extra])],ignore_index=True).to_parquet(self.root/'raw/conversations.parquet')
        self.data=SWEChatData(self.root)
        self.assertNotIn('UNUSED_METADATA', json.dumps(self.case().to_record()))

if __name__=='__main__':unittest.main()
