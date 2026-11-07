"""Join SWE-chat labels to observable conversation content, without model calls."""
from dataclasses import asdict, dataclass
from functools import lru_cache
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from .core import Event

ORDER_STRIDE = 1_000_000_000
# SWE-chat prompt_pushback -> binary label used everywhere in this package.
LABEL_MAP = {'non_pushback': 'no_pushback', 'correction': 'pushback',
             'rejection': 'pushback', 'failure_report': 'pushback'}
VISIBLE_TYPES = {'user_prompt', 'assistant_response', 'tool_use', 'tool_result'}
ACTIVITY_TYPES = VISIBLE_TYPES - {'user_prompt'}
RAW_COLUMNS = ['session_id', 'repo_id', 'turn_number', 'turn_type', 'content',
               'timestamp', 'prompt_pushback', 'tool_name', 'tool_input_json',
               'command', 'file_path', 'tool_call_id']
# Claude Code's texts for a user clicking "reject" on a tool call and for an interruption.
REJECTED_TOOL = ("The user doesn't want to proceed with this tool use", "The user doesn't want to take this action")
INTERRUPTED = '[Request interrupted'

def user_actions(rows):
    """Rejected tool calls and interruptions inside an activity window, as short readable strings."""
    actions, calls = [], {}
    for row in rows.to_dict('records'):
        content = row['content'] if isinstance(row['content'], str) else ''
        if row['turn_type'] == 'tool_use':
            calls[row.get('tool_call_id') or len(calls)] = row
        elif row['turn_type'] == 'tool_result' and content.startswith(REJECTED_TOOL):
            call = calls.get(row.get('tool_call_id')) or (list(calls.values())[-1] if calls else {})
            target = next((call.get(k) for k in ['command', 'file_path', 'tool_input_json']
                           if isinstance(call.get(k), str) and call.get(k)), '')
            actions.append(f"Rejected tool call {row.get('tool_name') or call.get('tool_name') or ''}: {target[:200]}".rstrip(': '))
        elif row['turn_type'] == 'user_prompt' and content.lstrip().startswith(INTERRUPTED):
            actions.append('Interrupted the agent' + (' during a tool call' if 'tool use' in content else ''))
    return actions

class DataAlignmentError(ValueError):
    """Missing/ambiguous data or temporal leakage; never silently repaired."""

@lru_cache(maxsize=1)
def processing_rules():
    # Reuse the exact human-prompt and activity rules used by this repository's EDA.
    path = Path(__file__).resolve().parents[2] / 'scripts' / 'targets.py'
    spec = importlib.util.spec_from_file_location('_swechat_target_rules', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

@dataclass
class PreparedCase:
    history: list[Event]
    target: dict
    expected: dict
    metadata: dict

    def model_inputs(self):
        """Only these fields may enter the agent. expected is evaluation-only."""
        return {'history': self.history, **self.target}

    def to_record(self):
        return {'history': [asdict(e) for e in self.history], 'target': self.target,
                'expected': self.expected, 'metadata': self.metadata}

class SWEChatData:
    def __init__(self, data_dir):
        self.root = Path(data_dir).resolve()
        self.targets = pd.read_parquet(self.root / 'targets.parquet')
        self.splits = pd.read_parquet(self.root / 'repo_split.parquet')
        self.sessions = pd.read_parquet(self.root / 'sessions.parquet')
        self.raw_path = self.root / 'raw' / 'conversations.parquet'
        if not self.raw_path.exists(): raise FileNotFoundError(self.raw_path)
        if self.splits.repo_id.duplicated().any(): raise DataAlignmentError('Duplicate repository split')
        if self.sessions.session_id.duplicated().any(): raise DataAlignmentError('Duplicate session ID')
        if self.targets.duplicated(['session_id', 'turn_number']).any():
            raise DataAlignmentError('Duplicate target key')
        mapping = self.splits.set_index('repo_id')['split']
        actual = self.targets.repo_id.map(mapping).fillna('unassigned')
        if not actual.eq(self.targets.split.fillna('unassigned')).all():
            raise DataAlignmentError('targets split disagrees with repo_split')
        if not self.targets.prompt_pushback.isin(LABEL_MAP).all():
            raise DataAlignmentError('Unknown label')
        if not self.targets.t1.eq(self.targets.prompt_pushback.ne('non_pushback').astype(int)).all():
            raise DataAlignmentError('Binary and four-class labels disagree')
        self.sessions['start'] = pd.to_datetime(self.sessions.session_started_at, utc=True, errors='coerce').fillna(
            pd.to_datetime(self.sessions.created_at, utc=True, errors='coerce'))
        self.sessions = self.sessions.sort_values(['repo_id', 'start', 'session_id']).copy()
        self.sessions['rank'] = self.sessions.groupby('repo_id').cumcount()
        self.sessions['n_sess'] = self.sessions.groupby('repo_id').session_id.transform('size')
        index = self.sessions.set_index('session_id')
        for column in ['repo_id', 'rank', 'n_sess']:
            if not self.targets.session_id.map(index[column]).eq(self.targets[column]).all():
                raise DataAlignmentError(f'Target/session {column} mismatch')
        self._raw = {}
        # Off by default; when on, sessions whose raw rows carry one older repository name are accepted.
        self.allow_repo_rename, self.renamed = False, {}

    def select_targets(self, *, split, k, limit=1, repo_id=None):
        if split not in {'train', 'holdout', 'test'}: raise ValueError('Invalid split')
        if k not in {1, 2, 4, 8}: raise ValueError('k must be 1, 2, 4 or 8')
        if limit < 1: raise ValueError('limit must be positive')
        t = self.targets
        selected = t[(t.split == split) & (t.n_sess >= k + 1) & (t['rank'] >= k)]
        if repo_id is not None: selected = selected[selected.repo_id == repo_id]
        selected = selected.sort_values(['repo_id', 'rank', 'turn_number']).head(limit)
        if selected.empty: raise DataAlignmentError('No eligible targets for this selection')
        return selected

    def _row_groups(self):
        """session_id -> row groups, built once from the session_id column only."""
        if getattr(self, '_groups', None) is None:
            source, index = pq.ParquetFile(self.raw_path), {}
            for i in range(source.num_row_groups):
                for sid in source.read_row_group(i, columns=['session_id']).column(0).unique().to_pylist():
                    index.setdefault(sid, []).append(i)
            self._groups = index
        return self._groups

    def _load_sessions(self, ids):
        missing = sorted(set(ids) - set(self._raw))
        if not missing: return
        import pyarrow as pa
        import pyarrow.compute as pc
        source = pq.ParquetFile(self.raw_path)
        columns = [c for c in RAW_COLUMNS if c in set(source.schema_arrow.names)]
        # Read only row groups holding the requested sessions; bounded memory on the 6 GB file.
        groups = sorted({g for sid in missing for g in self._row_groups().get(sid, [])})
        wanted = pa.array(missing)
        parts = [t.filter(pc.is_in(t['session_id'], value_set=wanted))
                 for t in (source.read_row_group(g, columns=columns) for g in groups)]
        frame = (pa.concat_tables(parts).to_pandas() if parts
                 else source.schema_arrow.empty_table().select(columns).to_pandas())
        for column in RAW_COLUMNS:
            if column not in frame: frame[column] = None
        for sid in missing:
            rows = frame[frame.session_id == sid].sort_values('turn_number').reset_index(drop=True)
            if rows.empty or rows.turn_number.duplicated().any():
                raise DataAlignmentError(f'Missing or duplicate raw turns in session {sid}')
            if (rows.turn_number < 0).any() or (rows.turn_number >= ORDER_STRIDE).any():
                raise DataAlignmentError('turn_number outside supported order range')
            rows['timestamp'] = pd.to_datetime(rows.timestamp, utc=True, errors='coerce')
            self._raw[sid] = rows

    @staticmethod
    def _render(rows):
        # Deliberate allowlist: no labels, intent, outcome annotations or reasoning traces.
        records = []
        for row in rows.to_dict('records'):
            if row['turn_type'] not in VISIBLE_TYPES: continue
            record = {'turn_type': row['turn_type'], 'content': row['content'] if pd.notna(row['content']) else ''}
            for key in ['tool_name', 'tool_input_json', 'command', 'file_path']:
                if pd.notna(row[key]): record[key] = row[key]
            records.append(record)
        return json.dumps(records, ensure_ascii=False)

    def _episode(self, label_row):
        sid = label_row['session_id']; raw = self._raw[sid]
        if not raw.repo_id.eq(label_row['repo_id']).all():
            # A renamed repository keeps one consistent old name per session; session_id stays the join key.
            names = raw.repo_id.dropna().unique()
            if not (getattr(self, 'allow_repo_rename', False) and len(names) == 1):
                raise DataAlignmentError('Raw repository mismatch')
            self.renamed.setdefault(sid, names[0])
        numbers = raw.turn_number.eq(label_row['turn_number'])
        positions = raw.index[numbers].tolist()
        if len(positions) != 1: raise DataAlignmentError('Feedback row not uniquely located')
        end = positions[0]; feedback = raw.iloc[end]
        if feedback.turn_type != 'user_prompt' or feedback.prompt_pushback != label_row['prompt_pushback']:
            raise DataAlignmentError('Feedback label does not match raw row')
        rules = processing_rules()
        heads = raw.content.fillna('').str.slice(0, 300)
        human = raw.turn_type.eq('user_prompt') & ~rules.is_non_human(heads)
        if not human.iloc[end]: raise DataAlignmentError('Feedback is not a human prompt')
        earlier = raw.index[human & (raw.index < end)].tolist()
        if not earlier: raise DataAlignmentError('No preceding human prompt')
        begin = earlier[-1]
        activities = raw.iloc[begin + 1:end]
        heads = activities.content.fillna('').str.slice(0, 300).str.strip()
        valid_text = activities.turn_type.eq('assistant_response') & ~heads.isin(rules.EMPTY_REPLIES) & ~heads.str.match(rules.API_ERROR)
        if not (activities.turn_type.eq('tool_use').any() or valid_text.any()):
            raise DataAlignmentError('No substantive agent activity before feedback')
        return {'context': self._render(raw.iloc[:begin + 1]),
                'agent_turn': self._render(activities[activities.turn_type.isin(ACTIVITY_TYPES)]),
                'user_actions': user_actions(activities),
                'follow_up': feedback.content if pd.notna(feedback.content) else '',
                'label': LABEL_MAP[feedback.prompt_pushback],
                'order': int(label_row['rank']) * ORDER_STRIDE + int(feedback.turn_number)}

    def build_case(self, label_row, *, k):
        row = dict(label_row); repo = row['repo_id']; sid = row['session_id']
        if k not in {1, 2, 4, 8} or row['rank'] < k or row['n_sess'] < k + 1:
            raise DataAlignmentError('Invalid fixed-k target')
        expected_split = self.splits.set_index('repo_id')['split'].get(repo)
        if row['split'] not in {'train', 'holdout', 'test'} or expected_split != row['split']:
            raise DataAlignmentError('Target has no valid repository split')
        sessions = self.sessions[self.sessions.repo_id == repo]
        history_sessions = sessions.head(k)
        history_ids = history_sessions.session_id.tolist()
        self._load_sessions(history_ids + [sid])
        target_start = sessions.loc[sessions.session_id == sid, 'start'].iloc[0]
        if pd.isna(target_start): raise DataAlignmentError('Missing target session start time')
        for hsid in history_ids:
            history_raw = self._raw[hsid]
            # Only records that can enter model-visible history need temporal bounds.
            # Harness metadata such as last_prompt may legitimately have no timestamp.
            times = history_raw.loc[history_raw.turn_type.isin(VISIBLE_TYPES), 'timestamp']
            if times.empty or times.isna().any() or times.max() >= target_start:
                raise DataAlignmentError('Missing timestamps or history overlaps target session')
        history_rows = self.targets[(self.targets.repo_id == repo) & self.targets.session_id.isin(history_ids)]
        if not history_rows.split.eq(row['split']).all(): raise DataAlignmentError('History split mismatch')
        history = []
        for item in history_rows.sort_values(['rank', 'turn_number']).to_dict('records'):
            episode = self._episode(item)
            history.append(Event(id=f"{item['session_id']}:{item['turn_number']}", repo=repo, **episode))
        episode = self._episode(row)
        # Current feedback text and label are never part of target/model_inputs.
        target = {'repo': repo, 'target_order': episode['order'],
                  'context': episode['context'], 'response': episode['agent_turn']}
        if any(e.order >= target['target_order'] for e in history):
            raise DataAlignmentError('History is not before target')
        return PreparedCase(history, target,
                            {'label': episode['label'], 't1': int(row['t1'])},
                            {'repo_id': repo, 'session_id': sid, 'feedback_turn_number': int(row['turn_number']),
                             'session_rank': int(row['rank']), 'split': row['split'], 'k': k,
                             'history_session_ids': history_ids, 'history_events': len(history)})

    def cases(self, *, split, k, limit=1, repo_id=None):
        selected = self.select_targets(split=split, k=k, limit=limit, repo_id=repo_id)
        for row in selected.to_dict('records'):
            yield self.build_case(row, k=k)

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Export SWE-chat cases; does not call any model')
    parser.add_argument('--data-dir', type=Path, default=Path(__file__).resolve().parents[2] / 'data')
    parser.add_argument('--split', choices=['train', 'holdout', 'test'], default='train')
    parser.add_argument('--k', type=int, choices=[1, 2, 4, 8], default=1)
    parser.add_argument('--limit', type=int, default=1)
    parser.add_argument('--repo-id')
    parser.add_argument('--history-per-session', type=int, help='Optional stratified temporal history budget; default preserves all events')
    parser.add_argument('--history-seed', type=int, default=329)
    parser.add_argument('--history-context-chars', type=int)
    parser.add_argument('--history-agent-chars', type=int)
    parser.add_argument('--history-feedback-chars', type=int)
    parser.add_argument('--skip-invalid', action='store_true', help='Record temporal/alignment exclusions in manifest.json')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.history_per_session is not None and args.history_per_session < 1:
        parser.error('--history-per-session must be positive')
    caps = [args.history_context_chars, args.history_agent_chars, args.history_feedback_chars]
    if any(cap is not None for cap in caps) and args.history_per_session is None:
        parser.error('History character caps require --history-per-session')
    from .sampling import OMITTED
    if any(cap is not None and cap < len(OMITTED) + 4 for cap in caps):
        parser.error('History character caps must fit the omission marker')
    data = SWEChatData(args.data_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any(args.output_dir.glob('case-*.json')) or (args.output_dir / 'manifest.json').exists():
        raise FileExistsError('Choose a fresh output directory to avoid stale or overwritten cases')
    selected = data.select_targets(split=args.split, k=args.k, limit=args.limit, repo_id=args.repo_id)
    count, skipped = 0, []
    for row in selected.to_dict('records'):
        try:
            case = data.build_case(row, k=args.k)
        except DataAlignmentError as error:
            if not args.skip_invalid: raise
            skipped.append({'session_id': row['session_id'], 'turn_number': int(row['turn_number']),
                            'reason': str(error)})
            continue
        if args.history_per_session is not None:
            from .sampling import sample_history
            case.history, audit = sample_history(case.history, per_session=args.history_per_session, seed=args.history_seed,
                context_chars=args.history_context_chars, agent_chars=args.history_agent_chars,
                feedback_chars=args.history_feedback_chars)
            case.metadata['history_sampling'] = audit
            case.metadata['history_events'] = len(case.history)
        count += 1
        (args.output_dir / f'case-{count:04}.json').write_text(
            json.dumps(case.to_record(), ensure_ascii=False, indent=2), encoding='utf-8')
    manifest = {'selected_targets': len(selected), 'exported_cases': count, 'skipped': skipped,
                'split': args.split, 'k': args.k, 'output_dir': str(args.output_dir)}
    (args.output_dir / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({key: value for key, value in manifest.items() if key != 'skipped'}))
    if count == 0: raise SystemExit('No cases exported; inspect manifest.json or select another repository')

if __name__ == '__main__': main()
