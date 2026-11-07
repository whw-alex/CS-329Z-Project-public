"""Turn a relabeling run (e.g. work/relabel-100) into batch-style case files. Never calls a model.

Each relabeled target is joined to data/targets.parquet, rebuilt from raw conversations with the same
fixed-k history and caps as `preference_agent.batch`, and given the new label as `expected`.
The original SWE-chat label is kept in metadata only. Outputs case-*.json and plan.json, which
`preference_agent.baselines` and `preference_agent.batch` (agent paired runs) read directly.

Label precedence: a filled human_label (N/P) overrides model_binary. NF (not human feedback) and AMBIG
targets are excluded and listed in plan.json. History events keep their SWE-chat labels: only targets
were relabeled.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from .core import save_result
from .sampling import cap_text, sample_history

# Relabel fine labels -> binary label.
FINE_TO_LABEL = {'continue': 'no_pushback', 'clarification': 'no_pushback', 'correction': 'pushback',
                 'rejection': 'pushback', 'failure_report': 'pushback'}

def resolve_label(row):
    """Returns (label, t1, source, note) or None when the target is excluded.

    label is binary: 'pushback' or 'no_pushback'. The fine label stays in metadata.relabel.
    """
    human = row.get('human_label')
    human = str(human).strip().upper() if isinstance(human, str) and human.strip() else ''
    binary, source = (human, 'human') if human else (str(row['model_binary']).strip().upper(), 'model')
    if binary not in {'N', 'P'}:
        return None
    if binary == 'N':
        return 'no_pushback', 0, source, ''
    fine = FINE_TO_LABEL.get(str(row.get('model_fine_label', '')).strip())
    # Human P over a model N leaves no pushback type; only the binary target is used.
    return 'pushback', 1, source, '' if fine and fine != 'no_pushback' else 'no_pushback_fine_label'

def load_history_labels(path):
    """event id -> resolved label tuple, or None for NF/AMBIG (not human feedback: dropped from history)."""
    frame = pd.read_csv(path, dtype={'session_id': str})
    return {f"{r['session_id']}:{int(r['turn_number'])}": resolve_label(r) for r in frame.to_dict('records')}

def apply_history_labels(history, labels):
    """Relabeled history events take the new binary label; NF events are removed; unlabeled ones keep SWE-chat's."""
    kept, audit = [], {'relabeled': 0, 'dropped_not_feedback': 0, 'kept_swechat_label': 0, 'changed': 0}
    for event in history:
        if event.id not in labels:
            audit['kept_swechat_label'] += 1; kept.append(event); continue
        resolved = labels[event.id]
        if resolved is None:
            audit['dropped_not_feedback'] += 1; continue
        audit['relabeled'] += 1
        audit['changed'] += resolved[0] != event.label
        event.label = resolved[0]; kept.append(event)
    return kept, audit

def build(args):
    from .data import DataAlignmentError, SWEChatData
    labels = pd.read_csv(args.relabel_dir / args.labels_file)
    if 'n' not in labels:
        # Curated sets (e.g. work/k4-30/selection.csv) number cases by k4_id.
        if 'k4_id' not in labels: raise ValueError('Labels need an n or k4_id column')
        labels = labels.assign(n=labels.k4_id)
    if labels.duplicated(['session_id', 'turn_number']).any(): raise ValueError('Duplicate relabeled target')
    data = SWEChatData(args.data_dir)
    data.allow_repo_rename = args.allow_repo_rename
    targets = data.targets.set_index(['session_id', 'turn_number'])
    out, cases, skipped = args.output_dir, [], []
    # One pass over the raw file for every needed target and history session.
    needed = set()
    for row in labels.to_dict('records'):
        key = (row['session_id'], int(row['turn_number']))
        if resolve_label(row) is None or key not in targets.index: continue
        repo = targets.loc[key, 'repo_id']
        needed |= {key[0], *data.sessions[data.sessions.repo_id == repo].head(args.k).session_id}
    data._load_sessions(sorted(needed))
    history_labels = load_history_labels(args.history_labels) if args.history_labels else None
    for row in labels.sort_values('n').to_dict('records'):
        key = (row['session_id'], int(row['turn_number']))
        def skip(reason):
            skipped.append({'n': int(row['n']), 'session_id': key[0], 'turn_number': key[1], 'reason': reason})
        resolved = resolve_label(row)
        if resolved is None:
            skip(f"excluded label {row.get('human_label') if isinstance(row.get('human_label'), str) else row['model_binary']}")
            continue
        if key not in targets.index:
            skip('not in targets.parquet'); continue
        target_row = {'session_id': key[0], 'turn_number': key[1], **targets.loc[key].to_dict()}
        if target_row['rank'] < args.k or target_row['n_sess'] < args.k + 1:
            skip(f'fewer than k={args.k} earlier sessions'); continue
        try:
            case = data.build_case(target_row, k=args.k)
        except DataAlignmentError as error:
            skip(str(error)); continue
        # None keeps every event of each history session; field caps still apply.
        per_session = args.history_per_session or max(len(case.history), 1)
        if history_labels is not None:
            case.history, relabel_audit = apply_history_labels(case.history, history_labels)
            case.metadata['history_relabel'] = relabel_audit
        case.history, audit = sample_history(case.history, per_session=per_session,
            seed=args.history_seed, context_chars=args.history_context_chars,
            agent_chars=args.history_agent_chars, feedback_chars=args.history_feedback_chars)
        if args.history_per_session is None: audit['per_session'] = 'all'
        lengths = {'context': len(case.target['context']), 'response': len(case.target['response'])}
        case.target['context'] = cap_text(case.target['context'], args.target_context_chars)
        case.target['response'] = cap_text(case.target['response'], args.target_response_chars)
        label, t1, source, note = resolved
        swechat = case.expected
        case.expected = {'label': label, 't1': t1}
        sessions = case.metadata['history_session_ids'] + [key[0]]
        case.metadata.update(history_sampling=audit, history_events=len(case.history),
            renamed_raw_repo={sid: data.renamed[sid] for sid in sessions if sid in data.renamed},
            target_caps={'source_chars': lengths, 'context_chars': args.target_context_chars,
                         'response_chars': args.target_response_chars},
            relabel={'n': int(row['n']), 'source': source, 'model_fine_label': row.get('model_fine_label'),
                     'model_binary': row.get('model_binary'), 'confidence': row.get('confidence'),
                     'note': note, 'swechat_label': swechat['label'], 'swechat_t1': swechat['t1']})
        name = f'case-{len(cases) + 1:04}.json'
        (out / name).write_text(json.dumps(case.to_record(), ensure_ascii=False, indent=2), encoding='utf-8')
        cases.append(name)
    return cases, skipped

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--relabel-dir', type=Path, required=True, help='Directory with labels.csv')
    parser.add_argument('--labels-file', default='labels.csv', help='CSV inside --relabel-dir, e.g. selection.csv')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, default=Path(__file__).resolve().parents[2] / 'data')
    parser.add_argument('--k', type=int, choices=[1, 2, 4, 8], default=4)
    # Default keeps full history (batch.py defaults to 2); field and target caps match batch.py.
    parser.add_argument('--history-per-session', type=int, default=None,
                        help='Events kept per history session; default keeps all of them')
    parser.add_argument('--history-seed', type=int, default=329)
    parser.add_argument('--history-context-chars', type=int, default=1500)
    parser.add_argument('--history-agent-chars', type=int, default=2500)
    parser.add_argument('--history-feedback-chars', type=int, default=1000)
    parser.add_argument('--target-context-chars', type=int, default=8000)
    parser.add_argument('--target-response-chars', type=int, default=12000)
    parser.add_argument('--history-labels', type=Path,
                        help='labels.csv from relabel.py run on the history turns; NF history turns are dropped')
    parser.add_argument('--allow-repo-rename', action=argparse.BooleanOptionalAction, default=True,
                        help='Accept sessions whose raw rows carry one older repository name (recorded in metadata)')
    args = parser.parse_args()
    if args.history_per_session is not None and args.history_per_session < 1:
        parser.error('--history-per-session must be positive')
    if not (args.relabel_dir / args.labels_file).is_file(): parser.error(f'--relabel-dir has no {args.labels_file}')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any(args.output_dir.iterdir()): parser.error('Choose a fresh --output-dir')
    cases, skipped = build(args)
    settings = {key: (str(value) if isinstance(value, Path) else value) for key, value in vars(args).items()
                if key not in {'output_dir', 'data_dir'}}
    labels = [json.loads((args.output_dir / c).read_text(encoding='utf-8'))['expected'] for c in cases]
    plan = {'kind': 'relabel', 'selection': settings, 'cases': cases, 'skipped': skipped,
            'label_counts': pd.Series([l['label'] for l in labels]).value_counts().to_dict() if labels else {},
            'binary_counts': {'P': sum(l['t1'] for l in labels), 'N': sum(1 - l['t1'] for l in labels)}}
    save_result(plan, args.output_dir / 'plan.json')
    print(json.dumps({k: plan[k] for k in ['label_counts', 'binary_counts']} |
                     {'cases': len(cases), 'skipped': len(skipped)}, indent=2))

if __name__ == '__main__': main()
