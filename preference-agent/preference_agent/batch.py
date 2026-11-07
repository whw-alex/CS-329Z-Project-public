"""Small balanced paired pilot over SWE-chat cases with checkpoints and resume.

Two phases: case preparation never calls a model; running makes paid calls.
Target labels are used only to balance the selection and to score results.
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .core import Event, save_result
from .sampling import cap_text

DONE = {'ok', 'partial', 'failed', 'stopped'}

def select_balanced(targets, *, split, k, repos, per_label, seed):
    """Seeded repository choice; per repository, shuffled candidates for each binary label."""
    t = targets[(targets.split == split) & (targets.n_sess >= k + 1) & (targets['rank'] >= k)]
    counts = t.groupby(['repo_id', 't1']).size().unstack(fill_value=0).reindex(columns=[0, 1], fill_value=0)
    eligible = sorted(counts.index[(counts[0] >= per_label) & (counts[1] >= per_label)])
    if len(eligible) < repos:
        raise ValueError(f'Only {len(eligible)} repositories have {per_label} targets of each label')
    rng = np.random.default_rng(seed)
    chosen = sorted(rng.choice(eligible, size=repos, replace=False).tolist())
    plan = {}
    for repo in chosen:
        for label in (1, 0):
            rows = t[(t.repo_id == repo) & (t.t1 == label)].sort_values(['rank', 'turn_number'])
            plan[(repo, label)] = rows.iloc[rng.permutation(len(rows))].to_dict('records')
    return chosen, plan

def temporally_valid(data, targets, *, k):
    """Targets whose session starts after every visible history record; mirrors build_case's check once per repo."""
    from .data import VISIBLE_TYPES
    starts = data.sessions.set_index('session_id')['start']
    keep = []
    for repo, rows in targets.groupby('repo_id'):
        history_ids = data.sessions[data.sessions.repo_id == repo].head(k).session_id.tolist()
        data._load_sessions(history_ids)
        times = [data._raw[sid].loc[data._raw[sid].turn_type.isin(VISIBLE_TYPES), 'timestamp'] for sid in history_ids]
        if any(x.empty or x.isna().any() for x in times): continue
        end = max(x.max() for x in times)
        keep.append(rows[rows.session_id.map(starts) > end])
    return pd.concat(keep) if keep else targets.iloc[:0]

def eligible_targets(data, args):
    """Fixed-k targets after optional language/user-action filters and the temporal check, with a count funnel."""
    t = data.targets[(data.targets.split == args.split) & (data.targets.n_sess >= args.k + 1)
                     & (data.targets['rank'] >= args.k)]
    funnel = [('fixed-k targets in split', len(t))]
    if args.english_only or args.exclude_user_actions:
        path = args.data_dir / 'target_flags.parquet'
        if not path.exists(): raise FileNotFoundError(f'{path}; run python scripts/target_flags.py first')
        flags = pd.read_parquet(path, columns=['session_id', 'turn_number', 'language', 'repo_language',
                                               'user_action_in_window'])
        t = t.merge(flags, on=['session_id', 'turn_number'], how='left', validate='one_to_one')
        if t.user_action_in_window.isna().any(): raise ValueError('target_flags.parquet is stale; rebuild it')
        if args.english_only:
            # English feedback in a mostly-English repository, so history is English too.
            t = t[t.language.eq('ENGLISH') & t.repo_language.eq('ENGLISH')]
            funnel.append(('... English target and repository', len(t)))
        if args.exclude_user_actions:
            t = t[~t.user_action_in_window.astype(bool)]
            funnel.append(('... no tool rejection/interruption before feedback', len(t)))
    t = temporally_valid(data, t, k=args.k)
    funnel.append(('... history ends before target session', len(t)))
    return t, funnel

def prepare(args, out):
    from .data import DataAlignmentError, SWEChatData
    from .sampling import sample_history
    data = SWEChatData(args.data_dir)
    eligible, funnel = eligible_targets(data, args)
    chosen, plan = select_balanced(eligible, split=args.split, k=args.k,
                                   repos=args.repos, per_label=args.per_label, seed=args.seed)
    cases, skipped, shortfalls = [], [], []
    for repo in chosen:
        for label in (1, 0):
            built = 0
            for row in plan[(repo, label)]:
                if built == args.per_label: break
                try:
                    case = data.build_case(row, k=args.k)
                except DataAlignmentError as error:
                    skipped.append({'session_id': row['session_id'], 'turn_number': int(row['turn_number']),
                                    'reason': str(error)})
                    continue
                case.history, audit = sample_history(case.history, per_session=args.history_per_session,
                    seed=args.history_seed, context_chars=args.history_context_chars,
                    agent_chars=args.history_agent_chars, feedback_chars=args.history_feedback_chars)
                case.metadata.update(history_sampling=audit, history_events=len(case.history))
                lengths = {'context': len(case.target['context']), 'response': len(case.target['response'])}
                case.target['context'] = cap_text(case.target['context'], args.target_context_chars)
                case.target['response'] = cap_text(case.target['response'], args.target_response_chars)
                case.metadata['target_caps'] = {'source_chars': lengths, 'context_chars': args.target_context_chars,
                                                'response_chars': args.target_response_chars}
                name = f'case-{len(cases) + 1:04}.json'
                (out / name).write_text(json.dumps(case.to_record(), ensure_ascii=False, indent=2), encoding='utf-8')
                cases.append(name); built += 1
            if built < args.per_label:
                shortfalls.append({'repo_id': repo, 't1': label, 'built': built})
    return {'cases': cases, 'repositories': chosen, 'funnel': funnel, 'skipped': skipped, 'shortfalls': shortfalls}

def run_cases(args, out, plan):
    from .dspy_backend import DSPyAgent, configure_backend
    from .paired import PairedEvaluator, PairedExecutionError
    model_name = configure_backend(args.backend, args.model)
    model = DSPyAgent(max_calls=args.per_case_calls, critic_mode='react', max_react_iters=args.max_react_iters,
                      max_tool_calls=args.max_tool_calls, tool_output_chars=args.tool_output_chars,
                      retrieval_policy=args.retrieval_policy, max_citation_repairs=args.max_citation_repairs,
                      memory_mode=args.memory_mode, session_chunk_size=args.session_chunk_size,
                      stated_preferences=args.session_evidence == 'stated').stages
    # One evaluator per process: cases sharing a repository's frozen history reuse its memory.
    evaluator = PairedEvaluator(model)
    names = plan['cases'][:args.limit] if args.limit else plan['cases']
    for name in names:
        path = out / name.replace('case-', 'result-')
        if path.exists():
            saved = json.loads(path.read_text(encoding='utf-8'))
            if saved.get('batch', {}).get('model') not in {None, model_name}:
                raise SystemExit(f"{path.name} was run with {saved['batch']['model']}; use a separate --output-dir per model")
            if saved.get('status') in DONE: continue
        if model.calls >= args.max_total_calls:
            print(f'Stopping: total call budget {args.max_total_calls} reached'); break
        case = json.loads((out / name).read_text(encoding='utf-8'))
        history = [Event(**e) for e in case['history']]
        before, start = model.calls, time.time()
        model.max_calls = min(before + args.per_case_calls, args.max_total_calls)
        def save(result):
            result['batch'] = {'case': name, 'model': model_name, 'memory_mode': args.memory_mode, 'session_evidence': args.session_evidence,
                               'stage_calls': model.calls - before,
                               'seconds': round(time.time() - start, 1)}
            save_result(result, path)
        try:
            result = evaluator.run(history=history, **case['target'], on_update=save)
        except PairedExecutionError as error:
            result = error.partial_result
        save(result)
        print(f"{name}: {result['status']} calls={model.calls - before} {time.time() - start:.0f}s", flush=True)

def summarize(out, plan, bootstrap=1000, seed=0):
    from .evaluate import binary_metrics
    rows = []
    for name in plan['cases']:
        case = json.loads((out / name).read_text(encoding='utf-8'))
        path = out / name.replace('case-', 'result-')
        result = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {'status': 'not_run'}
        row = {'case': name, 'repo': case['metadata']['repo_id'], 'label': case['expected']['label'],
               't1': case['expected']['t1'], 'status': result['status'],
               'stage_calls': result.get('batch', {}).get('stage_calls'), 'seconds': result.get('batch', {}).get('seconds')}
        for arm in ('no_memory', 'agent'):
            prediction = (result.get(arm) or {}).get('prediction')
            if not prediction: continue
            p = prediction['pushback_probability']
            row.update({f'{arm}_p': p, f'{arm}_brier': (p - row['t1']) ** 2, f'{arm}_pred': int(p >= .5),
                        f'{arm}_tools': [t['tool'] for t in prediction.get('tool_trace', [])],
                        f'{arm}_cited': prediction.get('evidence_ids', []),
                        f'{arm}_first_pass': prediction.get('first_pass_success')})
        memory = (result.get('agent') or {}).get('memory', [])
        row['active_rules'] = sum(r['status'] == 'active' for r in memory)
        row['candidate_rules'] = sum(r['status'] == 'candidate' for r in memory)
        rows.append(row)
    pairs = [r for r in rows if r['status'] == 'ok']
    summary = {'cases': len(rows), 'complete_pairs': len(pairs),
               'status_counts': {s: sum(r['status'] == s for r in rows) for s in sorted({r['status'] for r in rows})},
               'repositories': len({r['repo'] for r in rows})}
    if pairs:
        summary['metrics'] = {arm: binary_metrics([r['t1'] for r in pairs], [r[f'{arm}_p'] for r in pairs])
                              for arm in ('no_memory', 'agent')}
        diff = np.array([r['no_memory_brier'] - r['agent_brier'] for r in pairs])
        repos = sorted({r['repo'] for r in pairs})
        blocks = [np.array([i for i, r in enumerate(pairs) if r['repo'] == repo]) for repo in repos]
        rng = np.random.default_rng(seed)
        draws = [diff[np.concatenate([blocks[i] for i in rng.integers(0, len(blocks), len(blocks))])].mean()
                 for _ in range(bootstrap)]
        summary['paired_brier'] = {'no_memory_minus_agent': float(diff.mean()), 'positive_favors': 'agent',
                                   'repository_bootstrap_ci95': [float(x) for x in np.quantile(draws, [.025, .975])]}
        summary['agent_diagnostics'] = {
            'first_pass_success': sum(bool(r.get('agent_first_pass')) for r in pairs),
            'cases_with_active_rules': sum(r['active_rules'] > 0 for r in pairs),
            'cases_citing_history': sum(bool(r.get('agent_cited')) for r in pairs),
            'mean_get_episode_reads': float(np.mean([r['agent_tools'].count('get_episode') for r in pairs]))}
    calls = [r['stage_calls'] for r in rows if r['stage_calls'] is not None]
    summary['total_stage_calls'] = int(sum(calls))
    (out / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    fields = ['case', 'repo', 'label', 't1', 'status', 'no_memory_p', 'agent_p', 'no_memory_brier', 'agent_brier',
              'no_memory_pred', 'agent_pred', 'active_rules', 'candidate_rules', 'agent_tools', 'agent_cited',
              'agent_first_pass', 'stage_calls', 'seconds']
    with open(out / 'cases.csv', 'w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fields, extrasaction='ignore')
        writer.writeheader(); writer.writerows(rows)
    return summary, rows

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, default=Path(__file__).resolve().parents[2] / 'data')
    parser.add_argument('--split', choices=['train', 'holdout', 'test'], default='holdout')
    parser.add_argument('--k', type=int, choices=[1, 2, 4, 8], default=4)
    parser.add_argument('--repos', type=int, default=5)
    parser.add_argument('--per-label', type=int, default=2, help='Cases per repository for each binary label')
    parser.add_argument('--seed', type=int, default=329)
    parser.add_argument('--history-per-session', type=int, default=2)
    parser.add_argument('--history-seed', type=int, default=329)
    parser.add_argument('--history-context-chars', type=int, default=1500)
    parser.add_argument('--history-agent-chars', type=int, default=2500)
    parser.add_argument('--history-feedback-chars', type=int, default=1000)
    parser.add_argument('--target-context-chars', type=int, default=8000)
    parser.add_argument('--target-response-chars', type=int, default=12000)
    parser.add_argument('--english-only', action=argparse.BooleanOptionalAction, default=True,
                        help='Keep English feedback in mostly-English repositories (needs data/target_flags.parquet)')
    parser.add_argument('--exclude-user-actions', action=argparse.BooleanOptionalAction, default=True,
                        help='Drop targets whose agent activity contains a user tool rejection or interruption')
    parser.add_argument('--prepare-only', action='store_true', help='Write cases and plan; no model calls')
    parser.add_argument('--summary-only', action='store_true', help='Rebuild summary from saved results')
    parser.add_argument('--limit', type=int, help='Run only the first N planned cases')
    parser.add_argument('--backend', choices=['openrouter', 'nvidia', 'dspy'], default='openrouter',
                        help='dspy means direct Gemini')
    parser.add_argument('--model', help='Model ID; otherwise OPENROUTER_MODEL, NVIDIA_MODEL or GEMINI_MODEL')
    parser.add_argument('--env-file', type=Path)
    parser.add_argument('--memory-mode', choices=['event', 'session'], default='event',
                        help='session: a few calls per history session; event: one call per pushback and rule-event pair')
    parser.add_argument('--session-chunk-size', type=int, default=20, help='Turns per session-memory call')
    parser.add_argument('--session-evidence', choices=['stated', 'pushback'], default='stated',
                        help='stated: explicit statements and rejected tool calls in any turn count; pushback: pushback turns only')
    parser.add_argument('--per-case-calls', type=int, default=60)
    parser.add_argument('--max-total-calls', type=int, default=1500, help='Cost stop across the whole run')
    parser.add_argument('--retrieval-policy', choices=['autonomous', 'search_first'], default='autonomous')
    parser.add_argument('--max-citation-repairs', type=int, choices=[0, 1], default=1)
    parser.add_argument('--max-react-iters', type=int, default=5)
    parser.add_argument('--max-tool-calls', type=int, default=4)
    parser.add_argument('--tool-output-chars', type=int, default=12000)
    args = parser.parse_args()
    out = args.output_dir
    selection = {key: getattr(args, key) for key in ['split', 'k', 'repos', 'per_label', 'seed', 'history_per_session',
                 'history_seed', 'history_context_chars', 'history_agent_chars', 'history_feedback_chars',
                 'target_context_chars', 'target_response_chars', 'english_only', 'exclude_user_actions']}
    plan_path = out / 'plan.json'
    if plan_path.exists():
        plan = json.loads(plan_path.read_text(encoding='utf-8'))
        # Externally prepared case sets (relabel_cases.py) fix their own selection.
        if plan.get('kind') != 'relabel' and plan['selection'] != selection:
            parser.error('Selection settings differ from the existing plan.json; use a fresh --output-dir')
    else:
        if args.summary_only: parser.error('No plan.json in --output-dir')
        out.mkdir(parents=True, exist_ok=True)
        if any(out.iterdir()): parser.error('Choose a fresh or previously planned --output-dir')
        plan = {'selection': selection, **prepare(args, out)}
        save_result(plan, plan_path)
        print(f"Prepared {len(plan['cases'])} cases from {len(plan['repositories'])} repositories; "
              f"skipped {len(plan['skipped'])}")
    if not (args.prepare_only or args.summary_only):
        if args.env_file is not None:
            if not args.env_file.is_file(): parser.error('--env-file must point to an existing local file')
            from dotenv import load_dotenv
            load_dotenv(args.env_file, override=False, interpolate=False)
        run_cases(args, out, plan)
    summary, _ = summarize(out, plan)
    print(json.dumps(summary, indent=2))

if __name__ == '__main__': main()
