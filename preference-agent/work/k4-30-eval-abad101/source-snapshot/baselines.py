"""LLM baselines on prepared batch cases.

B2  no-history LLM: one forecast call that sees only the current request context and agent activity.
B3  BM25 + LLM: the same call, plus the top-k BM25-retrieved historical episodes of the same repository
    (before target_order) placed directly in the prompt. No rule extraction, no tools.

Both arms use one signature and one validator, so the only difference is the retrieved history.
Cases are the case-*.json files written by `python -m preference_agent.batch --prepare-only`;
`expected` labels are read only for scoring. Paid calls happen only when --summary-only is absent.
"""
import argparse
import csv
from collections import Counter
from dataclasses import dataclass
import json
import math
import re
import time
from pathlib import Path

import dspy

from .core import Event, FeedbackMiner, binary_label, save_result
from .pushback_definition import forecast_instructions
from .sampling import cap_text

ARMS = ('b2', 'b3')
DONE = {'ok', 'failed'}
TOKEN = re.compile(r'\w+')

def tokenize(text):
    return TOKEN.findall(text.casefold())

class BM25:
    """Okapi BM25 over a fixed document list; pure Python, deterministic tie-break by input order."""
    def __init__(self, documents, k1=1.5, b=0.75):
        if k1 < 0 or not 0 <= b <= 1: raise ValueError('Invalid BM25 parameters')
        self.k1, self.b = k1, b
        self.docs = [Counter(tokenize(d)) for d in documents]
        self.lengths = [sum(d.values()) for d in self.docs]
        self.avgdl = sum(self.lengths) / len(self.docs) if self.docs else 0.
        df = Counter(term for d in self.docs for term in d)
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + .5) / (f + .5)) for t, f in df.items()}

    def scores(self, query):
        terms = Counter(tokenize(query))
        out = []
        for doc, length in zip(self.docs, self.lengths):
            norm = self.k1 * (1 - self.b + self.b * length / self.avgdl) if self.avgdl else self.k1
            out.append(sum(qf * self.idf[t] * doc[t] * (self.k1 + 1) / (doc[t] + norm)
                           for t, qf in terms.items() if t in doc))
        return out

    def top(self, query, k):
        scored = [(s, i) for i, s in enumerate(self.scores(query)) if s > 0]
        return sorted(scored, key=lambda x: (-x[0], x[1]))[:k]

def document(event):
    return '\n'.join([event.context, event.agent_turn, event.follow_up, *event.user_actions])

@dataclass
class RetrievalConfig:
    top_k: int = 5
    query_context_chars: int = 4000
    query_response_chars: int = 4000
    request_chars: int = 1500
    agent_chars: int = 2500
    feedback_chars: int = 1000
    k1: float = 1.5
    b: float = 0.75

def retrieve(history, repo, target_order, context, response, config):
    """Scoped BM25: same repository, strictly before target_order. Returns prompt episodes and a trace."""
    past = FeedbackMiner().run([e for e in history if e.repo == repo and e.order < target_order])
    query = context[-config.query_context_chars:] + '\n' + response[-config.query_response_chars:]
    hits = BM25([document(e) for e in past], config.k1, config.b).top(query, config.top_k) if past else []
    # Chronological in the prompt; ranks and scores stay in the trace.
    chosen = sorted(hits, key=lambda h: (past[h[1]].order, past[h[1]].id))
    # Binary history labels, as in the agent's tools (history keeps SWE-chat labels).
    episodes = [{'event_id': past[i].id, 'label': binary_label(past[i].label),
                 'request': cap_text(past[i].context, config.request_chars),
                 'agent_activity': cap_text(past[i].agent_turn, config.agent_chars),
                 'user_actions': list(past[i].user_actions),
                 'follow_up': cap_text(past[i].follow_up, config.feedback_chars)} for _, i in chosen]
    trace = {'history_events': len(past), 'retrieved': len(hits),
             'ranking': [{'event_id': past[i].id, 'rank': r + 1, 'score': round(s, 4)}
                         for r, (s, i) in enumerate(hits)]}
    return episodes, trace

class DirectForecast(dspy.Signature):
    """Forecast whether the developer's NEXT message AFTER the supplied agent_turn will be pushback.
    That future message is hidden. context contains only earlier requests; do not classify the visible
    current user request. Absence of feedback in the input is NOT evidence against pushback. An assistant
    claim of success is not verified success.
    past_episodes are earlier interactions from the same repository, each with the developer's actual
    follow_up and its label; the list may be empty. Use them only as evidence about this developer and
    repository. Give a probability under uncertainty.
    evidence_ids may contain only event_id values from past_episodes that support your forecast."""
    context: str = dspy.InputField()
    agent_turn: str = dspy.InputField()
    past_episodes: list[dict] = dspy.InputField()
    pushback_probability: float = dspy.OutputField(desc='Probability in [0, 1] that the next message is pushback')
    rationale: str = dspy.OutputField(desc='Brief evidence-based explanation')
    evidence_ids: list[str] = dspy.OutputField()

# Same pushback standard as the agent's Forecast: the relabeling rules in prompts/relabel_pushback.md.
DirectForecast = DirectForecast.with_instructions(forecast_instructions(DirectForecast.instructions))

def validate_probability(value):
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError('Expected a pushback probability in [0, 1]')
    return float(value)

class BaselineForecaster(dspy.Module):
    """B2/B3 forecaster with a logical call budget shared across all cases of a run."""
    def __init__(self, max_calls=100, retrieval=None):
        super().__init__()
        from .dspy_backend import POLICY
        self.forecaster = dspy.Predict(DirectForecast.with_instructions(POLICY + DirectForecast.instructions))
        self.retrieval = retrieval or RetrievalConfig()
        self.max_calls, self.calls = max_calls, 0

    def forward(self, arm, history, repo, target_order, context, response):
        if arm not in ARMS: raise ValueError(f'Unknown baseline arm {arm}')
        if arm == 'b3':
            episodes, trace = retrieve(history, repo, target_order, context, response, self.retrieval)
        else:
            episodes, trace = [], {'history_events': 0, 'retrieved': 0, 'ranking': []}
        if self.calls >= self.max_calls: raise RuntimeError('Baseline call budget exhausted')
        self.calls += 1
        result = self.forecaster(context=context, agent_turn=response, past_episodes=episodes)
        allowed = {e['event_id'] for e in episodes}
        evidence = result.evidence_ids
        if not isinstance(evidence, list) or any(not isinstance(e, str) or e not in allowed for e in evidence):
            raise ValueError('Forecast cited an event that was not in past_episodes')
        if not isinstance(result.rationale, str) or not result.rationale.strip():
            raise ValueError('Forecast returned an empty rationale')
        return {'pushback_probability': validate_probability(result.pushback_probability),
                'rationale': result.rationale, 'evidence_ids': evidence, 'calibrated': False,
                'arm': arm, 'retrieval': trace, 'prompt_episode_ids': sorted(allowed)}

def run_case(model, arm, case):
    """Returns a result record; validation failures are recorded, other errors propagate."""
    history = [Event(**e) for e in case['history']] if arm == 'b3' else []
    try:
        return {'status': 'ok', 'prediction': model(arm, history, **case['target'])}
    except (ValueError, TypeError, KeyError) as error:
        return {'status': 'failed', 'error': {'type': type(error).__name__, 'message': str(error)[:500]}}

def case_names(out):
    plan = out / 'plan.json'
    if plan.exists(): return json.loads(plan.read_text(encoding='utf-8'))['cases']
    return sorted(p.name for p in out.glob('case-*.json'))

def result_path(out, arm, name):
    return out / name.replace('case-', f'baseline-{arm}-')

def settings(args):
    return {'retrieval': vars(retrieval_config(args)), 'model': args.model, 'backend': args.backend}

def retrieval_config(args):
    return RetrievalConfig(top_k=args.top_k, query_context_chars=args.query_context_chars,
                           query_response_chars=args.query_response_chars, request_chars=args.request_chars,
                           agent_chars=args.agent_chars, feedback_chars=args.feedback_chars, k1=args.k1, b=args.b)

def run(args, out, names):
    from .dspy_backend import configure_backend
    model_name = configure_backend(args.backend, args.model)
    model = BaselineForecaster(max_calls=args.max_total_calls, retrieval=retrieval_config(args))
    config = {**settings(args), 'model': model_name}
    for name in names[:args.limit] if args.limit else names:
        case = json.loads((out / name).read_text(encoding='utf-8'))
        for arm in args.arms:
            path = result_path(out, arm, name)
            if path.exists():
                saved = json.loads(path.read_text(encoding='utf-8'))
                if saved.get('settings', {}).get('model') not in {None, model_name}:
                    raise SystemExit(f"{path.name} was run with {saved['settings']['model']}; use a separate --cases-dir per model")
                if saved.get('status') in DONE:
                    if arm == 'b3' and saved.get('settings', {}).get('retrieval') != config['retrieval']:
                        raise SystemExit(f'{path.name} used different retrieval settings; use them or a fresh copy')
                    continue
            if model.calls >= model.max_calls:
                print(f'Stopping: total call budget {model.max_calls} reached'); return
            start = time.time()
            result = run_case(model, arm, case)
            result.update(case=name, arm=arm, settings=config, seconds=round(time.time() - start, 1))
            save_result(result, path)
            print(f"{name} {arm}: {result['status']} {result['seconds']}s", flush=True)

def summarize(out, names, bootstrap=1000, seed=0):
    import numpy as np
    from .evaluate import binary_metrics
    arms = ['b2', 'b3', 'no_memory', 'agent']
    rows = []
    for name in names:
        case = json.loads((out / name).read_text(encoding='utf-8'))
        row = {'case': name, 'repo': case['metadata'].get('repo_id', case['target']['repo']),
               'label': case['expected']['label'], 't1': case['expected']['t1'],
               'history_events': len(case.get('history', []))}
        sources = {arm: result_path(out, arm, name) for arm in ARMS}
        paired = out / name.replace('case-', 'result-')
        paired = json.loads(paired.read_text(encoding='utf-8')) if paired.exists() else {}
        for arm in arms:
            if arm in ARMS:
                path = sources[arm]
                saved = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {'status': 'not_run'}
                prediction = saved.get('prediction') if saved['status'] == 'ok' else None
                row[f'{arm}_status'] = saved['status']
            else:
                prediction = (paired.get(arm) or {}).get('prediction')
                row[f'{arm}_status'] = 'ok' if prediction else ('not_run' if not paired else 'failed')
            if prediction:
                # Agent/no_memory results give 1 - P(no_pushback); baselines give the probability directly.
                p = prediction['pushback_probability']
                row.update({f'{arm}_p': p, f'{arm}_brier': (p - row['t1']) ** 2, f'{arm}_pred': int(p >= .5)})
                if arm == 'b3':
                    row['b3_retrieved'] = prediction['retrieval']['retrieved']
                    row['b3_cited'] = prediction.get('evidence_ids', [])
        rows.append(row)

    def score(subset, arm):
        return binary_metrics([r['t1'] for r in subset], [r[f'{arm}_p'] for r in subset])

    rng = np.random.default_rng(seed)
    def paired_brier(subset, a, b):
        diff = np.array([r[f'{a}_brier'] - r[f'{b}_brier'] for r in subset])
        repos = sorted({r['repo'] for r in subset})
        blocks = [np.array([i for i, r in enumerate(subset) if r['repo'] == repo]) for repo in repos]
        draws = [diff[np.concatenate([blocks[i] for i in rng.integers(0, len(blocks), len(blocks))])].mean()
                 for _ in range(bootstrap)]
        return {'mean': float(diff.mean()), 'positive_favors': b, 'cases': len(subset),
                'repository_bootstrap_ci95': [float(x) for x in np.quantile(draws, [.025, .975])]}

    present = [arm for arm in arms if any(r[f'{arm}_status'] != 'not_run' for r in rows)]
    summary = {'cases': len(rows), 'repositories': len({r['repo'] for r in rows}),
               'status_counts': {arm: dict(Counter(r[f'{arm}_status'] for r in rows)) for arm in present},
               'metrics_per_arm': {}, 'common_cases': {}}
    for arm in present:
        ok = [r for r in rows if r[f'{arm}_status'] == 'ok']
        if ok: summary['metrics_per_arm'][arm] = score(ok, arm)
    # Fair comparison: only cases where every present arm succeeded.
    common = [r for r in rows if all(r[f'{arm}_status'] == 'ok' for arm in present)]
    if common:
        summary['common_cases'] = {'cases': len(common), 'arms': present,
                                   'metrics': {arm: score(common, arm) for arm in present},
                                   'paired_brier': {}}
        comparisons = [('b2', 'b3'), ('b3', 'agent'), ('b2', 'agent'), ('no_memory', 'b2')]
        for a, b in comparisons:
            if a in present and b in present:
                summary['common_cases']['paired_brier'][f'{a}_minus_{b}'] = paired_brier(common, a, b)
        # Cases with no history cannot benefit from memory or retrieval; report the rest separately too.
        with_history = [r for r in common if r['history_events'] > 0]
        summary['common_cases']['no_history_cases'] = [r['case'] for r in common if r['history_events'] == 0]
        if with_history:
            summary['common_cases_with_history'] = {
                'cases': len(with_history), 'metrics': {arm: score(with_history, arm) for arm in present},
                'paired_brier': {f'{a}_minus_{b}': paired_brier(with_history, a, b) for a, b in comparisons
                                 if a in present and b in present}}
    b3 = [r for r in rows if r.get('b3_status') == 'ok']
    if b3:
        summary['b3_diagnostics'] = {'mean_retrieved': float(np.mean([r['b3_retrieved'] for r in b3])),
                                     'cases_citing_history': sum(bool(r['b3_cited']) for r in b3)}
    (out / 'baselines-summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    fields = ['case', 'repo', 'label', 't1', 'history_events'] + [f'{arm}_{key}' for arm in arms for key in ('status', 'p', 'brier', 'pred')] \
        + ['b3_retrieved', 'b3_cited']
    with open(out / 'baselines-cases.csv', 'w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fields, extrasaction='ignore')
        writer.writeheader(); writer.writerows(rows)
    return summary, rows

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--cases-dir', type=Path, required=True, help='Batch directory with case-*.json (and plan.json)')
    parser.add_argument('--arms', nargs='+', choices=ARMS, default=list(ARMS))
    parser.add_argument('--summary-only', action='store_true', help='Rebuild summary from saved results; no model calls')
    parser.add_argument('--limit', type=int, help='Run only the first N cases')
    parser.add_argument('--backend', choices=['openrouter', 'nvidia', 'dspy'], default='openrouter',
                        help='dspy means direct Gemini')
    parser.add_argument('--model', help='Model ID; otherwise OPENROUTER_MODEL, NVIDIA_MODEL or GEMINI_MODEL')
    parser.add_argument('--env-file', type=Path)
    parser.add_argument('--max-total-calls', type=int, default=200, help='Logical forecast calls across the run')
    parser.add_argument('--top-k', type=int, default=5, help='B3: episodes placed in the prompt')
    parser.add_argument('--query-context-chars', type=int, default=4000, help='B3: tail of target context used as query')
    parser.add_argument('--query-response-chars', type=int, default=4000, help='B3: tail of agent activity used as query')
    parser.add_argument('--request-chars', type=int, default=1500, help='B3: per-episode request cap in the prompt')
    parser.add_argument('--agent-chars', type=int, default=2500, help='B3: per-episode agent activity cap')
    parser.add_argument('--feedback-chars', type=int, default=1000, help='B3: per-episode follow-up cap')
    parser.add_argument('--k1', type=float, default=1.5)
    parser.add_argument('--b', type=float, default=.75)
    parser.add_argument('--bootstrap', type=int, default=1000)
    args = parser.parse_args()
    out = args.cases_dir
    if not out.is_dir(): parser.error('--cases-dir must be an existing directory')
    if args.top_k < 1: parser.error('--top-k must be positive')
    names = case_names(out)
    if not names: parser.error('No case-*.json files; run python -m preference_agent.batch --prepare-only first')
    if not args.summary_only:
        if args.env_file is not None:
            if not args.env_file.is_file(): parser.error('--env-file must point to an existing local file')
            from dotenv import load_dotenv
            load_dotenv(args.env_file, override=False, interpolate=False)
        run(args, out, names)
    summary, _ = summarize(out, names, bootstrap=args.bootstrap)
    print(json.dumps(summary, indent=2))

if __name__ == '__main__': main()
