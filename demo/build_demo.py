"""Build demo/swechat_demo.html from the k4-100 evaluation run.

    python demo/build_demo.py
    python demo/build_demo.py --eval-dir preference-agent/work/k4-100-eval --cases 72 13 59 30 42 10

Everything on the page is computed from files, so re-running after the data or results change updates the page:
  * Overview / data scale   data/sessions.parquet, data/conv_turns.parquet, data/targets.parquet,
                            data/english-targets/targets_english.parquet
  * Overview / results      <eval-dir>/baselines-summary.json, summary.json (paired bootstrap), plan.json
  * Relabeling numbers      <relabel-dir>/labels.csv (SWE-chat label vs model relabel);
                            <human-pilot-dir>/labels.csv (30-episode human-verified pilot)
  * Demo cases              <eval-dir>/case-NNNN.json, result-NNNN.json, baselines-cases.csv, joined to
                            <relabel-dir>/labels.csv + episodes.jsonl by session_id + feedback_turn_number
Case numbers are the evaluation's case-NNNN numbers (not the `n` column of labels.csv).
"""
import argparse, csv, json, re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / 'preference-agent/work'
ARMS = [('b2', 'LLM, no history', 'B2'), ('b3', 'BM25 history + LLM', 'B3'),
        ('no_memory', 'ReAct, no memory', 'ReAct'), ('agent', 'ReAct + preference memory', 'Ours')]


def clip(s, n):
    s = re.sub(r'\s+', ' ', str(s).replace('\\n', ' ')).strip()
    return s if len(s) <= n else s[:n - 1].rstrip() + '…'


def turns(s):
    """Parse a stored turn list; fall back to a regex when the stored JSON was truncated."""
    try:
        return json.loads(s)
    except Exception:
        out = []
        pat = r'"turn_type": "(\w+)", "content": "((?:[^"\\]|\\.)*)(?:"(?:, "tool_name": "([^"]*)")?)?'
        for m in re.finditer(pat, s):
            try:
                c = json.loads('"' + m.group(2) + '"')
            except Exception:
                c = m.group(2)
            out.append({'turn_type': m.group(1), 'content': c, **({'tool_name': m.group(3)} if m.group(3) else {})})
        return out or [{'turn_type': 'user_prompt', 'content': s}]


def is_command(t):
    return str(t['content']).lstrip().startswith('<command')


def last_text(ts, kind):
    for t in reversed(ts):
        if t['turn_type'] == kind and not is_command(t):
            return t['content']
    return ''


def conversation(ctx, resp):
    """Developer/agent text that led up to the target, then what the agent did in response."""
    talk = [t for t in ctx if t['turn_type'] in ('user_prompt', 'assistant_response') and not is_command(t)]
    shown = [{'t': 'user' if t['turn_type'] == 'user_prompt' else 'agent', 'text': clip(t['content'], 520)}
             for t in talk[-3:]]
    tools = Counter(t.get('tool_name') or 'tool' for t in resp if t['turn_type'] == 'tool_use')
    agent_text = [t['content'] for t in resp if t['turn_type'] == 'assistant_response' and str(t['content']).strip()]
    return {
        'ctx': shown, 'ctxOmitted': max(0, len(talk) - len(shown)),
        'tools': tools.most_common(), 'nTools': sum(tools.values()),
        'first': clip(agent_text[0], 260) if len(agent_text) > 1 else '',
        'final': clip(agent_text[-1], 900) if agent_text else '',
    }


def data_scale(data):
    """Dataset numbers for the Overview; any missing file just drops its numbers."""
    import pandas as pd
    out = {}
    try:
        s = pd.read_parquet(data / 'sessions.parquet')
        out.update(sessions=len(s), repos=int(s.repo_id.nunique()), developers=int(s.user_id.nunique()),
                   first=str(s.created_at.min())[:7], last=str(s.created_at.max())[:7])
    except Exception as e:
        print('  sessions.parquet:', e)
    try:
        c = pd.read_parquet(data / 'conv_turns.parquet', columns=['turn_type'])
        vc = c.turn_type.value_counts()
        out.update(turns=len(c), prompts=int(vc.get('user_prompt', 0)))
    except Exception as e:
        print('  conv_turns.parquet:', e)
    for key, f in (('targets', 'targets.parquet'), ('english', 'english-targets/targets_english.parquet')):
        try:
            t = pd.read_parquet(data / f, columns=['repo_id'])
            out[key] = {'n': len(t), 'repos': int(t.repo_id.nunique())}
        except Exception as e:
            print(f'  {f}:', e)
    return out


def relabel_stats(relabel):
    """SWE-chat original vs model relabel on the sampled targets and their history turns."""
    def one(path):
        rows = [r for r in csv.DictReader(open(path)) if r['status'] == 'ok' and r['model_binary'] in ('P', 'N')]
        n = len(rows)
        return {'n': n,
                'orig_p': sum(r['swechat_binary'] == 'P' for r in rows) / n,
                'new_p': sum(r['model_binary'] == 'P' for r in rows) / n,
                'agree': sum(r['swechat_binary'] == r['model_binary'] for r in rows) / n}
    out = {}
    for key, rel in (('targets', 'labels.csv'), ('history', 'history/labels.csv')):
        try:
            out[key] = one(relabel / rel)
        except Exception as e:
            print(f'  relabel {rel}:', e)
    return out


def human_check(pilot):
    """Agreement with the human-verified pilot (draft LLM labels, written before the prompt rules were fitted on it)."""
    allrows = [r for r in csv.DictReader(open(pilot / 'labels.csv')) if r['human_label']]
    rows = [r for r in allrows if r['human_label'] in ('P', 'N')]
    return {'total': len(allrows), 'n': len(rows),
            'llm': sum(r['my_binary'] == r['human_label'] for r in rows),
            'swechat': sum(r['swechat_binary'] == r['human_label'] for r in rows)}


def results(eval_dir):
    b = json.load(open(eval_dir / 'baselines-summary.json'))
    s = json.load(open(eval_dir / 'summary.json'))
    m = b['metrics_per_arm']
    ref = m['b2']
    prior = ref['pushback_rate']
    out = {'arms': [], 'cases': b['cases'], 'repos': b['repositories'],
           'positives': round(ref['pushback_rate'] * ref['samples']), 'samples': ref['samples'],
           'prior': prior, 'majorityAcc': 1 - prior, 'priorBrier': prior * (1 - prior)}
    for key, name, short in ARMS:
        x = m[key]
        out['arms'].append({'key': key, 'name': name, 'short': short, 'n': x['samples'], 'brier': x['brier'],
                            'auroc': x['auroc'], 'meanPred': x['mean_prediction'], 'acc': x['accuracy_at_0_5'], 'f1': x['f1_at_0_5'], 'auprc': x['auprc']})
    pb = s.get('paired_brier', {})
    out['agentVsNoMem'] = {'diff': pb.get('no_memory_minus_agent'), 'ci': pb.get('repository_bootstrap_ci95'),
                           'pairs': s.get('complete_pairs')}
    p3 = b.get('common_cases', {}).get('paired_brier', {}).get('b3_minus_agent')
    if p3:
        out['agentVsB3'] = {'diff': p3['mean'], 'ci': p3['repository_bootstrap_ci95'], 'pairs': p3['cases']}
    try:
        out['model'] = json.load(open(eval_dir / 'run-manifest.json')).get('model')
    except Exception:
        pass
    return out


def build_case(eval_dir, n, labels, replies, probs):
    c = json.load(open(eval_dir / f'case-{n:04d}.json'))
    r = json.load(open(eval_dir / f'result-{n:04d}.json'))
    md = c['metadata']
    key = (md['session_id'], str(md['feedback_turn_number']))
    lab = labels.get(key, {})
    reply = replies.get(key, '')
    hist = [{'id': h['id'], 'label': h['label'],
             'task': clip(last_text(turns(h['context']), 'user_prompt'), 170),
             'agent': clip(last_text(turns(h['agent_turn']), 'assistant_response'), 150),
             'feedback': clip(h['follow_up'], 170)} for h in c['history']]
    rules = [{'text': m['text'], 'status': m['status'], 'conf': round(m['confidence'], 2),
              'supports': m['independent_supports'],
              'quote': next((v['feedback_excerpt'] for v in m['verification'] if v['verdict'] == 'support'), '')}
             for m in r['agent']['memory']]
    pred = r['agent']['prediction']
    return {
        'n': n, 'repo': md['repo_id'], 'k': md['k'], 'nHist': len(hist), 'history': hist,
        **conversation(turns(c['target']['context']), turns(c['target']['response'])),
        'p': probs[n],
        'agent': {'why': pred['rationale'], 'evidence': pred['evidence_ids'],
                  'trace': [{'tool': t['tool'], 'args': t['arguments'], 'ev': t['evidence_ids']} for t in pred['tool_trace']]},
        'rules': rules,
        'actual': {'push': c['expected']['t1'] == 1, 'text': clip(reply, 600), 'truncated': len(reply) > 600,
                   'fine': lab.get('model_fine_label', ''), 'orig': lab.get('swechat_label', ''),
                   'origPush': md['relabel']['swechat_t1'] == 1 if 'relabel' in md else None,
                   'conf': lab.get('confidence', ''), 'reason': lab.get('reasoning', ''),
                   'quote': lab.get('evidence_quote', '')},
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--eval-dir', default=str(WORK / 'k4-100-eval'))
    ap.add_argument('--relabel-dir', default=str(WORK / 'k4-100'))
    ap.add_argument('--human-pilot-dir', default=str(WORK / 'relabel-pilot'))
    ap.add_argument('--data-dir', default=str(ROOT / 'data'))
    ap.add_argument('--cases', type=int, nargs='+', default=[72, 13, 59, 30, 42, 10])
    ap.add_argument('--output', default=str(ROOT / 'demo/swechat_demo.html'))
    a = ap.parse_args()
    eval_dir, relabel = Path(a.eval_dir), Path(a.relabel_dir)

    labels = {(r['session_id'], r['turn_number']): r for r in csv.DictReader(open(relabel / 'labels.csv'))}
    replies = {}
    for line in open(relabel / 'episodes.jsonl'):
        e = json.loads(line)
        replies[(e['session_id'], str(e['turn_number']))] = e.get('reply', '')
    probs = {}
    for r in csv.DictReader(open(eval_dir / 'baselines-cases.csv')):
        num = int(re.search(r'\d+', r['case']).group())
        probs[num] = {'b2': r['b2_p'], 'b3': r['b3_p'], 'no_memory': r['no_memory_p'], 'agent': r['agent_p']}
        probs[num] = {k: (float(v) if v not in ('', None) else None) for k, v in probs[num].items()}

    overview = {'data': data_scale(Path(a.data_dir)), 'relabel': relabel_stats(relabel), 'human': human_check(Path(a.human_pilot_dir)), 'results': results(eval_dir)}
    cases = [build_case(eval_dir, n, labels, replies, probs) for n in a.cases]

    tpl = (Path(__file__).parent / 'template.html').read_text()
    html = (tpl.replace('/*OVERVIEW*/', json.dumps(overview, ensure_ascii=False))
               .replace('/*CASES*/', json.dumps(cases, ensure_ascii=False).replace('</', '<\\/')))
    Path(a.output).write_text(html)
    print(f'wrote {a.output}: {len(cases)} cases')
    for c in cases:
        x = c['actual']
        print(f"  case {c['n']:>2}  {c['repo']:<30} actual={'pushback' if x['push'] else 'no_pushback':<12} "
              f"fine={x['fine']:<15} orig={x['orig']:<14} message={'yes' if x['text'] else 'MISSING'}  p={c['p']}")


if __name__ == '__main__':
    main()
