"""Offline binary B0/B1 evaluation using EDA splits. Never calls a language model.

B0: training-target pushback rate. B1: the same repository's first-k labelled history pushback rate,
falling back to B0 when that history has no labelled turns. Labels are SWE-chat t1 (pushback vs not).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss, f1_score,
                             log_loss, roc_auc_score)

SWECHAT_LABELS = ['correction', 'rejection', 'failure_report', 'non_pushback']


def binary_metrics(y, p):
    """Binary pushback metrics; threshold 0.5 is fixed, not tuned."""
    y, p = np.asarray(y, dtype=int), np.asarray(p, dtype=float)
    both = len(np.unique(y)) == 2
    return {'samples': len(y), 'pushback_rate': float(y.mean()), 'mean_prediction': float(p.mean()),
            'auroc': float(roc_auc_score(y, p)) if both else None,
            'auprc': float(average_precision_score(y, p)) if both else None,
            'brier': float(brier_score_loss(y, p)),
            'log_loss': float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1])),
            'accuracy_at_0_5': float(accuracy_score(y, p >= .5)),
            'f1_at_0_5': float(f1_score(y, p >= .5, zero_division=0))}

def metrics(y, p):
    """Binary pushback metrics; shared with baselines.py and batch.py."""
    y, p = np.asarray(y, dtype=int), np.asarray(p, dtype=float)
    if y.shape != p.shape or not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError('Expected finite pushback probabilities in [0, 1]')
    return binary_metrics(y, p)

def prepare(targets, repo_split, k, split):
    if k not in {1, 2, 4, 8}: raise ValueError('Invalid k')
    if split not in {'holdout', 'test'}: raise ValueError('Evaluate holdout or test only')
    if repo_split.repo_id.duplicated().any() or targets.duplicated(['session_id','turn_number']).any():
        raise ValueError('Duplicate split or target key')
    mapping = repo_split.set_index('repo_id')['split']
    if not targets.repo_id.map(mapping).fillna('unassigned').eq(targets.split.fillna('unassigned')).all():
        raise ValueError('Inconsistent repository splits')
    if not targets.prompt_pushback.isin(SWECHAT_LABELS).all(): raise ValueError('Invalid label')
    if not targets.t1.eq(targets.prompt_pushback.ne('non_pushback').astype(int)).all(): raise ValueError('Inconsistent binary label')
    eligible = targets[targets.n_sess >= k + 1]
    train = eligible[(eligible.split == 'train') & (eligible['rank'] >= k)]
    evaluation = eligible[(eligible.split == split) & (eligible['rank'] >= k)].copy().reset_index(drop=True)
    if train.empty or evaluation.empty: raise ValueError('No eligible train/evaluation targets')
    prior = float(train.t1.mean())
    # Each repository's first k sessions only. Evaluation labels never enter history rates.
    history = eligible[eligible['rank'] < k]
    rates = history.groupby('repo_id').t1.mean()
    b1 = evaluation.repo_id.map(rates).to_numpy(dtype=float, copy=True)
    missing = np.isnan(b1)
    b1[missing] = prior
    b0 = np.full(len(evaluation), prior)
    evaluation['history_available'] = ~missing
    return evaluation, train, b0, b1, prior

def paired_bootstrap(frame, b0, b1, repetitions, seed):
    rng = np.random.default_rng(seed)
    blocks = list(frame.groupby('repo_id', sort=True).indices.values())
    values = {'auroc_B1_minus_B0': [], 'brier_B0_minus_B1': []}
    for _ in range(repetitions):
        ix = np.concatenate([blocks[i] for i in rng.integers(0, len(blocks), len(blocks))])
        y = frame.t1.to_numpy()[ix]
        p0, p1 = b0[ix], b1[ix]
        if len(np.unique(y)) == 2:
            values['auroc_B1_minus_B0'].append(float(roc_auc_score(y, p1) - roc_auc_score(y, p0)))
        values['brier_B0_minus_B1'].append(float(np.mean((p0 - y) ** 2) - np.mean((p1 - y) ** 2)))
    return {key: {'estimate_direction': 'positive favors B1', 'valid_replicates': len(v),
                  'ci95': [float(x) for x in np.quantile(v, [.025, .975])] if v else None}
            for key, v in values.items()}

def evaluate(data_dir, output_dir, k=1, split='holdout', bootstrap=200, seed=0):
    root, out = Path(data_dir), Path(output_dir)
    if out.exists() and any(out.iterdir()): raise FileExistsError('Use a fresh output directory')
    targets = pd.read_parquet(root/'targets.parquet')
    splits = pd.read_parquet(root/'repo_split.parquet')
    frame, train, b0, b1, prior = prepare(targets, splits, k, split)
    covered = frame.history_available.to_numpy()
    y = frame.t1.to_numpy()
    result = {'kind': 'offline_baselines', 'target': 'binary pushback (SWE-chat t1)', 'paid_api_calls': 0,
              'k': k, 'split': split, 'training_target_rows': len(train),
              'training_repositories': train.repo_id.nunique(), 'evaluation_repositories': frame.repo_id.nunique(),
              'B0_training_pushback_rate': prior, 'history_covered_rows': int(covered.sum()),
              'history_covered_repositories': frame.loc[covered, 'repo_id'].nunique(),
              'B1_fallback_rows': int((~covered).sum()),
              'B1_fallback': 'B0 rate if first-k sessions contain no labelled eligible turns',
              'metrics': {'B0': metrics(y, b0), 'B1': metrics(y, b1)},
              'covered_subset_metrics': {name: metrics(y[covered], p[covered])
                                         for name, p in [('B0', b0), ('B1', b1)]} if covered.any() else {},
              'bootstrap': paired_bootstrap(frame, b0, b1, bootstrap, seed),
              'settings': {'threshold': .5, 'threshold_tuned': False, 'bootstrap_unit': 'repository',
                           'bootstrap_repetitions': bootstrap, 'seed': seed},
              'limitations': ['No LLM/agent evaluated; no DSPy optimization performed',
                              'Uses SWE-chat labels, not relabeled targets',
                              'Uses EDA rank-based session history; raw timestamp-overlap checks are not applied',
                              'Known-user split isolation is limited by missing user IDs',
                              'B1 rates are unsmoothed; log loss clips 0/1 probabilities']}
    out.mkdir(parents=True, exist_ok=True)
    (out/'metrics.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    rows = frame[['session_id', 'turn_number', 'repo_id', 'rank', 'prompt_pushback', 't1', 'history_available']].copy()
    rows['B0_pushback_probability'], rows['B1_pushback_probability'] = b0, b1
    rows.to_parquet(out/'predictions.parquet', index=False)
    return result

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path(__file__).resolve().parents[2]/'data')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--split', choices=['holdout', 'test'], default='holdout')
    parser.add_argument('--k', type=int, choices=[1, 2, 4, 8], default=1)
    parser.add_argument('--bootstrap', type=int, default=200)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    if args.bootstrap < 1: parser.error('--bootstrap must be positive')
    r = evaluate(args.data_dir, args.output_dir, args.k, args.split, args.bootstrap, args.seed)
    print(json.dumps({key: r[key] for key in ['k', 'split', 'history_covered_rows', 'B1_fallback_rows', 'metrics', 'bootstrap']}, indent=2))

if __name__ == '__main__': main()
