# Scoped memory: merged results after recovery

This is a recovery merge, not a new full evaluation. Keep the original 84 successful scoped forecasts unchanged; forecast only 8 cases with rebuilt memory. All 8 recovery forecasts succeeded, with 5 correct. Case-0052 remains memory_failed. Original full and memory-only retest directories are unchanged.

## Overall accuracy

- Scoped valid-output accuracy: 57/92 = 61.96%.
- Scoped correct valid outputs over all attempted targets, counting the remaining failure as unsuccessful: 57/93 = 61.29%.
- Old Agent: 56/93 = 60.22%.
- Coverage: scoped 92/93; old 93/93.

## Same 92 valid targets

| Variant | Correct | Accuracy | Precision | Recall | F1 | AUROC | AUPRC | Brier (lower is better) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| old | 55/92 | 0.598 | 0.207 | 0.300 | 0.245 | 0.461 | 0.214 | 0.283 |
| scoped | 57/92 | 0.620 | 0.200 | 0.250 | 0.222 | 0.457 | 0.256 | 0.252 |

Accuracy and Brier are better in this merge, but pushback recall/F1 are lower. Do not treat the small correct-count difference as a demonstrated overall improvement.
Recovered case-0031 is the only positive among the 8 recovered cases; it was missed. The 5 correct recovered predictions are all negatives.

Cases with any injected preference rules after recovery: 0.

## Provenance

The 84 original predictions use the original scoped memory build settings. The 8 recovered predictions use the memory-output fix (8192-token construction budget and bounded output schema/prompt); forecast settings remain unchanged. This mixed recovery result should not be reported as a homogeneous fresh run.
Old reference predictions were not rerun. No new memory extraction was performed in this recovery step.
Original missing outcomes are in initial-failed/. Restored memories are in recovered-memory/. Merged scoped outputs are in scoped/.
No extra retries were made for the still-failed ebridges/wakil history. No GitHub push.

Project directory: preference-agent/work/k4-100-scoped-memory-recovered