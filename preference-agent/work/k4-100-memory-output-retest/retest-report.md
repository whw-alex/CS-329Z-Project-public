# Memory output truncation fix and targeted retest

Only the five failed history sets were rebuilt. No forecasts were rerun and the original full evaluation was not overwritten.

## Implementation

- Only session_extractor, rule_merger and session_verifier have max_tokens=8192. Global LM/forecast stays at 2048; applicability selection is unchanged.
- Extraction prompt requests at most four reusable preferences per batch. Evidence lists have at most two quotes, each at most 240 characters; rule text, scope, verification reasons and quotes have bounded lengths. One-off tasks need not be emitted.
- On AdapterParseError, recovery requires a NEW response with trusted finish_reason=length metadata. It splits a splittable input list in half once, counts both calls against the same budget and never recursively retries the children. Non-truncation errors and stale truncation metadata do not trigger recovery.
- 130 offline tests passed, including stage isolation, finish metadata shapes, input coverage after splitting, bounded recursion and shared budgets.

## API retest

| History set | Affected cases | Result | Logical memory calls | Request max_tokens |
|---|---|---|---:|---:|
| fogodev/ars-ui | [78, 85] | ok | 13 | [8192] |
| PJensen/JSHack | [31, 49, 70] | ok | 4 | [8192] |
| timelabs-npo/rhea-project | [83] | ok | 12 | [8192] |
| five2seven/stackmap | [32, 75] | ok | 2 | [8192] |
| ebridges/wakil | [52] | failed | 2 | [8192] |

Four of five builds recovered, supplying memory for eight of the nine previously affected targets. Total memory logical calls: 33. All requests used 8192, while the base LM setting stayed 2048. None of the five builds triggered split recovery; the four successes used the larger budget and bounded output schema/prompt directly.
The remaining ebridges/wakil failure is AdapterParseError without a truncation warning or trusted new length metadata; it was deliberately not retried. The saved diagnostic does not identify the exact invalid field, so do not assume it is still an output-limit issue.
This is a reliability retest, not an updated prediction evaluation. The nine original missing predictions remain missing until separately forecast. The correctness of generated preferences and their prediction value were not measured here.

Project retest directory: preference-agent/work/k4-100-memory-output-retest

No GitHub push was performed.