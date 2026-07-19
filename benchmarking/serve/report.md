# Query server benchmark

- scale factor: 1.0
- warm iterations per query: 5 (median reported)
- warm-up iterations (excluded from stats): 1
- generated: 2026-07-19 14:40:03 IST

## Timings (p50 seconds, warm)

| query | daft-inprocess | daft-serve | duckdb-inprocess | duckdb-quack |
|---|---|---|---|---|
| q1 | 1.408 | 1.399 | 0.075 | 0.093 |
| q2 | 0.222 | 0.234 | 0.030 | 0.046 |
| q3 | 0.683 | 0.682 | 0.057 | 0.074 |
| q4 | 0.443 | 0.445 | 0.038 | 0.053 |
| q5 | 0.589 | 0.595 | 0.059 | 0.073 |
| q6 | 0.512 | 0.506 | 0.024 | 0.041 |
| q7 | 0.699 | 0.703 | 0.062 | 0.077 |
| q8 | 0.680 | 0.682 | 0.066 | 0.083 |
| q9 | 0.984 | 1.015 | 0.102 | 0.116 |
| q10 | 1.007 | 1.016 | 0.093 | 0.109 |
| q11 | 0.267 | 0.271 | 0.017 | 0.032 |
| q12 | 0.969 | 0.968 | 0.042 | 0.056 |
| q13 | 0.953 | 0.956 | 0.184 | 0.199 |
| q14 | 0.435 | 0.431 | 0.040 | 0.055 |
| q15 | 0.524 | 0.532 | 0.027 | 0.043 |
| q16 | 0.330 | 0.341 | 0.035 | 0.050 |
| q17 | 0.669 | 0.661 | 0.043 | 0.060 |
| q18 | 1.401 | 1.430 | 0.079 | 0.099 |
| q19 | 1.335 | 1.349 | 0.057 | 0.071 |
| q20 | 0.704 | 0.731 | 0.046 | 0.061 |
| q21 | 2.939 | 2.961 | 0.127 | 0.138 |
| q22 | 0.201 | 0.206 | 0.030 | 0.046 |

## Cold start (first warm-up iteration, seconds)

| query | daft-inprocess | daft-serve | duckdb-inprocess | duckdb-quack |
|---|---|---|---|---|
| q1 | 1.668 | 1.396 | 0.083 | 0.091 |
| q2 | 0.226 | 0.242 | 0.035 | 0.047 |
| q3 | 0.684 | 0.682 | 0.065 | 0.077 |
| q4 | 0.461 | 0.458 | 0.039 | 0.056 |
| q5 | 0.586 | 0.620 | 0.056 | 0.075 |
| q6 | 0.502 | 0.497 | 0.027 | 0.040 |
| q7 | 0.713 | 0.709 | 0.060 | 0.074 |
| q8 | 0.676 | 0.698 | 0.067 | 0.085 |
| q9 | 0.993 | 1.019 | 0.101 | 0.124 |
| q10 | 1.014 | 1.012 | 0.099 | 0.112 |
| q11 | 0.280 | 0.274 | 0.020 | 0.032 |
| q12 | 0.981 | 1.017 | 0.044 | 0.056 |
| q13 | 0.962 | 0.979 | 0.188 | 0.203 |
| q14 | 0.436 | 0.436 | 0.038 | 0.055 |
| q15 | 0.515 | 0.524 | 0.027 | 0.043 |
| q16 | 0.343 | 0.345 | 0.037 | 0.053 |
| q17 | 0.685 | 0.677 | 0.043 | 0.060 |
| q18 | 1.417 | 1.374 | 0.085 | 0.100 |
| q19 | 1.310 | 1.351 | 0.056 | 0.073 |
| q20 | 0.712 | 0.711 | 0.047 | 0.060 |
| q21 | 3.090 | 2.972 | 0.135 | 0.132 |
| q22 | 0.210 | 0.196 | 0.032 | 0.046 |

## Correctness gate

- daft-inprocess-vs-daft-serve/q1: PASS
- daft-inprocess-vs-daft-serve/q10: PASS
- daft-inprocess-vs-daft-serve/q11: PASS
- daft-inprocess-vs-daft-serve/q12: PASS
- daft-inprocess-vs-daft-serve/q13: PASS
- daft-inprocess-vs-daft-serve/q14: PASS
- daft-inprocess-vs-daft-serve/q15: PASS
- daft-inprocess-vs-daft-serve/q16: PASS
- daft-inprocess-vs-daft-serve/q17: PASS
- daft-inprocess-vs-daft-serve/q18: PASS
- daft-inprocess-vs-daft-serve/q19: PASS
- daft-inprocess-vs-daft-serve/q2: PASS
- daft-inprocess-vs-daft-serve/q20: PASS
- daft-inprocess-vs-daft-serve/q21: PASS
- daft-inprocess-vs-daft-serve/q22: PASS
- daft-inprocess-vs-daft-serve/q3: PASS
- daft-inprocess-vs-daft-serve/q4: PASS
- daft-inprocess-vs-daft-serve/q5: PASS
- daft-inprocess-vs-daft-serve/q6: PASS
- daft-inprocess-vs-daft-serve/q7: PASS
- daft-inprocess-vs-daft-serve/q8: PASS
- daft-inprocess-vs-daft-serve/q9: PASS
- daft-inprocess-vs-duckdb-inprocess/q1: PASS
- daft-inprocess-vs-duckdb-inprocess/q10: PASS
- daft-inprocess-vs-duckdb-inprocess/q11: PASS
- daft-inprocess-vs-duckdb-inprocess/q12: PASS
- daft-inprocess-vs-duckdb-inprocess/q13: PASS
- daft-inprocess-vs-duckdb-inprocess/q14: PASS
- daft-inprocess-vs-duckdb-inprocess/q15: PASS
- daft-inprocess-vs-duckdb-inprocess/q16: PASS
- daft-inprocess-vs-duckdb-inprocess/q17: PASS
- daft-inprocess-vs-duckdb-inprocess/q18: PASS
- daft-inprocess-vs-duckdb-inprocess/q19: PASS
- daft-inprocess-vs-duckdb-inprocess/q2: PASS
- daft-inprocess-vs-duckdb-inprocess/q20: PASS
- daft-inprocess-vs-duckdb-inprocess/q21: PASS
- daft-inprocess-vs-duckdb-inprocess/q22: PASS
- daft-inprocess-vs-duckdb-inprocess/q3: PASS
- daft-inprocess-vs-duckdb-inprocess/q4: PASS
- daft-inprocess-vs-duckdb-inprocess/q5: PASS
- daft-inprocess-vs-duckdb-inprocess/q6: PASS
- daft-inprocess-vs-duckdb-inprocess/q7: PASS
- daft-inprocess-vs-duckdb-inprocess/q8: PASS
- daft-inprocess-vs-duckdb-inprocess/q9: PASS
- daft-inprocess-vs-duckdb-quack/q1: PASS
- daft-inprocess-vs-duckdb-quack/q10: PASS
- daft-inprocess-vs-duckdb-quack/q11: PASS
- daft-inprocess-vs-duckdb-quack/q12: PASS
- daft-inprocess-vs-duckdb-quack/q13: PASS
- daft-inprocess-vs-duckdb-quack/q14: PASS
- daft-inprocess-vs-duckdb-quack/q15: PASS
- daft-inprocess-vs-duckdb-quack/q16: PASS
- daft-inprocess-vs-duckdb-quack/q17: PASS
- daft-inprocess-vs-duckdb-quack/q18: PASS
- daft-inprocess-vs-duckdb-quack/q19: PASS
- daft-inprocess-vs-duckdb-quack/q2: PASS
- daft-inprocess-vs-duckdb-quack/q20: PASS
- daft-inprocess-vs-duckdb-quack/q21: PASS
- daft-inprocess-vs-duckdb-quack/q22: PASS
- daft-inprocess-vs-duckdb-quack/q3: PASS
- daft-inprocess-vs-duckdb-quack/q4: PASS
- daft-inprocess-vs-duckdb-quack/q5: PASS
- daft-inprocess-vs-duckdb-quack/q6: PASS
- daft-inprocess-vs-duckdb-quack/q7: PASS
- daft-inprocess-vs-duckdb-quack/q8: PASS
- daft-inprocess-vs-duckdb-quack/q9: PASS

## Addendum: validation runs (2026-07-19)

### Native A/B (parent 76932f9ab vs HEAD, daft-inprocess, warm p50)

No regression: median delta +0.3%, max +3.5% (within noise); several queries
faster on HEAD (q2 −15%, q9 −7.7%) from gating per-query notification
payloads on attached subscribers.

### Row-group splitting (`enable_scan_task_row_group_splitting=True`)

Flag-off is the recorded baseline above (default; code path unchanged when
off). Flag-on, both engine lanes, all 22 queries correctness-PASS:

| lane | median | best | worst |
|---|---|---|---|
| daft-inprocess | 1.08x | 1.43x (q19) | 0.98x |
| daft-serve | 1.07x | 1.44x (q19) | 0.97x |

Scan-bound queries gain most: q6 1.40x, q19 1.43x, q14 1.31x, q20 1.21x,
q7 1.18x, q3 1.17x.
