# Query server benchmark

- scale factor: 1.0
- warm iterations per query: 5 (median reported)
- warm-up iterations (excluded from stats): 1
- generated: 2026-07-19 16:49:05 IST

## Timings (p50 seconds, warm)

| query | daft-inprocess | daft-serve | duckdb-inprocess | duckdb-quack |
|---|---|---|---|---|
| q1 | 0.099 | 0.100 | 0.072 | 0.094 |
| q2 | 0.036 | 0.038 | 0.030 | 0.045 |
| q3 | 0.084 | 0.084 | 0.059 | 0.073 |
| q4 | 0.041 | 0.040 | 0.039 | 0.053 |
| q5 | 0.073 | 0.072 | 0.064 | 0.077 |
| q6 | 0.039 | 0.041 | 0.025 | 0.041 |
| q7 | 0.086 | 0.087 | 0.059 | 0.077 |
| q8 | 0.093 | 0.084 | 0.068 | 0.083 |
| q9 | 0.124 | 0.120 | 0.099 | 0.117 |
| q10 | 0.131 | 0.129 | 0.095 | 0.113 |
| q11 | 0.033 | 0.034 | 0.016 | 0.031 |
| q12 | 0.102 | 0.104 | 0.039 | 0.057 |
| q13 | 0.118 | 0.117 | 0.185 | 0.198 |
| q14 | 0.050 | 0.050 | 0.042 | 0.055 |
| q15 | 0.062 | 0.064 | 0.027 | 0.045 |
| q16 | 0.041 | 0.040 | 0.035 | 0.052 |
| q17 | 0.064 | 0.061 | 0.045 | 0.061 |
| q18 | 0.166 | 0.165 | 0.082 | 0.095 |
| q19 | 0.124 | 0.126 | 0.057 | 0.072 |
| q20 | 0.079 | 0.083 | 0.045 | 0.059 |
| q21 | 0.310 | 0.318 | 0.122 | 0.135 |
| q22 | 0.024 | 0.025 | 0.029 | 0.044 |

## Cold start (first warm-up iteration, seconds)

| query | daft-inprocess | daft-serve | duckdb-inprocess | duckdb-quack |
|---|---|---|---|---|
| q1 | 0.403 | 0.110 | 0.089 | 0.092 |
| q2 | 0.045 | 0.042 | 0.035 | 0.048 |
| q3 | 0.104 | 0.085 | 0.062 | 0.075 |
| q4 | 0.044 | 0.040 | 0.039 | 0.058 |
| q5 | 0.077 | 0.075 | 0.057 | 0.085 |
| q6 | 0.043 | 0.039 | 0.026 | 0.044 |
| q7 | 0.088 | 0.093 | 0.058 | 0.075 |
| q8 | 0.088 | 0.087 | 0.068 | 0.095 |
| q9 | 0.133 | 0.131 | 0.104 | 0.117 |
| q10 | 0.142 | 0.134 | 0.097 | 0.109 |
| q11 | 0.036 | 0.035 | 0.018 | 0.033 |
| q12 | 0.102 | 0.102 | 0.043 | 0.056 |
| q13 | 0.124 | 0.116 | 0.180 | 0.201 |
| q14 | 0.050 | 0.052 | 0.037 | 0.058 |
| q15 | 0.084 | 0.075 | 0.026 | 0.045 |
| q16 | 0.041 | 0.041 | 0.036 | 0.054 |
| q17 | 0.073 | 0.066 | 0.040 | 0.057 |
| q18 | 0.170 | 0.170 | 0.085 | 0.098 |
| q19 | 0.128 | 0.126 | 0.054 | 0.070 |
| q20 | 0.084 | 0.081 | 0.046 | 0.059 |
| q21 | 0.341 | 0.307 | 0.121 | 0.139 |
| q22 | 0.026 | 0.026 | 0.031 | 0.044 |

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
