Scripted policies, cycle, T = 20, 200 runs per condition; predictors simulated 20,000 times. ε(p_test) is pooled over states with p_test's weights. Δ is the largest binned TV over rounds between the swarm and the predictor's fixed reference law; the floor in brackets is the simultaneous null: the 95th percentile of that maximum for R whole runs drawn from a separate model pool against the same law (review C4).

| condition | expected | ε_test actions | ε_test signal | max Δ actions (floor) | max Δ signal (floor) | bound B_T | exact law, n = 4 (gap / q99, MC p) | G5 path, n = 4 | result |
|---|---|---|---|---|---|---|---|---|---|
| coin/counts/n4 | ε = 0; Δ inside the noise floor | 0.0000 | – | 0.095 (0.122) | – | 0.00 | 0.043 / 0.098 (p = 0.464) ✓ | ✓ | PASS |
| coin/counts/n10 | ε = 0; Δ inside the noise floor | 0.0000 | – | 0.081 (0.118) | – | 0.00 | – | – | PASS |
| coin/cheap_talk/n4 | ε = 0; Δ inside the noise floor | 0.0000 | 0.0000 | 0.100 (0.122) | 0.100 (0.123) | 0.00 | 0.052 / 0.098 (p = 0.351) ✓ | ✓ | PASS |
| coin/cheap_talk/n10 | ε = 0; Δ inside the noise floor | 0.0000 | 0.0000 | 0.089 (0.118) | 0.086 (0.116) | 0.00 | – | – | PASS |
| copy/counts/n4 | actions-only ε = 0; Δ inside the noise floor (full history) | 0.0000 | – | 0.109 (0.124) | – | 0.00 | 0.062 / 0.105 (p = 0.243) ✓ | ✓ | PASS |
| copy/counts/n10 | actions-only ε = 0; Δ inside the noise floor (full history) | 0.0000 | – | 0.118 (0.123) | – | 0.00 | – | – | PASS |
| copy/cheap_talk/n4 | actions-only ε = 0; Δ inside the noise floor (full history) | 0.0000 | 0.0000 | 0.106 (0.124) | 0.106 (0.124) | 0.00 | 0.058 / 0.105 (p = 0.285) ✓ | ✓ | PASS |
| copy/cheap_talk/n10 | actions-only ε = 0; Δ inside the noise floor (full history) | 0.0000 | 0.0000 | 0.076 (0.123) | 0.081 (0.120) | 0.00 | – | – | PASS |
| two_round/counts/n4 | actions-only ε > 0; bound above the measured Δ | 0.0769 | – | 0.148 (0.122) | – | 6.54 | 0.124 / 0.101 (p = 0.001) ✗ | ✓ | PASS |
| two_round/counts/n10 | actions-only ε > 0; bound above the measured Δ | 0.0769 | – | 0.100 (0.119) | – | 15.75 | – | – | PASS |
| hidden_signal/cheap_talk/n4 | actions-only ε > 0; signal-aware ε = 0; signal-aware Δ inside the floor | 0.2815 | 0.0000 | 0.211 (0.123) | 0.095 (0.124) | 21.82 | 0.069 / 0.104 (p = 0.143) ✓ | ✓ | PASS |
| hidden_signal/cheap_talk/n10 | actions-only ε > 0; signal-aware ε = 0; signal-aware Δ inside the floor | 0.2815 | 0.0000 | 0.280 (0.120) | 0.071 (0.117) | 54.28 | – | – | PASS |

Paired CRPS difference D = CRPS(actions-only) − CRPS(signal-aware) (> 0 favours keeping the signals), 95% whole-run bootstrap:

| condition | D | 95% interval |
|---|---|---|
| coin/cheap_talk/n4 | -0.0000 | [-0.0000, 0.0000] |
| coin/cheap_talk/n10 | 0.0000 | [-0.0001, 0.0001] |
| copy/cheap_talk/n4 | 0.0000 | [-0.0000, 0.0000] |
| copy/cheap_talk/n10 | -0.0000 | [-0.0000, 0.0000] |
| hidden_signal/cheap_talk/n4 | 0.0123 | [0.0043, 0.0203] |
| hidden_signal/cheap_talk/n10 | 0.0280 | [0.0191, 0.0367] |
