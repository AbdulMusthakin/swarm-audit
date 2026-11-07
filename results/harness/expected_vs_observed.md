Scripted policies, cycle, T = 20, 200 runs per condition; predictors simulated 20,000 times. ε(p_test) is pooled over states with p_test's weights. Δ is the largest binned TV over rounds; the floor is the 95th percentile of the same maximum between two model samples of the swarm's size.

| condition | expected | ε_test actions | ε_test signal | max Δ actions (floor) | max Δ signal (floor) | bound B_T | exact law, n = 4 (gap / q99) | G5 path, n = 4 | result |
|---|---|---|---|---|---|---|---|---|---|
| coin/counts/n4 | ε = 0; Δ inside the noise floor | 0.0000 | – | 0.095 (0.165) | – | 0.00 | 0.043 / 0.098 ✓ | ✓ | PASS |
| coin/counts/n10 | ε = 0; Δ inside the noise floor | 0.0000 | – | 0.090 (0.165) | – | 0.00 | – | – | PASS |
| coin/cheap_talk/n4 | ε = 0; Δ inside the noise floor | 0.0000 | 0.0000 | 0.100 (0.165) | 0.100 (0.170) | 0.00 | 0.052 / 0.098 ✓ | ✓ | PASS |
| coin/cheap_talk/n10 | ε = 0; Δ inside the noise floor | 0.0000 | 0.0000 | 0.090 (0.165) | 0.089 (0.165) | 0.00 | – | – | PASS |
| copy/counts/n4 | actions-only ε = 0; Δ inside the noise floor (full history) | 0.0000 | – | 0.109 (0.175) | – | 0.00 | 0.062 / 0.105 ✓ | ✓ | PASS |
| copy/counts/n10 | actions-only ε = 0; Δ inside the noise floor (full history) | 0.0000 | – | 0.118 (0.170) | – | 0.00 | – | – | PASS |
| copy/cheap_talk/n4 | actions-only ε = 0; Δ inside the noise floor (full history) | 0.0000 | 0.0000 | 0.106 (0.175) | 0.106 (0.175) | 0.00 | 0.058 / 0.105 ✓ | ✓ | PASS |
| copy/cheap_talk/n10 | actions-only ε = 0; Δ inside the noise floor (full history) | 0.0000 | 0.0000 | 0.079 (0.170) | 0.080 (0.170) | 0.00 | – | – | PASS |
| two_round/counts/n4 | actions-only ε > 0; bound above the measured Δ | 0.0769 | – | 0.148 (0.175) | – | 6.54 | 0.124 / 0.101 ✗ | ✓ | PASS |
| two_round/counts/n10 | actions-only ε > 0; bound above the measured Δ | 0.0769 | – | 0.100 (0.165) | – | 15.75 | – | – | PASS |
| hidden_signal/cheap_talk/n4 | actions-only ε > 0; signal-aware ε = 0; signal-aware Δ inside the floor | 0.2815 | 0.0000 | 0.211 (0.170) | 0.095 (0.175) | 21.82 | 0.069 / 0.104 ✓ | ✓ | PASS |
| hidden_signal/cheap_talk/n10 | actions-only ε > 0; signal-aware ε = 0; signal-aware Δ inside the floor | 0.2815 | 0.0000 | 0.282 (0.170) | 0.067 (0.165) | 54.28 | – | – | PASS |

Paired CRPS difference D = CRPS(actions-only) − CRPS(signal-aware) (> 0 favours keeping the signals), 95% whole-run bootstrap:

| condition | D | 95% interval |
|---|---|---|
| coin/cheap_talk/n4 | -0.0000 | [-0.0000, 0.0000] |
| coin/cheap_talk/n10 | 0.0000 | [-0.0001, 0.0001] |
| copy/cheap_talk/n4 | 0.0000 | [-0.0000, 0.0000] |
| copy/cheap_talk/n10 | 0.0000 | [-0.0001, 0.0001] |
| hidden_signal/cheap_talk/n4 | 0.0123 | [0.0039, 0.0207] |
| hidden_signal/cheap_talk/n10 | 0.0286 | [0.0195, 0.0377] |
