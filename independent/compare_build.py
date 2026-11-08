"""Post-recording comparison only. Run after committing check.py's reports.

This intentionally imports the build's implementation; check.py does not.
"""
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from swarm_audit import stats
from swarm_audit.audit import AuditTable
from swarm_audit.prediction import ActionsChain, SignalChain, exact_law
from swarm_audit.topology import make

from check import AUDIT, OUT, SEEDS, simulate, read_table


def main():
    recorded = json.loads((OUT/'part1_results.json').read_text())
    scored = json.loads((OUT/'part2_arm4b.json').read_text())
    gate = json.loads(Path('results/gates/G5_cheap_talk.json').read_text())['result']['per_run']
    g = make('cycle', 4)
    signal = SignalChain(AuditTable.load(AUDIT/'cheap_talk_d2_enumerated_signal.json').lookup(), 2, True)
    actions = ActionsChain(AuditTable.load(AUDIT/'cheap_talk_d2_enumerated_actions.json').lookup(), 2)
    laws = {'signal': exact_law(signal, g, 20)['y'], 'actions': exact_law(actions, g, 20)['y']}
    s = stats.crps(stats.cdf_from_law(laws['signal']), np.array(recorded['y']))
    a = stats.crps(stats.cdf_from_law(laws['actions']), np.array(recorded['y']))
    bs = stats.bootstrap_mean(a-s, 2000, np.random.default_rng(SEEDS['bootstrap']))
    comparison = dict(
        independent_precomparison_commit='1a7fb83dfebc7b991d9eed4faf6260dd97a6594b',
        gate_gap_abs_diff=abs(gate['gap']-recorded['gap']),
        gate_mean_abs_diff=abs(gate['exact_mean_y']-recorded['exact_mean']),
        gate_p_abs_diff=abs(gate['p_value']-recorded['p_value']),
        signal_pmf_max_abs_diff=float(np.max(np.abs(laws['signal']-recorded['pmf']))),
        actions_pmf_max_abs_diff=float(np.max(np.abs(laws['actions']-scored['actions_pmf']))),
        signal_crps_max_abs_diff=float(np.max(np.abs(s-scored['signal_scores']))),
        actions_crps_max_abs_diff=float(np.max(np.abs(a-scored['actions_scores']))),
        build_crps=dict(signal=float(s.mean()), actions=float(a.mean()), paired=bs),
        build_published_crps_output_available=False,
        core_runs_available=False,
    )
    assert comparison['gate_gap_abs_diff'] < 1e-12
    assert comparison['gate_mean_abs_diff'] < 1e-12
    assert comparison['gate_p_abs_diff'] < .002
    for name in ['signal', 'actions']:
        assert comparison[f'{name}_pmf_max_abs_diff'] < 1e-12
        assert comparison[f'{name}_crps_max_abs_diff'] < 1e-12
    assert abs(bs['mean']-scored['mean_difference']) < 1e-12
    assert np.max(np.abs(np.array([bs['lo'], bs['hi']])-scored['ci95'])) < 1e-12
    # Exercise the future n=10 simulation path against the build on the available
    # calibration tables. This is a sampler validation, not a core-data result.
    sampler_checks = {}
    for name, chain, enabled in [('signal', signal, True), ('actions', actions, False)]:
        path = AUDIT/f'cheap_talk_d2_enumerated_{name}.json'
        law = simulate(read_table(path), enabled, 10, 20000, SEEDS['core_'+name])
        build_draws = chain.simulate(make('cycle', 10), 20, 20000, np.random.default_rng(103 if enabled else 104))
        build_y = stats.high_count(build_draws)
        difference = float(np.max(np.abs(law.cumsum()-stats.cdf_from_samples(build_y, 50))))
        # Two independent samples: a union of 99.5% one-sample DKW bands.
        bound = float(2*np.sqrt(np.log(400)/(2*20000)))
        assert difference < bound
        sampler_checks[name] = dict(cdf_gap=difference, combined_dkw99=bound)
    comparison['n10_calibration_table_sampler_checks'] = sampler_checks
    (OUT/'comparison.json').write_text(json.dumps(comparison, indent=2)+'\n')
    (OUT/'comparison.md').write_text(f'''# Comparison after independent results were committed

The independent Part 1 and Part 2 reports were committed locally as `1a7fb83dfebc7b991d9eed4faf6260dd97a6594b` before opening any pipeline code, gate output, or analysis output. The later GitHub upload may give that snapshot a different commit ID; its file content is unchanged. `check.py` remains independent. `compare_build.py` is explicitly a subsequent comparison and imports the build's code.

## Calibration verdict: AGREE

- Gate gap difference: {comparison['gate_gap_abs_diff']:.3g}.
- Exact mean difference: {comparison['gate_mean_abs_diff']:.3g}.
- Independent p = {recorded['p_value']:.12f}; published gate p = {gate['p_value']:.12f}; absolute difference {comparison['gate_p_abs_diff']:.12f}, within Monte Carlo error.
- Independent null 99th percentile = {recorded['null_q99']:.12f}; gate = {gate['threshold_q99']:.12f}. These are separate finite Monte Carlo samples of a discrete statistic; the p-value determines the stated gate rule.
- The full signal-aware and actions-only PMFs also match the build's recomputed exact laws to {max(comparison['signal_pmf_max_abs_diff'], comparison['actions_pmf_max_abs_diff']):.3g}.

The gate **fails** in both implementations. Log/table, state wiring, and draw checks show no discrepancy. This confirms the calculation; it does not demonstrate that chance is the cause. A correct null can produce rare failures, and this one remains a failed prespecified test.

## Arm 4b CRPS verdict: AGREE with recomputed build functions

There is no published arm 4b CRPS analysis output in the checked source snapshot; its calibration analysis file contains G5 tests only. After committing our results, we recomputed both exact laws using `swarm_audit.prediction.exact_law`, then used the build's CRPS and bootstrap functions with the same bootstrap seed. Both per-run score arrays, the paired mean and the interval agree to 1e-12. This comparison is a recomputation, not a claim of agreement with a nonexistent published output.

Signal-aware mean: {s.mean():.12f}; actions-only mean: {a.mean():.12f}; paired difference: {bs['mean']:.12f}; 95% interval: [{bs['lo']:.12f}, {bs['hi']:.12f}]. The interval spans zero, so these 50 calibration runs do not show a clear CRPS benefit from retaining messages.

## Core comparison: pending data

No `results/runs/core_cheap_talk/cheap_talk_n10_cycle_*/run_*.jsonl.gz` files or fitted core table pair were found in any fetched branch (main, core, build-handoff, hand-off-review, extra-arms, independent-check). No core score is reported.

The independent simulator supports the ten-agent cycle. Its n=10 path was checked against the build's sampler using the available calibration tables: {sampler_checks}. These checks are not core experiment results.

When the fitted tables and logs arrive, verify their names and run:

```sh
OPENBLAS_NUM_THREADS=1 python independent/check.py \\
  --core results/runs/core_cheap_talk/cheap_talk_n10_cycle_FULL_DIRECTORY \\
  --signal results/audit/llama-3.1-8b/cheap_talk_d2_signal.json \\
  --actions VERIFIED_DERIVED_ACTIONS_TABLE_PATH
```

Each predictor uses 20,000 simulated runs of its last-round chain and denominator 50. Commit `part2_core.md` and `part2_core.json` before comparing with the build's numbers (target agreement about 0.001). No future monitoring or external team message has been configured.
''')
    print(json.dumps(comparison, indent=2))


if __name__ == '__main__':
    main()
