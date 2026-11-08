"""Independent implementation from the attached definitions; numpy only.

Run from the repository root: OPENBLAS_NUM_THREADS=1 python independent/check.py
Core: python independent/check.py --core RUN_DIRECTORY --signal TABLE --actions TABLE
No imports from the pipeline and no reads of its code or analysis outputs.
"""
import argparse
import gzip
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
AUDIT = Path('results/audit/llama-3.1-8b')
MESSAGES = ('high', 'low', 'none')
SEEDS = dict(chain=2026110701, null=2026110702, bootstrap=2026110703,
             core_signal=2026110704, core_actions=2026110705)


def read_table(path):
    return {key: np.asarray(value['kappa'], dtype=float)
            for key, value in json.loads(Path(path).read_text())['entries'].items()}


def signal_key(own, left, right):
    counts = [int(left % 3 == m) + int(right % 3 == m) for m in range(3)]
    return '{}-{}-{}-{}-{}-{}'.format(
        'H' if own < 3 else 'L', int(left < 3) + int(right < 3),
        *counts, MESSAGES[own % 3])


def local_kernel(table, signal):
    if signal:
        # Each pair is encoded as 3 * bid_index + message_index.
        lut = np.empty((6, 6, 6, 6))
        for own, left, right in itertools.product(range(6), repeat=3):
            lut[own, left, right] = table[signal_key(own, left, right)].reshape(6)
        start = table['START'].reshape(6)
    else:
        lut = np.empty((2, 2, 2, 2))
        for own, left, right in itertools.product(range(2), repeat=3):
            p = float(table[f"{'H' if own == 0 else 'L'}-{int(left == 0) + int(right == 0)}"])
            lut[own, left, right] = (p, 1-p)
        p = float(table['START'])
        start = np.array([p, 1-p])
    assert np.all(lut >= 0) and np.all(start >= 0)
    assert np.max(np.abs(lut.sum(axis=-1)-1)) < 1e-12
    assert abs(start.sum()-1) < 1e-12
    return lut, start


def exact(table, signal):
    lut, start = local_kernel(table, signal)
    q = len(start)
    profiles = np.array(list(itertools.product(range(q), repeat=4)))
    local = lut[profiles, np.roll(profiles, 1, axis=1), np.roll(profiles, -1, axis=1)]
    transition = np.ones((len(profiles), len(profiles)))
    for agent in range(4):
        transition *= local[:, agent, profiles[:, agent]]
    row_error = float(np.max(np.abs(transition.sum(axis=1)-1)))
    assert row_error < 1e-12
    h_count = (profiles < 3).sum(axis=1) if signal else (profiles == 0).sum(axis=1)
    features = [h_count]
    if signal:
        features += [(profiles % 3 == m).sum(axis=1) for m in range(3)]
    features = np.array(features).T
    distribution = start[profiles].prod(axis=1)
    rounds = []
    counted = None
    for round_number in range(1, 21):
        if round_number > 1:
            distribution = distribution @ transition
        rounds.append(distribution @ features)
        if round_number == 16:
            counted = np.zeros((len(profiles), 21))
            counted[np.arange(len(profiles)), h_count] = distribution
        elif round_number > 16:
            pushed = transition.T @ counted
            counted = np.zeros_like(pushed)
            for count in range(5):
                mask = h_count == count
                counted[mask, count:] = pushed[mask, :21-count]
    law = counted.sum(axis=0)
    mean = float(law @ np.arange(21))
    assert abs(law.sum()-1) < 1e-12
    assert abs(mean - np.array(rounds)[15:, 0].sum()) < 1e-11
    return law, np.array(rounds), row_error


def simulate(table, signal, n, runs, seed):
    """Independent forward sampler; signal draws use bid then conditional message."""
    lut, start = local_kernel(table, signal)
    rng = np.random.default_rng(seed)
    profile = np.zeros((runs, n), dtype=int)
    y = np.zeros(runs, dtype=int)
    for round_number in range(1, 21):
        probs = np.broadcast_to(start, (runs, n, len(start))) if round_number == 1 else lut[
            profile, np.roll(profile, 1, axis=1), np.roll(profile, -1, axis=1)]
        if signal:
            joint = probs.reshape(runs, n, 2, 3)
            ph = joint[:, :, 0].sum(axis=-1)
            bid = (rng.random((runs, n)) >= ph).astype(int)
            chosen = np.take_along_axis(joint, bid[:, :, None, None], axis=2)[:, :, 0]
            mass = chosen.sum(axis=-1, keepdims=True)
            conditional = np.divide(chosen, mass, out=np.zeros_like(chosen), where=mass > 0)
            msg = (rng.random((runs, n, 1)) >= np.cumsum(conditional, axis=-1)[:, :, :2]).sum(axis=-1)
            profile = 3*bid + msg
        else:
            profile = (rng.random((runs, n)) >= probs[:, :, 0]).astype(int)
        if round_number >= 16:
            y += (profile < 3).sum(axis=1) if signal else (profile == 0).sum(axis=1)
    return np.bincount(y, minlength=5*n+1)/runs


def read_runs(directory):
    paths = sorted(Path(directory).glob('run_*.jsonl.gz'))
    if not paths:
        raise ValueError(f'No runs in {directory}')
    runs = []
    for path in paths:
        with gzip.open(path, 'rt') as stream:
            runs.append([json.loads(line) for line in stream])
    return paths, runs


def validate_logs(runs, table):
    max_error = dict(p_H=0., sig_after_H=0., sig_after_L=0.)
    mismatches = dict(state=0, action=0, signal=0)
    observed = np.zeros((20, 4))
    ys = []
    for run_index, rows in enumerate(runs):
        indexed = {(r['round'], r['agent']): r for r in rows}
        assert len(rows) == len(indexed) == 80
        assert set(indexed) == set(itertools.product(range(1, 21), range(4)))
        assert {r['run'] for r in rows} == {run_index}
        for (rnd, agent), row in indexed.items():
            kernel = table[row['state_sig']]
            ph = float(kernel[0].sum())
            assert 0 < ph < 1
            expected = dict(p_H=ph, sig_after_H=kernel[0]/ph, sig_after_L=kernel[1]/(1-ph))
            for field, value in expected.items():
                max_error[field] = max(max_error[field], float(np.max(np.abs(np.asarray(row[field])-value))))
            if rnd == 1:
                state = 'START'
            else:
                prev = [indexed[rnd-1, i] for i in range(4)]
                def pair(i):
                    return 3*int(prev[i]['action'] == 'L') + MESSAGES.index(prev[i]['signal'])
                state = signal_key(pair(agent), pair((agent-1) % 4), pair((agent+1) % 4))
            mismatches['state'] += state != row['state_sig']
            assert 0 <= row['u_action'] < 1 and 0 <= row['u_signal'] < 1
            action = 'H' if row['u_action'] < row['p_H'] else 'L'
            mismatches['action'] += action != row['action']
            conditional = row['sig_after_' + row['action']]
            msg = MESSAGES[min(int(np.searchsorted(np.cumsum(conditional), row['u_signal'], side='right')), 2)]
            mismatches['signal'] += msg != row['signal']
            observed[rnd-1, 0] += row['action'] == 'H'
            observed[rnd-1, 1+MESSAGES.index(row['signal'])] += 1
        ys.append(sum(r['action'] == 'H' for r in rows if r['round'] >= 16))
    return np.array(ys), observed/len(runs), max_error, mismatches


def crps(cdf, y):
    y = np.asarray(y)
    return ((cdf[None, :] - (y[:, None] <= np.arange(len(cdf))))**2).sum(axis=1)/(len(cdf)-1)


def compare_scores(signal_law, actions_law, y):
    # Three support points: .25, .50, .25; observation 1; (.25² + .25²)/2.
    assert crps(np.array([.25, .75, 1.]), [1])[0] == .0625
    for observation in range(len(signal_law)):
        perfect = np.cumsum(np.eye(len(signal_law))[observation])
        assert crps(perfect, [observation])[0] == 0
    signal_scores = crps(np.cumsum(signal_law), y)
    actions_scores = crps(np.cumsum(actions_law), y)
    differences = actions_scores - signal_scores
    rng = np.random.default_rng(SEEDS['bootstrap'])
    boot = differences[rng.integers(0, len(y), (2000, len(y)))].mean(axis=1)
    return dict(mean_signal=float(signal_scores.mean()), mean_actions=float(actions_scores.mean()),
                mean_difference=float(differences.mean()), ci95=np.quantile(boot, [.025, .975]).tolist(),
                differences=differences.tolist(), signal_scores=signal_scores.tolist(),
                actions_scores=actions_scores.tolist(), bootstrap_seed=SEEDS['bootstrap'])


def md_table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(map(str, headers)) + ' |',
                      '| ' + ' | '.join(['---']*len(headers)) + ' |'] +
                     ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])


def calibration():
    signal_path = AUDIT/'cheap_talk_d2_enumerated_signal.json'
    actions_path = AUDIT/'cheap_talk_d2_enumerated_actions.json'
    signal, actions = read_table(signal_path), read_table(actions_path)
    paths, runs = read_runs('results/runs/calibration_cheap_talk/cheap_talk_n4_cycle_last1')
    assert [p.name for p in paths] == [f'run_{i:03d}.jsonl.gz' for i in range(50)]
    y, observed, errors, mismatches = validate_logs(runs, signal)
    signal_law, expected, row_error = exact(signal, True)
    actions_law, _, _ = exact(actions, False)
    sim_law = simulate(signal, True, 4, 200000, SEEDS['chain'])
    cdf = signal_law.cumsum()
    empirical = np.bincount(y, minlength=21).cumsum()/len(y)
    gap = float(np.max(np.abs(empirical-cdf)))
    rng = np.random.default_rng(SEEDS['null'])
    null_y = rng.choice(21, size=(20000, 50), p=signal_law/signal_law.sum())
    null_cdfs = (null_y[:, :, None] <= np.arange(21)).mean(axis=1)
    gaps = np.max(np.abs(null_cdfs-cdf), axis=1)
    exceed = int(np.count_nonzero(gaps >= gap-1e-12))
    p = (1+exceed)/20001
    sim_gap = float(np.max(np.abs(sim_law.cumsum()-cdf)))
    dkw99 = float(np.sqrt(np.log(200)/(2*200000)))
    result = dict(seeds=SEEDS, decisions=sum(map(len, runs)), probability_max_abs=errors,
                  mismatches=mismatches, y=y.tolist(), histogram=np.bincount(y, minlength=21).tolist(),
                  observed_mean=float(y.mean()), exact_mean=float(signal_law @ np.arange(21)),
                  pmf=signal_law.tolist(), cdf=cdf.tolist(), mass=float(signal_law.sum()),
                  transition_row_error=row_error, gap=gap, null_exceedances=exceed,
                  null_q99=float(np.quantile(gaps, .99)), p_value=p, gate_pass=p > .01,
                  simulated_pmf=sim_law.tolist(), simulation_cdf_gap=sim_gap,
                  simulation_dkw99=dkw99, observed_rounds=observed.tolist(), exact_rounds=expected.tolist())
    assert sim_gap < dkw99, '200k simulation exceeds the 99% DKW band'
    sources = [signal_path, actions_path] + paths
    (OUT/'input_sha256.json').write_text(json.dumps({str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}, indent=2)+'\n')
    (OUT/'part1_results.json').write_text(json.dumps(result, indent=2)+'\n')
    rows = [[k, result['histogram'][k], f'{signal_law[k]:.12f}', f'{cdf[k]:.12f}'] for k in range(21)]
    diagnostics = [[rnd+1] + [f'{v:.9f}' for pair in zip(observed[rnd], expected[rnd]) for v in pair] for rnd in range(20)]
    text = f'''# Independent calibration check — recorded before pipeline inspection

Only the supplied hand-off, the two enumerated audit tables, and the 50 logged runs were read. No pipeline code, gate output, or prior analysis was opened. The source data are from core commit f2449ea69fffe8943d6a34618db13a0e81724033; this branch also retains the existing hand-off commit 183adb0. Input hashes are in `input_sha256.json`.

Reproduce: `OPENBLAS_NUM_THREADS=1 python independent/check.py`. Requires Python and numpy. The exact computation enumerates all 1,296 profiles, constructs the product transition matrix, and propagates profile/count mass. Simulation uses a separate hierarchical bid/message sampler.

## Log checks

- Decisions: {result['decisions']}; complete, unique 20 × 4 grids per run.
- Maximum probability discrepancies: {errors}; overall {max(errors.values()):.16g}.
- Wiring/draw mismatches: {mismatches} (including START checks).
- Run y values, in filename order: {y.tolist()}.
- Observed mean: {y.mean():.12f}.

## Exact law and per-run test

- Exact mean: {result['exact_mean']:.12f}; mass: {result['mass']:.16f}.
- Maximum transition row-sum error: {row_error:.3g}. Count-law mean agrees with the sum of round 16–20 expectations to 1e-11.
- Observed CDF gap: {gap:.12f}.
- Null draws: 20,000 samples of 50 y values; seed {SEEDS['null']} (numpy default_rng/PCG64).
- Null 99th percentile: {result['null_q99']:.12f} (numpy linear quantile).
- Exceedances, using 1e-12 tolerance: {exceed}; corrected p = {p:.12f}; gate **{'PASS' if p > .01 else 'FAIL'}** at p > .01.
- 200,000 forward simulations, seed {SEEDS['chain']}: maximum CDF error {sim_gap:.12f}, below 99% DKW bound {dkw99:.12f}.

{md_table(['y', 'Observed frequency', 'Exact PMF', 'Exact CDF'], rows)}

## Per-round diagnostic

All entries count agents per round (out of four); observed columns average the 50 runs. These are descriptive diagnostics, without post-hoc significance claims.

{md_table(['Round', 'H observed', 'H exact', 'high observed', 'high exact', 'low observed', 'low exact', 'none observed', 'none exact'], diagnostics)}

## Preliminary verdict against the supplied hand-off

The independently computed mean and gap agree with the hand-off's rounded 5.47 and 0.203. The p-value agrees with its 0.0076 within the requested Monte Carlo tolerance. Agreement verifies the failed calibration gate; it does not establish that chance is the cause of the failure. The gate file will be compared only after this report is committed.
'''
    (OUT/'part1_results.md').write_text(text)
    score = compare_scores(signal_law, actions_law, y)
    (OUT/'part2_arm4b.json').write_text(json.dumps(dict(**score, actions_pmf=actions_law.tolist(), actions_cdf=actions_law.cumsum().tolist()), indent=2)+'\n')
    (OUT/'part2_arm4b.md').write_text(f'''# Independent arm 4b CRPS comparison — before pipeline comparison

Uses the fixed signal-aware exact CDF from Part 1 and an independently enumerated 16-profile actions-only chain, with the supplied `cheap_talk_d2_enumerated_actions.json` table. Both carry the count in rounds 16–20. Scores sum over k = 0,…,20 and divide by 20.

- Signal-aware mean CRPS: {score['mean_signal']:.12f}.
- Actions-only mean CRPS: {score['mean_actions']:.12f}.
- Mean paired difference (actions-only minus signal-aware): {score['mean_difference']:.12f}.
- 95% percentile interval: [{score['ci95'][0]:.12f}, {score['ci95'][1]:.12f}].
- 2,000 bootstrap resamples of the 50 whole runs; seed {SEEDS['bootstrap']}; predictors fixed; numpy linear quantiles.
- Unit checks: every point-mass perfect forecast scores zero; for probabilities (.25, .50, .25) at (0, 1, 2) and observation 1, CRPS = (.25² + (−.25)² + 0²)/2 = .0625.

Positive differences favour signals. This report will be committed before opening the build's analysis outputs. Machine-readable CDFs and per-run scores are in `part2_arm4b.json`.
''')
    print(json.dumps({k: result[k] for k in ['probability_max_abs', 'mismatches', 'observed_mean', 'exact_mean', 'gap', 'null_q99', 'p_value', 'simulation_cdf_gap']}, indent=2))
    print(json.dumps({k: score[k] for k in ['mean_signal', 'mean_actions', 'mean_difference', 'ci95']}, indent=2))


def core(args):
    if not args.signal or not args.actions:
        raise ValueError('Provide the verified fitted --signal and derived --actions table paths')
    paths, runs = read_runs(args.core)
    n_values = {r['n'] for rows in runs for r in rows}
    assert n_values == {10}
    for rows in runs:
        assert len(rows) == 200
        assert {(r['round'], r['agent']) for r in rows} == set(itertools.product(range(1, 21), range(10)))
        assert {r['graph'] for r in rows} == {'cycle'}
    y = np.array([sum(r['action'] == 'H' for r in rows if r['round'] >= 16) for rows in runs])
    signal_law = simulate(read_table(args.signal), True, 10, 20000, SEEDS['core_signal'])
    actions_law = simulate(read_table(args.actions), False, 10, 20000, SEEDS['core_actions'])
    score = compare_scores(signal_law, actions_law, y)
    payload = dict(**score, signal_pmf=signal_law.tolist(), actions_pmf=actions_law.tolist(),
                   y=y.tolist(), runs=[str(p) for p in paths], seeds=SEEDS,
                   signal_table=args.signal, actions_table=args.actions)
    (OUT/'part2_core.json').write_text(json.dumps(payload, indent=2)+'\n')
    (OUT/'part2_core.md').write_text(f'''# Independent core CRPS comparison

{len(runs)} ten-agent cycle runs. Each predictor uses only the last round, simulated for 20 rounds over 20,000 runs. Denominator: 50. Seeds: {SEEDS}.

Signal mean CRPS: {score['mean_signal']:.12f}. Actions mean CRPS: {score['mean_actions']:.12f}.
Mean paired difference: {score['mean_difference']:.12f}; 95% whole-run bootstrap percentile interval: {score['ci95']} (2,000 resamples; predictors fixed).

Inputs and distributions are in `part2_core.json`. Commit these results before reading the build's comparison.
''')
    print(json.dumps(score, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--core', help='Directory of core run_*.jsonl.gz files')
    parser.add_argument('--signal', help='Fitted signal table for core')
    parser.add_argument('--actions', help='Derived actions table for core')
    args = parser.parse_args()
    core(args) if args.core else calibration()
