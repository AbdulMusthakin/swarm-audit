"""Verify the hosted-audit results from the published aggregate records."""

from collections import Counter
from itertools import accumulate
from math import isclose
from pathlib import Path
import json


ROOT = Path(__file__).parent


def read(name):
    return json.loads((ROOT / name).read_text())


def verify_adaption():
    preflight = read('adaption-preflight-summary.json')
    assert preflight['valid'] == preflight['attempts'] == preflight['first_pass_valid'] == 96
    data = read('adaption-balanced-counts.json')
    assert (data['attempts'], data['valid'], data['rate_limited']) == (743, 587, 156)
    assert len(data['states']) == 116
    draws, high = Counter(), Counter()
    for state in data['states']:
        assert sum(state['counts'].values()) == 4
        if state['talking'] and state['history']:
            signal = tuple(state['history'][0]['neighbour_signals'])
            if signal in {(2, 0, 0), (0, 2, 0)}:
                draws[signal] += 4
                high[signal] += sum(n for label, n in state['counts'].items() if label.startswith('H'))
    assert draws == {(2, 0, 0): 72, (0, 2, 0): 72}
    assert high == {(2, 0, 0): 9, (0, 2, 0): 5}


def verify_openrouter():
    data = read('openrouter-matched-counts.json')
    status = {(row['phase'], row['accepted']): row['count'] for row in data['statuses']}
    assert status == {('audit', True): 3712, ('counts', True): 3072, ('counts', False): 25}
    games = read('openrouter-counts-games.json')
    assert len(games) == 96
    visits = Counter()
    for game in games:
        assert len(game) == 8
        for j, round_bids in enumerate(game):
            assert len(round_bids) == 4
            for agent, (bid, signal) in enumerate(round_bids):
                assert bid in ('H', 'L') and signal is None
                state = []
                if j:
                    prior = game[j - 1]
                    neighbours = ((agent - 1) % 4, (agent + 1) % 4)
                    state.append({'bid': prior[agent][0],
                                  'neighbour_high': sum(prior[k][0] == 'H' for k in neighbours)})
                visits[json.dumps(state, sort_keys=True)] += 1
    calibration = read('openrouter-calibration.json')
    totals = [sum(bid == 'H' for step in game[-3:] for bid, _ in step) for game in games]
    histogram = [totals.count(i) for i in range(13)]
    assert histogram == calibration['observed_histogram']
    assert isclose(sum(totals) / (96 * 12), calibration['observed_mean'])
    frozen = read('openrouter-frozen-forecasts.json')['counts']['pmf']
    gap = max(abs(a - b) for a, b in zip(accumulate(frozen), accumulate(n / 96 for n in histogram)))
    assert isclose(gap, calibration['forecasts']['counts']['cdf_sup'])
    assert not calibration['passed'] and gap > calibration['cdf_sup_reference_99']
    matched = {(row['phase'], json.dumps(row['history'], sort_keys=True)): row for row in data['states']}
    assert len(matched) == 14
    for state, n in visits.items():
        assert matched['counts', state]['valid'] == n
        assert matched['counts', state]['prompt_fingerprints'] == matched['audit', state]['prompt_fingerprints']
    dominant = max(visits, key=visits.get)
    assert json.loads(dominant) == [{'bid': 'L', 'neighbour_high': 0}]
    diagnostics = read('openrouter-diagnostics.json')
    for phase, key in (('audit', 'dominant_history_audit_p_high'), ('counts', 'dominant_history_live_p_high')):
        row = matched[phase, dominant]
        assert isclose(row['sum_p_high'] / row['sum_p_valid'], diagnostics[key])
    print(f'OpenRouter counts calibration failed as recorded. CDF gap {gap:.3f}.')


if __name__ == '__main__':
    verify_adaption()
    verify_openrouter()
    print('Verified hosted-audit evidence from the public tables.')
