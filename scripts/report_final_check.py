#!/usr/bin/env python3
"""Independently check the scores hand-off's new numbers from committed inputs.

Imports no experiment or report-generation code. Reconstructs message spreads,
paired score estimates and bootstrap intervals from per-run scores, and all-high
survival by enumerating the 81 message profiles. Run after report_persistence.py,
report_70b.py and extra_followup_check.py. Writes a deterministic verification
record with input SHA-256 hashes; exits nonzero on any discrepancy.

    OPENBLAS_NUM_THREADS=1 python scripts/report_final_check.py
"""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
HASHES, CHECKS = {}, {}


def read(path):
    raw = (ROOT / path).read_bytes()
    HASHES[path] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)


def check(name, actual, expected):
    CHECKS[name] = bool(np.allclose(actual, expected, atol=1e-12, rtol=1e-12))


def interval(values, rng):
    # Draw whole games, preserving pairing; each bootstrap mean is computed
    # separately from the raw paired differences.
    means = [float(np.mean(values[rng.integers(0, len(values), len(values))]))
             for _ in range(2000)]
    return [float(np.mean(values)), *np.percentile(means, [2.5, 97.5])]


def main():
    report = read("results/followup/report_70b.json")
    rng = np.random.default_rng(20261108)
    for model, folder in (("8b", "results"), ("70b", "interest/model-70b")):
        table = read(f"{folder}/audit/llama-3.1-{model}/cheap_talk_d2_enumerated_signal.json")["entries"]
        spreads = []
        for own in ("H", "L"):
            probs = [sum(table[f"{own}-1-{messages}-none"]["kappa"][0])
                     for messages in ("2-0-0", "0-2-0", "0-0-2", "1-1-0")]
            spreads.append(max(probs) - min(probs))
        check(f"{model}_message_spread", sum(spreads) / 2,
              report["responsiveness"][model]["mean"])
        for n in (10, 20):
            key = f"{model}_n{n}"
            saved = report["conditions"][key]
            source = read(saved["source"])
            scores = source["scores"]
            per_run = {k: np.asarray(scores["crps_per_run"][k])
                       for k in ("order0", "actions_only", "signal_aware")}
            d = interval(per_run["actions_only"] - per_run["signal_aware"], rng)
            check(key + "_D_rederived", d,
                  [saved["D_rederived"][k] for k in ("mean", "lo", "hi")])
            check(key + "_D_committed_mean", d[0], scores["paired_D"]["mean"])
            check(key + "_D_published_interval",
                  [saved["D_committed"][k] for k in ("mean", "lo", "hi")],
                  [scores["paired_D"][k] for k in ("mean", "lo", "hi")])
            for audit in ("actions_only", "signal_aware"):
                diff = interval(per_run[audit] - per_run["order0"], rng)
                check(key + "_" + audit + "_minus_order0", diff,
                      [saved["minus_order0"][audit][k] for k in ("mean", "lo", "hi")])
                delta = scores["delta"][audit]
                check(key + "_" + audit + "_roundwise",
                      [max(delta), delta.index(max(delta)) + 1,
                       scores["floor"][audit]["max_over_rounds"]],
                      [saved["roundwise"][audit][k] for k in ("max_TV", "round", "cutoff")])
            for audit, values in per_run.items():
                check(key + "_" + audit + "_CRPS", np.mean(values), saved["crps"][audit])
                # Sign convention: F_error is live minus forecast.
                live = np.mean(source["X"]["swarm"]["mean"][15:20])
                # The order-zero score uses an exact binomial law, whereas X
                # records a simulation. Use the scored endpoint CDF for all three.
                forecast = np.sum(1 - np.asarray(scores["G"][audit])) / (5 * n)
                check(key + "_" + audit + "_F_error", live - forecast,
                      saved["F_error_live_minus_forecast"][audit]["mean"])

    persistence = read("results/extra/followup/persistence.json")
    raw = read("results/extra/audit/olmo-3-7b/cheap_talk_d2_enumerated_signal.json")["entries"]
    actions = read("results/extra/audit/olmo-3-7b/cheap_talk_d2_enumerated_actions.json")["entries"]
    messages = ("high", "low", "none")
    profiles = list(itertools.product(range(3), repeat=4))
    start = np.array([np.prod([raw["START"]["kappa"][0][m] for m in p]) for p in profiles])
    matrix = []
    for p in profiles:
        row = np.array([1.0])
        for i in range(4):
            neighbours = [p[(i - 1) % 4], p[(i + 1) % 4]]
            counts = "-".join(str(neighbours.count(m)) for m in range(3))
            probs = raw[f"H-2-{counts}-{messages[p[i]]}"]["kappa"][0]
            row = np.multiply.outer(row, probs).ravel()
        matrix.append(row)
    # Substochastic transition: probability lost here is an exit from all-high.
    stay = start @ np.linalg.matrix_power(np.array(matrix), 19)
    survival_sa = stay.sum() / start.sum()
    keep_ao = actions["H-2"]["kappa"]
    survival_ao = keep_ao ** 76
    for audit, survival in (("actions_only", survival_ao), ("signal_aware", survival_sa)):
        check(audit + "_all_high_start", start.sum(), persistence["p_all_high_round1"][audit])
        check(audit + "_survival", survival, persistence["survival_rounds_2_20"][audit])
        check(audit + "_start_times_survival", start.sum() * survival,
              persistence["start_times_survival"][audit])
    check("survival_ratio", survival_sa / survival_ao, persistence["survival_ratio_signal_over_actions"])
    check("effective_signal_retention", survival_sa ** (1 / 76),
          persistence["keep_high_per_agent_round"]["signal_aware_effective"])
    independent = read("results/extra/followup/independent_check.json")
    CHECKS["independent_exact_check_passed"] = independent["pass"]
    for audit, variant in (("actions_only", "neither"), ("signal_aware", "both")):
        check(audit + "_exact_endpoint", independent["olmo-3-7b"][f"E1_{variant}"]["P_F1"],
              persistence["exact_P_F1"][audit])

    sensitivity = read("results/extra/followup/sensitivity/results.json")
    for name, test in sensitivity["tests"].items():
        for hypothesis in ("null", "alternative"):
            r = test[hypothesis]
            check(name + "_" + hypothesis + "_rate", r["rejections"] / r["batches"], r["rate"])
    event = sensitivity["tests"]["S3_persistence_event"]
    check("event_exact_false_positive_rate", 1 - (1 - persistence["exact_P_F1"]["actions_only"]) ** 50,
          event["exact_null_level"])
    check("event_exact_power", 1 - (1 - persistence["exact_P_F1"]["signal_aware"]) ** 50,
          event["exact_alternative_power"])

    out = {"pass": all(CHECKS.values()), "checks": CHECKS, "input_sha256": HASHES}
    (ROOT / "results/followup/report_final_check.json").write_text(json.dumps(out, indent=2) + "\n")
    failures = [k for k, v in CHECKS.items() if not v]
    print(f"{len(CHECKS)} checks; failures: {failures}")
    raise SystemExit(0 if out["pass"] else 1)


if __name__ == "__main__":
    main()
