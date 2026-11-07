#!/usr/bin/env python3
"""Gate G3, serving repeatability (GPU, main model; docs/handoff-build.md section 4).

About 200 prompts (audit prompts from p_test and swarm-like prompts from rounds 1-20,
both channels) are each queried (a) alone, (b) inside a shuffled batch of about 1,000
other prompts, and (c) again after the prefix cache is warm. Reports the maximum
difference in p_H and in each signal probability between any two of these calls, and
the throughput.

    python scripts/check_serving.py --config core.yaml
    SWARM_AUDIT_BATCH_INVARIANT=1 python scripts/check_serving.py --config core.yaml \
        --batch-invariant      # only if the first run's maximum is not 0

If the maximum is 0, keep the default settings. If not, rerun with batch invariance.
`--decide` then applies the rule settled on 7 Nov 2026 (decisions.g3 in the configs):
batch invariance if its maximum is smaller and it keeps at least half the throughput;
the chosen setting's maximum is the numerical floor, and a floor above 0.07 stops.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

from swarm_audit import build  # noqa: E402
from swarm_audit.audit import synthetic_history  # noqa: E402
from swarm_audit.policies import Query  # noqa: E402
from swarm_audit.states import all_actions_states  # noqa: E402
from swarm_audit.swarm import Condition, simulate  # noqa: E402
from swarm_audit.ladder import Record  # noqa: E402


def prompt_set(cfg: dict, n_audit: int, n_swarm: int, seed: int) -> list[Query]:
    """Audit prompts (p_test) and swarm-like prompts (histories from a scripted Copy
    swarm on the 10-cycle, rounds 1-20), half per channel."""
    rng = np.random.default_rng(seed)
    pt = build.ptest(cfg)
    sts = all_actions_states(2)
    out = []
    for ch in ("counts", "cheap_talk"):
        for j in range(n_audit // 2):
            out.append(Query(ch, synthetic_history(rng, ch, 2, sts[j % len(sts)], pt), 2))
        cond = Condition("g3", 0, seed, ch, 10, runs=1)
        lines = simulate(cond, build.scripted("copy", cfg), [0])[0]
        hist: dict[int, list] = {}
        for ln in lines:
            r = ln["record"]
            rec = Record(r["action"], r["n_high"], r["n_low"], r.get("signal"), r.get("sig_high"),
                         r.get("sig_low"), r.get("sig_none"))
            hist.setdefault(ln["agent"], []).append(rec)
        picks = rng.integers(0, 20, size=n_swarm // 2), rng.integers(0, 10, size=n_swarm // 2)
        for t, i in zip(*picks):
            out.append(Query(ch, tuple(hist[int(i)][:int(t)]), 2))
    return out


def vec(o) -> np.ndarray:
    v = [o.p_H]
    if o.sig:
        v += list(o.sig["H"]) + list(o.sig["L"])
    return np.array(v)


def decide(cfg: dict) -> None:
    """Apply the settled G3 rule (decisions.g3 in the configs) to the measured results."""
    from swarm_audit import decisions

    sdir = REPO / "results" / "serving"
    r0 = json.loads((sdir / "g3_batchinv0.json").read_text())
    p1 = sdir / "g3_batchinv1.json"
    r1 = json.loads(p1.read_text()) if p1.exists() else None
    d = decisions.decide_g3(r0, r1, cfg["decisions"]["g3"])
    (sdir / "g3_decision.json").write_text(json.dumps(d, indent=1) + "\n")
    print(json.dumps(d, indent=1))
    if d["status"] == "need_batch_invariant_run":
        raise SystemExit("Default settings are not exactly repeatable: run\n  "
                         "SWARM_AUDIT_BATCH_INVARIANT=1 python scripts/check_serving.py "
                         "--batch-invariant\nthen --decide again.")
    if d["status"] == "stop":
        raise SystemExit(f"The G3 floor {d['numerical_floor']:.3g} is above "
                         f"{d['max_floor']}: stop and report to Abdul.")
    build.set_config_line("batch_invariant", "true" if d["batch_invariant"] else "false",
                          note="set by check_serving.py --decide (G3)")
    build.set_config_line("numerical_floor", repr(float(d["numerical_floor"])),
                          note="the G3 floor; set by check_serving.py --decide")
    print("Recorded in configs/{core,calibration,secondary}.yaml. Later GPU runs need "
          f"SWARM_AUDIT_BATCH_INVARIANT={'1' if d['batch_invariant'] else '0'}.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="core.yaml")
    ap.add_argument("--batch-invariant", action="store_true",
                    help="record that this run uses VLLM_BATCH_INVARIANT=1 (set the env var too)")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--others", type=int, default=1000)
    ap.add_argument("--decide", action="store_true",
                    help="apply the settled G3 rule to the measured results (no GPU)")
    args = ap.parse_args()
    cfg = build.load(args.config)
    if args.decide:
        decide(cfg)
        return
    cfg["serving"]["batch_invariant"] = args.batch_invariant
    test = prompt_set(cfg, args.n // 2, args.n // 2, seed=31)
    others = prompt_set(cfg, args.others // 2, args.others // 2, seed=32)
    build.check_cost(cfg, cfg["serving"]["timeout_minutes"])
    with build.gpu_session(cfg, ("model_confirmed", "modal_account")):
        be, _ = build.modal_backend(cfg, log_name="g3")
        t0 = time.time()
        alone = [vec(be.query([q], fresh=True)[0]) for q in test]
        t_alone = time.time() - t0
        rng = np.random.default_rng(0)
        allq = test + others
        perm = rng.permutation(len(allq))
        t1 = time.time()
        outs = be.query([allq[i] for i in perm], fresh=True)
        t_batch = time.time() - t1
        pos = {int(i): k for k, i in enumerate(perm)}
        batch = [vec(outs[pos[j]]) for j in range(len(test))]
        t2 = time.time()
        warm = [vec(o) for o in be.query(test, fresh=True)]
        t_warm = time.time() - t2
        sec = time.time() - t0
        build.record_spend(f"G3 batch_invariant={args.batch_invariant}", sec, cfg)
    diffs = {"alone_vs_batch": [], "alone_vs_warm": [], "batch_vs_warm": []}
    for a, b, w in zip(alone, batch, warm):
        diffs["alone_vs_batch"].append(np.abs(a - b))
        diffs["alone_vs_warm"].append(np.abs(a - w))
        diffs["batch_vs_warm"].append(np.abs(b - w))
    res = {"batch_invariant": args.batch_invariant, "prompts": len(test), "others": len(others),
           "max_abs_diff_p_H": float(max(float(x[0]) for v in diffs.values() for x in v)),
           "max_abs_diff_signal": float(max([float(x[1:].max()) for v in diffs.values() for x in v
                                             if len(x) > 1] or [0.0])),
           "by_pair": {k: float(max(float(x.max()) for x in v)) for k, v in diffs.items()},
           "seconds": {"alone": t_alone, "batch": t_batch, "warm": t_warm, "total": sec},
           "throughput_prompts_per_s_batch": len(allq) / t_batch}
    out = REPO / "results" / "serving" / f"g3_batchinv{int(args.batch_invariant)}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))
    print("Next: python scripts/check_serving.py --decide (the settled G3 rule picks the "
          "setting and the floor; both maxima and throughputs are reported to Abdul).")


if __name__ == "__main__":
    main()
