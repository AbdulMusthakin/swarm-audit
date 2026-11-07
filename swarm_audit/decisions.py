"""The decisions settled in advance on 7 Nov 2026 (`decisions:` in the build configs).

Pure functions, so the rules are tested on CPU; the GPU scripts call them on measured
numbers and apply the outcome mechanically.
"""

from __future__ import annotations

from .audit import table_states
from .swarm import Condition

FLOOR_EPS = 1e-12


# ------------------------------------------------------------------------------ G3


def g3_max(r: dict) -> float:
    return max(float(r["max_abs_diff_p_H"]), float(r["max_abs_diff_signal"]))


def decide_g3(default: dict, invariant: dict | None, rules: dict) -> dict:
    """Pick the serving setting and the numerical floor from the G3 results.

    - default max 0: keep the defaults, floor 0;
    - else, with the batch-invariant result: use it if its max is smaller and its
      throughput is at least `min_throughput_ratio` of the default's;
    - the floor is the chosen setting's max; above `max_floor` the gate stops."""
    d0 = g3_max(default)
    if d0 <= FLOOR_EPS:
        choice, floor, why = False, 0.0, "default settings are exactly repeatable"
    elif invariant is None:
        return {"status": "need_batch_invariant_run", "default_max": d0}
    else:
        d1 = g3_max(invariant)
        ratio = (float(invariant["throughput_prompts_per_s_batch"])
                 / float(default["throughput_prompts_per_s_batch"]))
        if d1 < d0 and ratio >= rules["min_throughput_ratio"]:
            choice, floor = True, d1
            why = f"batch invariance: max {d1:.3g} < {d0:.3g}, throughput ratio {ratio:.2f}"
        else:
            choice, floor = False, d0
            why = (f"default: batch invariance max {d1:.3g} vs {d0:.3g}, throughput ratio "
                   f"{ratio:.2f} (needs a smaller max and ratio >= {rules['min_throughput_ratio']})")
    status = "stop" if floor > rules["max_floor"] else "decided"
    return {"status": status, "batch_invariant": choice, "numerical_floor": floor,
            "reason": why, "default_max": d0,
            "invariant_max": g3_max(invariant) if invariant else None,
            "max_floor": rules["max_floor"]}


# ------------------------------------------------------------------------------ G4


def plan_items(cfgs: dict[str, dict], signal_audit: dict, sec_per_decision: dict) -> list[dict]:
    """Every remaining GPU job of the manifest with its projected seconds.
    `cfgs` maps config file names to loaded configs; `sec_per_decision[channel]` is the
    batched GPU time per decision (cheap talk includes its two signal read-outs)."""
    from .build import conditions, drift_panel

    core = cfgs["core.yaml"]
    own = core["own_signal_in_state"]
    a = core["audit"]["actions"]
    panel = len(drift_panel(core))
    sp = sec_per_decision
    items = []

    def add(arm, cond, channel, decisions, cuttable):
        items.append({"arm": arm, "condition": cond, "channel": channel, "decisions": decisions,
                      "seconds": decisions * sp[channel], "cut_tag": cuttable})

    for name, cfg in cfgs.items():
        for arm in cfg.get("arms", {}):     # pilot.yaml has no arms; it is added below
            for c in conditions(cfg, arm):
                tag = ("arm7" if arm.startswith("second") else
                       f"n{c.n}" if arm.startswith("core") else None)
                add(arm, c.name, c.channel, c.runs * c.n * c.T, tag)
                add(f"drift/{arm}", c.name, "cheap_talk", 3 * panel, tag)  # start, end, mixed
    n_sig = len(table_states("cheap_talk", 2, "signal", own)) - 1
    add("audit", "counts_d2", "counts", 6 * (a["n_fit"] + a["n_score"]) + 2, None)
    add("audit", "cheap_talk_d2_signal", "cheap_talk",
        n_sig * (signal_audit["n_fit"] + signal_audit["n_score"]) + 2, None)
    add("audit", "counts_d2_enumerated", "counts", 7, None)
    add("audit", "cheap_talk_d2_enumerated", "cheap_talk", n_sig + 1, None)
    for ch in ("counts", "cheap_talk"):
        add("audit", f"{ch}_d9_actions", ch, 20 * (a["n_fit"] + a["n_score"]) + 2, "arm7")
    pil = cfgs.get("pilot.yaml")
    if pil:
        add("pilot", "counts_n10", "counts", pil["runs"] * pil["n"] * core["T"], None)
    return items


def plan_g4(cfgs: dict[str, dict], sec_per_decision: dict, spent_usd: float) -> dict:
    """Fit the remaining manifest into the budget with the settled rules: first the
    reduced signal-aware audit, then the cut order (n = 20); arm 7 only if everything
    else fits at full size."""
    core = cfgs["core.yaml"]
    rules = core["decisions"]["g4"]
    srv = core["serving"]
    price, cap, margin = srv["l4_usd_per_hour"], srv["cost_cap_usd"], rules["cost_margin"]

    def cost(items):
        h = sum(i["seconds"] for i in items) / 3600
        return h, h * price

    def fits(items):
        h, usd = cost(items)
        return spent_usd + usd * margin <= cap and h * margin <= rules["max_gpu_hours"]

    steps = []
    audit = dict(core["audit"]["signal"])
    items = plan_items(cfgs, audit, sec_per_decision)
    base = [i for i in items if i["cut_tag"] != "arm7"]
    if not fits(base):
        audit = dict(rules["reduced_signal_audit"])
        steps.append(f"signal-aware audit reduced to {audit}")
        items = plan_items(cfgs, audit, sec_per_decision)
        base = [i for i in items if i["cut_tag"] != "arm7"]
    for tag in rules["cut_order"]:
        if fits(base):
            break
        base = [i for i in base if i["cut_tag"] != tag]
        steps.append(f"cut {tag} (hand-off cut order)")
    if not fits(base):
        h, usd = cost(base)
        return {"status": "stop", "steps": steps, "projected_hours": h, "projected_usd": usd,
                "reason": "arms 1-4b plus n = 10 in both channels do not fit the budget"}
    # Arm 7 is the optional extra: it runs only if everything else fits at full size
    # (nothing reduced or cut), never at the cost of the audit or n = 20.
    arm7 = [i for i in items if i["cut_tag"] == "arm7"]
    if core_arm7_allowed(cfgs) and not steps and fits(base + arm7):
        run = base + arm7
    else:
        run = base
        steps.append("arm 7 not run (does not fit at full size with the margin)"
                     if core_arm7_allowed(cfgs) else "arm 7 not approved")
    h, usd = cost(run)
    return {"status": "go", "steps": steps, "signal_audit": audit, "run": run,
            "projected_hours": h, "projected_usd": usd, "margin": margin,
            "spent_usd": spent_usd, "cap_usd": cap, "max_gpu_hours": rules["max_gpu_hours"],
            "cut": [i for i in items if i not in run]}


def core_arm7_allowed(cfgs: dict) -> bool:
    sec = cfgs.get("secondary.yaml", {})
    return sec.get("approved") in (True, "if_budget")


# ---------------------------------------------------------------------------- drift


def drift_action(change: float, floor: float, pause_above_by: float) -> str:
    """'ok' within the floor; 'report' above it by at most `pause_above_by`; else 'pause'."""
    if change <= floor + FLOOR_EPS:
        return "ok"
    return "report" if change <= floor + pause_above_by else "pause"


def condition_names(cfg: dict, arm: str) -> list[str]:
    from .build import conditions

    return [c.name for c in conditions(cfg, arm)]


_ = Condition
