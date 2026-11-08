"""Extra-arms overnight follow-up (docs/overnight-extra-plan.md)."""

from __future__ import annotations

from swarm_audit import build


def test_followup_config_differs_from_olmo_config_only_in_arms():
    a, b = build.load("extra-olmo.yaml"), build.load("extra-followup-olmo.yaml")
    for k in set(a) | set(b):
        if k in ("arms", "_path"):
            continue
        assert a.get(k) == b.get(k), k
    assert build.spend_file(a) == build.spend_file(b)       # one ledger, one $25 cap
    assert build.gates_dir(a) == build.gates_dir(b)         # 9a's G5 record is shared
    (c10,) = build.conditions(b, "followup_olmo_n10_last1")
    (c4,) = build.conditions(b, "followup_olmo_n4_full")
    assert (c10.arm_id, c10.n, c10.last_k, c10.runs) == (92, 10, 1, 50)
    assert (c4.arm_id, c4.n, c4.last_k, c4.runs) == (93, 4, None, 50)
    used = {int(v["arm_id"]) for cfg in (a, build.load("extra-scale.yaml"), build.load("core.yaml"),
                                         build.load("calibration.yaml")) for v in cfg["arms"].values()}
    assert not {92, 93} & used                               # disjoint seed namespaces
