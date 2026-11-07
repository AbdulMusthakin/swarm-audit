"""Review C3: G5 gate records, enforced at the core launch."""

import json

import pytest

from swarm_audit import calibration
from swarm_audit.audit import enumerated_table
from swarm_audit.game import PayoffParams
from swarm_audit.policies import ScriptedBackend
from swarm_audit.provenance import fingerprint, kernel_identity
from swarm_audit.swarm import Condition, load_runs, run_condition
from swarm_audit.topology import cycle


def _calibrate(tmp_path, channel, be):
    kind = "actions" if channel == "counts" else "signal"
    tab = {kind: enumerated_table(be, channel, 2, kind)}
    if channel == "cheap_talk":
        from swarm_audit.audit import derive_actions_table
        tab["actions"] = derive_actions_table(tab["signal"])
    c = Condition("cal", 4, 1, channel, 4, last_k=1, runs=40)
    data = load_runs(run_condition(c, be, tmp_path / "runs", log=lambda *a: None))
    res = calibration.check_arm(data, tab, cycle(4), floor=0.0, sims=2000)
    rec = calibration.gate_record(res, data, tab, 0.0)
    calibration.write_gate(rec, tmp_path / "gates")
    return rec


def test_matching_passing_records_permit_launch(tmp_path):
    be = ScriptedBackend("copy")
    for ch in ("counts", "cheap_talk"):
        assert _calibrate(tmp_path, ch, be)["pass"]
    calibration.require_g5(fingerprint(kernel_identity(be)), "cheap_talk", folder=tmp_path / "gates")
    calibration.require_g5(fingerprint(kernel_identity(be)), "counts", folder=tmp_path / "gates")


def test_missing_record_blocks(tmp_path):
    be = ScriptedBackend("copy")
    _calibrate(tmp_path, "counts", be)
    with pytest.raises(SystemExit, match="cheap_talk: no G5 record"):
        calibration.require_g5(fingerprint(kernel_identity(be)), "cheap_talk", folder=tmp_path / "gates")
    calibration.require_g5(fingerprint(kernel_identity(be)), "counts", folder=tmp_path / "gates")


def test_stale_record_blocks_after_a_payoff_change(tmp_path):
    be = ScriptedBackend("copy")
    for ch in ("counts", "cheap_talk"):
        _calibrate(tmp_path, ch, be)
    new_kernel = fingerprint(kernel_identity(ScriptedBackend("copy", params=PayoffParams(uLH=11))))
    with pytest.raises(SystemExit, match="stale"):
        calibration.require_g5(new_kernel, "cheap_talk", folder=tmp_path / "gates")


def test_failed_record_blocks_and_history_is_kept(tmp_path):
    be = ScriptedBackend("copy")
    for ch in ("counts", "cheap_talk"):
        _calibrate(tmp_path, ch, be)
    p = tmp_path / "gates" / "G5_counts.json"
    rec = json.loads(p.read_text())
    rec["pass"] = False
    calibration.write_gate(rec, tmp_path / "gates")
    with pytest.raises(SystemExit, match="G5 failed"):
        calibration.require_g5(fingerprint(kernel_identity(be)), "counts", folder=tmp_path / "gates")
    assert len((tmp_path / "gates" / "history.jsonl").read_text().splitlines()) == 3


def test_actions_only_finding_is_not_part_of_the_gate(tmp_path):
    rec = _calibrate(tmp_path, "cheap_talk", ScriptedBackend("hidden_signal"))
    assert set(rec["required"]) == {"per_decision", "per_run"}
    assert rec["finding_not_gate"] == "actions-only chain on cheap talk"
    assert rec["pass"]


def _fail_cheap(tmp_path, be):
    for ch in ("counts", "cheap_talk"):
        _calibrate(tmp_path, ch, be)
    p = tmp_path / "gates" / "G5_cheap_talk.json"
    rec = json.loads(p.read_text())
    rec["pass"], rec["required"] = False, {"per_decision": True, "per_run": False}
    p.write_text(json.dumps(rec))
    return rec


def test_exception_needs_a_passing_confirmation_and_only_pauses_cheap_talk(tmp_path):
    be = ScriptedBackend("copy")
    kfp = fingerprint(kernel_identity(be))
    gates = tmp_path / "gates"
    rec = _fail_cheap(tmp_path, be)
    with pytest.raises(SystemExit, match="no matching approved exception"):
        calibration.require_g5(kfp, "cheap_talk", folder=gates)
    calibration.write_exception("cheap_talk", rec, "Abdul", "2026-11-07", "chance", [], gates)
    with pytest.raises(SystemExit, match="confirmation has not run"):
        calibration.require_g5(kfp, "cheap_talk", folder=gates)
    conf = {"pass": False, "kernel_fingerprint": kfp}
    (gates / "G5_cheap_talk_confirmation.json").write_text(json.dumps(conf))
    with pytest.raises(SystemExit, match="paused"):
        calibration.require_g5(kfp, "cheap_talk", folder=gates)
    calibration.require_g5(kfp, "counts", folder=gates)            # counts is not paused
    conf["pass"] = True
    (gates / "G5_cheap_talk_confirmation.json").write_text(json.dumps(conf))
    calibration.require_g5(kfp, "cheap_talk", folder=gates)


def test_exception_is_bound_to_its_runs(tmp_path):
    be = ScriptedBackend("copy")
    kfp = fingerprint(kernel_identity(be))
    gates = tmp_path / "gates"
    rec = _fail_cheap(tmp_path, be)
    calibration.write_exception("cheap_talk", {**rec, "run_fingerprint": "other"}, "Abdul",
                                "2026-11-07", "chance", [], gates)
    (gates / "G5_cheap_talk_confirmation.json").write_text(
        json.dumps({"pass": True, "kernel_fingerprint": kfp}))
    with pytest.raises(SystemExit, match="no matching approved exception"):
        calibration.require_g5(kfp, "cheap_talk", folder=gates)


def test_per_decision_failures_cannot_be_excepted(tmp_path):
    rec = {"pass": False, "required": {"per_decision": False, "per_run": True},
           "kernel_fingerprint": "k", "run_fingerprint": "r", "runs": 50}
    with pytest.raises(ValueError):
        calibration.write_exception("cheap_talk", rec, "Abdul", "d", "x", [], tmp_path)
