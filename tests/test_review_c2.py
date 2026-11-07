"""Review C2: immutable provenance for runs and tables, checked on resume and load."""

import json
import shutil

import pytest

from swarm_audit import ladder
from swarm_audit.audit import fit_table
from swarm_audit.game import PayoffParams
from swarm_audit.policies import ScriptedBackend
from swarm_audit.provenance import ProvenanceError, check_tables_match_runs
from swarm_audit.swarm import Condition, load_runs, run_condition

QUIET = {"log": lambda *a: None}
P11 = PayoffParams(uLH=11)


def _cond(runs, params=PayoffParams()):
    return Condition("t", 1, 1, "counts", 4, runs=runs, params=params)


def _be(params=PayoffParams()):
    return ScriptedBackend("copy", params=params)


def test_identical_resume_keeps_the_original_provenance(tmp_path):
    out = run_condition(_cond(2), _be(), tmp_path, **QUIET)
    prov_bytes = (out / "provenance.json").read_bytes()
    run_condition(_cond(4), _be(), tmp_path, **QUIET)          # top up to 4 runs
    assert (out / "provenance.json").read_bytes() == prov_bytes
    man = json.loads((tmp_path / "t" / "manifest.json").read_text())["conditions"][out.name]
    assert man["provenance"] == json.loads(prov_bytes)
    assert [s["runs_added"] for s in man["sessions"]] == [2, 2]
    assert len(list(out.glob("run_*.jsonl.gz"))) == 4


def test_payoff_change_cannot_resume(tmp_path):
    run_condition(_cond(2), _be(), tmp_path, **QUIET)
    with pytest.raises(ProvenanceError, match="payoffs"):
        run_condition(_cond(4, P11), _be(P11), tmp_path, **QUIET)


def test_prompt_change_cannot_resume(tmp_path, monkeypatch):
    run_condition(_cond(2), _be(), tmp_path, **QUIET)
    pdir = tmp_path / "prompts"
    shutil.copytree(ladder.PROMPT_DIR, pdir)
    p = pdir / "counts.txt"
    p.write_text(p.read_text().replace("Choose your bid", "Now choose your bid"))
    monkeypatch.setattr(ladder, "PROMPT_DIR", pdir)
    ladder.load_template.cache_clear()
    try:
        with pytest.raises(ProvenanceError, match="templates"):
            run_condition(_cond(4), _be(), tmp_path, **QUIET)
    finally:
        ladder.load_template.cache_clear()


def test_runs_without_provenance_cannot_resume(tmp_path):
    out = run_condition(_cond(2), _be(), tmp_path, **QUIET)
    (out / "provenance.json").unlink()
    with pytest.raises(ProvenanceError, match="without provenance"):
        run_condition(_cond(4), _be(), tmp_path, **QUIET)


def test_old_payoff_tables_are_refused_for_new_payoff_runs(tmp_path):
    old_table = fit_table(_be(), "counts", 2, "actions", 3, 2)
    data = load_runs(run_condition(_cond(2, P11), _be(P11), tmp_path, **QUIET))
    with pytest.raises(ProvenanceError, match="different kernel"):
        check_tables_match_runs(data.meta["kernel_fingerprint"], {"actions": old_table})
    new_table = fit_table(_be(P11), "counts", 2, "actions", 3, 2)
    check_tables_match_runs(data.meta["kernel_fingerprint"], {"actions": new_table})


def test_table_fingerprint_covers_the_audit_design():
    a = fit_table(_be(), "counts", 2, "actions", 3, 2).meta["provenance"]
    b = fit_table(_be(), "counts", 2, "actions", 4, 2).meta["provenance"]
    assert a["kernel_fingerprint"] == b["kernel_fingerprint"]
    assert a["fingerprint"] != b["fingerprint"]
