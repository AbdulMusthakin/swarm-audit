"""Tests for the extra arms (docs/handoff-extra-arms.md): the random regular graph, the
graph parameters carried by a condition, and the configurable results paths."""

import json

import numpy as np
import pytest

from swarm_audit import build
from swarm_audit.swarm import Condition, graph_of, load_runs, run_condition
from swarm_audit.topology import _connected, diameter, make, random_regular


@pytest.mark.parametrize("n,d,seed", [(100, 4, 20261108), (20, 4, 1), (10, 3, 5)])
def test_random_regular_is_simple_regular_and_connected(n, d, seed):
    g = random_regular(n, d, seed)
    assert g.regular_degree == d
    assert all(i not in nb for i, nb in enumerate(g.in_neighbours))       # no self-loops
    assert all(len(set(nb)) == len(nb) for nb in g.in_neighbours)         # no repeated edges
    assert len(g.edges) == n * d // 2
    assert all(i in g.in_neighbours[j] for i, nb in enumerate(g.in_neighbours) for j in nb)
    assert _connected(g)


def test_random_regular_same_seed_same_graph():
    a, b = random_regular(100, 4, 20261108), random_regular(100, 4, 20261108)
    assert a.in_neighbours == b.in_neighbours
    assert random_regular(100, 4, 20261109).in_neighbours != a.in_neighbours
    assert make("random_regular", 100, d=4, seed=20261108).in_neighbours == a.in_neighbours


def test_random_regular_rejects_impossible_sizes():
    with pytest.raises(ValueError):
        random_regular(5, 3, 0)       # n * d odd
    with pytest.raises(ValueError):
        random_regular(4, 4, 0)       # d >= n


def test_diameter_of_known_graphs():
    assert diameter(make("cycle", 10)) == 5
    assert diameter(make("complete", 6)) == 1


def test_condition_graph_kwargs_and_unchanged_core_names(tmp_path):
    core = Condition("core_counts", 5, 10, "counts", 10)
    assert core.name == "counts_n10_cycle_full"                           # unchanged
    c = Condition("extra_rr4", 8, 20, "counts", 20, graph="random_regular", T=4, runs=2,
                  graph_kwargs=(("d", 4), ("seed", 3)))
    assert c.name == "counts_n20_random_regular_d4_seed3_full"
    assert c.make_graph().in_neighbours == random_regular(20, 4, 3).in_neighbours
    out = run_condition(c, build.scripted("copy", build.load("extra-scale.yaml")), tmp_path,
                        log=lambda *a: None)
    data = load_runs(out)
    assert data.meta["graph_kwargs"] == {"d": 4, "seed": 3}
    assert graph_of(data).in_neighbours == c.make_graph().in_neighbours
    # core runs carry no graph_kwargs field, so their logs are unchanged
    out2 = run_condition(Condition("core_counts", 5, 4, "counts", 4, T=2, runs=1),
                         build.scripted("coin", build.load("core.yaml")), tmp_path,
                         log=lambda *a: None)
    assert "graph_kwargs" not in load_runs(out2).meta


def test_extra_config_builds_the_arm8_condition():
    cfg = build.load("extra-scale.yaml")
    (c,) = build.conditions(cfg, "extra_rr4")
    assert (c.arm_id, c.n, c.runs, c.T, c.channel, c.last_k) == (8, 100, 50, 20, "counts", None)
    g = c.make_graph()
    assert g.regular_degree == 4 and _connected(g)
    assert set(cfg["inputs"]["forbidden_workspaces"]) >= {"abdulmusthakin", "aliyusufbelal"}
    core = build.load("core.yaml")
    for k in ("model", "payoffs", "T", "p_test", "own_signal_in_state", "audit", "seeds",
              "analysis", "decisions"):
        assert cfg[k] == core[k], k
    assert {k: v for k, v in cfg["serving"].items() if k != "cost_cap_usd"} == \
        {k: v for k, v in core["serving"].items() if k != "cost_cap_usd"}


def test_results_paths_default_to_core_and_follow_results_dir(tmp_path, monkeypatch):
    core = build.load("core.yaml")
    assert build.results_root(core) == build.RESULTS
    assert build.spend_file(core) == build.SPEND_FILE
    assert build.deploy_file(core) == build.DEPLOY_FILE
    assert build.table_dir(core, "x") == build.RESULTS / "audit" / "x"
    monkeypatch.setattr(build, "REPO_ROOT", tmp_path)
    cfg = {**core, "results_dir": "results/extra"}
    assert build.results_root(cfg) == tmp_path / "results" / "extra"
    assert build.spend_file(cfg) == tmp_path / "results" / "extra" / "spend.json"
    assert build.deploy_file(cfg) == tmp_path / "results" / "extra" / "serving" / "deployed.json"
    build.record_spend("test", 3600, cfg)
    assert build.spent_usd(cfg) == pytest.approx(core["serving"]["l4_usd_per_hour"])
    assert json.loads(build.spend_file(cfg).read_text())[0]["what"] == "test"


def test_collect_writes_only_under_the_extra_tree(tmp_path, monkeypatch):
    import importlib.util
    import sys

    from swarm_audit import analysis, prediction  # noqa: F401
    from swarm_audit.analysis import analyse_condition, to_jsonable
    from swarm_audit.audit import fit_table

    spec = importlib.util.spec_from_file_location("analyse_script", build.REPO_ROOT / "scripts" / "analyse.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cfg = build.load("extra-scale.yaml")
    be = build.scripted("copy", cfg)
    c = Condition("extra_test", 8, 8, "counts", 8, graph="random_regular", T=6, runs=6,
                  graph_kwargs=(("d", 4), ("seed", 3)))
    out = run_condition(c, be, tmp_path / "runs", log=lambda *a: None)
    from swarm_audit.analysis import Tables
    t = fit_table(be, "counts", 4, "actions", 4, 2, build.ptest(cfg), True)
    data = load_runs(out)
    res = analyse_condition(data, graph_of(data), Tables(t, None, None), 500, 1, 50, build.ptest(cfg))
    root = tmp_path / "extra"
    (root / "analysis" / "extra_test").mkdir(parents=True)
    (root / "analysis" / "extra_test" / f"{c.name}.json").write_text(json.dumps(to_jsonable(res)))
    core_tables = build.RESULTS / "tables"
    before = {p: p.stat().st_mtime for p in core_tables.glob("*")} if core_tables.exists() else {}
    mod.collect(root)
    assert (root / "tables" / "scores.md").exists() and (root / "tables" / "defects.md").exists()
    assert (root / "figures" / "fig2.png").exists()
    after = {p: p.stat().st_mtime for p in core_tables.glob("*")} if core_tables.exists() else {}
    assert before == after


def test_max_runs_runs_a_prefix_and_resumes(tmp_path):
    cfg = build.load("extra-scale.yaml")
    c = Condition("extra_test", 8, 12, "counts", 12, graph="random_regular", T=3, runs=7,
                  graph_kwargs=(("d", 4), ("seed", 5)))
    be = build.scripted("coin", cfg)
    out = run_condition(c, be, tmp_path, max_runs=5, log=lambda *a: None)
    assert sorted(p.name for p in out.glob("run_*")) == [f"run_{r:03d}.jsonl.gz" for r in range(5)]
    first = load_runs(out).actions.copy()
    run_condition(c, be, tmp_path, log=lambda *a: None)
    data = load_runs(out)
    assert data.R == 7 and (data.actions[:5] == first).all()
    man = json.loads((tmp_path / "extra_test" / "manifest.json").read_text())
    assert [s["runs"] for s in man["conditions"][c.name]["sessions"]] == [[0, 1, 2, 3, 4], [5, 6]]
    # the same 7 runs in one go give the same logs (lockstep batching does not matter)
    out2 = run_condition(c, be, tmp_path / "again", log=lambda *a: None)
    assert (load_runs(out2).actions == data.actions).all()


def test_core_pilot_payoff_check(tmp_path):
    cfg = build.load("extra-scale.yaml")
    with pytest.raises(SystemExit):
        build.check_core_pilot_payoffs(cfg, tmp_path / "missing.json")
    p = tmp_path / "summary.json"
    p.write_text(json.dumps({"frozen_payoffs": dict(cfg["payoffs"])}))
    assert build.check_core_pilot_payoffs(cfg, p) == cfg["payoffs"]
    p.write_text(json.dumps({"frozen_payoffs": {**cfg["payoffs"], "uLH": 11}}))
    with pytest.raises(SystemExit):
        build.check_core_pilot_payoffs(cfg, p)


def test_olmo_config_differs_only_in_model_and_arms():
    a, b = build.load("extra-scale.yaml"), build.load("extra-olmo.yaml")
    for k in set(a) | set(b):
        if k in ("model", "arms", "_path", "serving_subdir"):
            continue
        assert a.get(k) == b.get(k), k
    assert b["model"]["hf_id"] == "allenai/Olmo-3-7B-Instruct"
    assert b["model"]["revision"] == "6e5971d9eba42665f5bd5a0fcf047f299ce1dccc"
    assert build.spend_file(a) == build.spend_file(b)
    assert build.serving_dir(b) == build.serving_dir(a) / "olmo-3-7b"
    (ca,) = build.conditions(b, "olmo_calibration_cheap_talk")
    (cb,) = build.conditions(b, "olmo_core_cheap_talk")
    assert (ca.arm_id, ca.n, ca.last_k, ca.runs) == (9, 4, 1, 50)
    assert (cb.arm_id, cb.n, cb.last_k, cb.runs) == (91, 10, None, 50)
    calib = build.load("calibration.yaml")["arms"]["calibration_cheap_talk"]
    olmo = b["arms"]["olmo_calibration_cheap_talk"]
    assert {k: v for k, v in calib.items() if k != "arm_id"} == {k: v for k, v in olmo.items() if k != "arm_id"}


def test_extra_gates_never_touch_core_gates_and_lock_in_count():
    from swarm_audit.swarm import RunData

    assert build.gates_dir(build.load("core.yaml")) is None
    g = build.gates_dir(build.load("extra-olmo.yaml"))
    assert g == build.results_root(build.load("extra-olmo.yaml")) / "gates" / "olmo-3-7b"
    R, T, n = 50, 20, 4
    acts = np.zeros((R, T, n), np.int8)
    acts[:6, 17, 2] = 1            # 6 runs with a high bid in round 18
    acts[6:, 14, :] = 1            # round 15 does not count
    d = RunData("cheap_talk", n, "cycle", T, np.arange(R), acts, *(np.zeros((R, T, n)),) * 2,
                np.empty((R, T, n), object), np.empty((R, T, n), object))
    lk = build.lock_in(d)
    assert lk["runs_all_low"] == 44 and lk["stop"] is False
    acts[5, 17, 2] = 0
    assert build.lock_in(d)["stop"] is True
