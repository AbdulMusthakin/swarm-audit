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
