from swarm_audit.screening import build_grid, load_config


def test_grid_sizes_match_the_handoff():
    states = build_grid(load_config())
    counts = [s for s in states if s.step == "counts"]
    cheap = [s for s in states if s.step == "cheap_talk"]
    assert len(counts) == 72 and len(cheap) == 32
    assert {s.length for s in states} == {1, 5}
    assert len({s.prompt_id for s in states}) == len(states)


def test_fillers_are_shared_within_step_and_d():
    states = build_grid(load_config())
    a = next(s for s in states if s.step == "counts" and s.d == 4 and s.filler == 0)
    b = next(s for s in states if s.step == "counts" and s.d == 4 and s.filler == 0
             and s.state_id != a.state_id)
    assert a.history[:-1] == b.history[:-1] and a.history[-1] != b.history[-1]


def test_grid_is_deterministic():
    cfg = load_config()
    assert build_grid(cfg) == build_grid(cfg)
