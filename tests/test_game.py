import pytest

from swarm_audit.game import DEFAULT_PARAMS, PayoffParams, frac_high, u


def test_payoff_corners():
    p = DEFAULT_PARAMS
    assert u("H", 1.0, p) == p.uHH
    assert u("H", 0.0, p) == p.uHL
    assert u("L", 1.0, p) == p.uLH
    assert u("L", 0.0, p) == p.uLL


def test_payoff_corners_other_params():
    p = PayoffParams(uLH=7, uHH=5, uLL=2, uHL=-1)
    assert (u("H", 1, p), u("H", 0, p), u("L", 1, p), u("L", 0, p)) == (5, -1, 7, 2)


def test_payoff_is_linear_in_f():
    assert u("H", 0.5) == pytest.approx(5.0)
    assert u("L", 0.5) == pytest.approx(8.0)
    assert u("L", frac_high(3, 10)) == pytest.approx(4 + 0.3 * 8)


def test_default_params_are_a_prisoners_dilemma():
    assert DEFAULT_PARAMS.is_prisoners_dilemma()
    assert not PayoffParams(uLH=20, uHH=10, uLL=4, uHL=0).is_prisoners_dilemma()  # 2 uHH = uLH + uHL


@pytest.mark.parametrize("bad", [("X", 0.5), ("H", -0.1), ("L", 1.5)])
def test_payoff_rejects_bad_input(bad):
    with pytest.raises(ValueError):
        u(*bad)


def test_frac_high_checks_range():
    assert frac_high(2, 4) == 0.5
    with pytest.raises(ValueError):
        frac_high(5, 4)
    with pytest.raises(ValueError):
        frac_high(0, 0)
