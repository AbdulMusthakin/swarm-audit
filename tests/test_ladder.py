import re

import pytest

from swarm_audit.ladder import Labels, Record, load_template, pi_k, q, render

LABELS = Labels(" H", " L", " high", " low", " none")


def _no_placeholders(text):
    assert not re.search(r"\{[A-Za-z_]+\}", text), text


@pytest.mark.parametrize("step", ["counts", "cheap_talk"])
def test_round_one_is_the_prompt_alone(step):
    r = render(step, (), d=4, labels=LABELS)
    assert "first round" in r.user
    assert "earlier rounds" not in r.user and "payoff last round" not in r.user
    assert r.assistant == "Bid:"
    for text in (r.system, r.user):
        _no_placeholders(text)
        assert "DRAFT" not in text  # file comments are not sent
    assert "4 other firms" in r.system
    assert "0 + 10 x" in r.system and "4 + 8 x" in r.system  # the rule, as numbers
    assert '"Bid: H"' in r.user or "Bid: H" in r.user


def test_counts_history_round_by_round():
    h = (Record("L", 1, 3), Record("H", 4, 0), Record("H", 2, 2))
    r = render("counts", h, d=4, labels=LABELS)
    lines = [l for l in r.user.splitlines() if l.startswith("- ")]
    assert lines == [
        "- 3 rounds ago: you bid low. Of your 4 neighbours, 1 bid high and 3 bid low.",
        "- 2 rounds ago: you bid high. Of your 4 neighbours, 4 bid high and 0 bid low.",
        "- Last round: you bid high. Of your 4 neighbours, 2 bid high and 2 bid low.",
    ]
    # Only the last round's payoff, u(H, 0.5) = 5, and no running total.
    assert "Your payoff last round was 5 points." in r.user
    assert r.user.count("payoff") == 1
    assert "round 4" not in r.user.lower() and "rounds remain" not in r.user
    assert r.after_bid is None
    _no_placeholders(r.user)


def test_counts_payoff_formatting():
    r = render("counts", (Record("L", 3, 7),), d=10, labels=LABELS)
    assert "Your payoff last round was 6.4 points." in r.user


def test_cheap_talk_history_and_signal_prefix():
    h = (Record("L", 2, 2, signal="high", sig_high=3, sig_low=1, sig_none=0),
         Record("H", 2, 2, signal="none", sig_high=0, sig_low=0, sig_none=4))
    r = render("cheap_talk", h, d=4, labels=LABELS)
    lines = [l for l in r.user.splitlines() if l.startswith("- ")]
    assert lines[0] == ('- 2 rounds ago: you bid low and sent "Next round I\'ll bid high". Of '
                        'your 4 neighbours, 2 bid high and 2 bid low; 3 sent "Next round '
                        'I\'ll bid high", 1 sent "Next round I\'ll bid low", and 0 sent no message.')
    assert lines[1].startswith("- Last round: you bid high and sent no message.")
    assert 'Message: high for "Next round I\'ll bid high"' in r.user
    assert "about your bid in the next round" in r.system
    assert r.signal_prefix(" H") == "Bid: H\nMessage:"
    _no_placeholders(r.user)


def test_labels_are_template_parameters():
    r = render("counts", (), d=2, labels=Labels("A", "B"))
    assert '"Bid: A" to bid high' in r.user and '"Bid: B" to bid low' in r.user


def test_last_k_cuts_history_but_keeps_last_payoff():
    h = (Record("L", 0, 2), Record("H", 2, 0))
    r = render("counts", h, d=2, labels=LABELS, last_k=1)
    assert "2 rounds ago" not in r.user
    assert "Last round: you bid high" in r.user
    assert "payoff last round was 10 points" in r.user


def test_record_validation():
    with pytest.raises(ValueError):
        Record("X", 1, 1)
    with pytest.raises(ValueError):
        Record("H", 1, 1, signal="high", sig_high=1, sig_low=0, sig_none=0)  # sums to 1, d = 2
    with pytest.raises(ValueError):
        render("counts", (Record("H", 1, 1),), d=3, labels=LABELS)  # wrong d
    with pytest.raises(ValueError):
        render("cheap_talk", (Record("H", 1, 1),), d=2, labels=LABELS)  # no signals


def test_q_and_pi_k():
    r = Record("H", 1, 1, signal="low", sig_high=0, sig_low=2, sig_none=0)
    assert q(r) == Record("H", 1, 1)
    h = tuple(Record("H", i, 3 - i) for i in range(4))
    assert pi_k(h, 2) == h[2:]
    assert pi_k(h[:1], 2) == h[:1]
    assert pi_k(h, 0) == ()


def test_templates_have_the_needed_sections():
    for step in ("counts", "cheap_talk"):
        t = load_template(step)
        for s in ("system", "first_round", "history_header", "history_line", "when_last",
                  "when_ago", "last_payoff", "instruction", "assistant"):
            assert s in t, (step, s)
    assert "assistant_after_bid" in load_template("cheap_talk")
