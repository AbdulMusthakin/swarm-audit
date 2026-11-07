"""Review C1: fresh repeatability/drift measurements; serving identity in the cache key."""

import math

import pytest

from swarm_audit import build, serving
from swarm_audit.ladder import Labels, Record
from swarm_audit.policies import LLMBackend, Query

LABELS = Labels(" H", " L", " high", " low", " none")


class ChangingServer:
    """A fake server whose answers change on every call (as a drifting kernel would)."""

    def __init__(self):
        self.calls = 0

    def label_logprobs(self, prompts, ids):
        self.calls += 1
        p = 0.3 + 0.01 * self.calls
        raw = [p] + [(1 - p) / (len(ids) - 1)] * (len(ids) - 1)
        return [{"logprobs": [math.log(x) for x in raw], "probs": raw, "p_valid": 1.0}
                for _ in prompts]


def _backend(tok, server, identity):
    ids_a = tuple(tok.encode(x, add_special_tokens=False)[-1] for x in ("Bid: H", "Bid: L"))
    ids_s = tuple(tok.encode("Message:" + x, add_special_tokens=False)[-1]
                  for x in (" high", " low", " none"))

    def call(prompts, ids, kind, fresh=False):
        out, replayed = serving.call_logged(
            server.label_logprobs, prompts, ids, log_name="test", prompts=prompts,
            prompt_ids=[str(i) for i in range(len(prompts))],
            meta={"call": kind, "allowed_token_ids": ids, "serving": identity},
            fresh=fresh, return_status=True)
        return [{**r, "replayed": replayed} for r in out]
    return LLMBackend(tok, LABELS, ids_a, ids_s, call)


@pytest.fixture
def resume_on(monkeypatch, tmp_path):
    monkeypatch.setattr(serving, "LOG_DIR", tmp_path / "serving")
    monkeypatch.setenv("SWARM_AUDIT_RESUME", "1")


PANEL = [Query("counts", (Record("H", 1, 1),), 2, 1),
         Query("cheap_talk", (Record("L", 0, 2, "high", 1, 0, 1),), 2, 1)]


def test_drift_panel_is_measured_fresh_even_with_resume(toy_tokenizer, resume_on):
    srv = ChangingServer()
    be = _backend(toy_tokenizer, srv, {"rev": "a"})
    start = build.panel_outputs(be, PANEL)
    calls_after_start = srv.calls
    end = build.panel_outputs(be, PANEL)
    assert srv.calls == 2 * calls_after_start          # the end panel hit the server again
    assert build.panel_max_change(start, end) > 0       # and the change is detected


def test_mixed_batch_is_fresh(toy_tokenizer, resume_on):
    srv = ChangingServer()
    be = _backend(toy_tokenizer, srv, {"rev": "a"})
    start = build.panel_outputs(be, PANEL)
    mixer = build.PanelMixer(be, PANEL, min_batch=1)
    mixer.query([Query("counts", (), 2)])
    assert build.panel_max_change(start, mixer.mixed) > 0


def test_ordinary_inference_still_resumes(toy_tokenizer, resume_on):
    srv = ChangingServer()
    be = _backend(toy_tokenizer, srv, {"rev": "a"})
    q = [Query("cheap_talk", (Record("H", 2, 0, "low", 0, 2, 0),), 2)]
    first = be.query(q)[0]
    n = srv.calls
    again = be.query(q)[0]
    assert srv.calls == n and again.replayed and not first.replayed
    assert again.p_H == first.p_H


def test_changed_serving_identity_cannot_reuse_the_cache(toy_tokenizer, resume_on):
    srv = ChangingServer()
    q = [Query("counts", (Record("H", 2, 0),), 2)]
    first = _backend(toy_tokenizer, srv, {"rev": "a", "batch_invariant": True}).query(q)[0]
    n = srv.calls
    other = _backend(toy_tokenizer, srv, {"rev": "a", "batch_invariant": False}).query(q)[0]
    assert srv.calls == n + 1 and not other.replayed and other.p_H != first.p_H


def test_fresh_backend_refuses_replays(toy_tokenizer, resume_on):
    srv = ChangingServer()
    be = build.FreshBackend(_backend(toy_tokenizer, srv, {"rev": "a"}))
    q = [Query("counts", (), 2)]
    be.query(q)
    n = srv.calls
    be.query(q)
    assert srv.calls == n + 1


def test_serving_identity_covers_the_kernel_settings():
    ident = build.serving_identity(build.load("core.yaml"))
    for k in ("revision", "method", "dtype", "engine_kwargs", "vllm_version",
              "batch_invariant", "serving_code_sha256"):
        assert k in ident
