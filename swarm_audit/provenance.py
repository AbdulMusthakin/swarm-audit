"""Provenance fingerprints for runs and audit tables (review C2).

- The *kernel identity* is everything that can change an agent's output distribution:
  the prompt template files, the backend (model, revision, labels and their token ids,
  chat settings and, for the LLM, the full serving identity), and the payoffs.
- A *run* fingerprint adds the run design (arm, channel, n, graph, memory, horizon,
  seeds, own-signal state) and the sampling protocol. The number of runs is not part of
  it, so a condition can be topped up by a resume.
- A *table* fingerprint adds the audit design (channel, degree, kind, enumerated or
  p_test with its parameters and sizes).

A condition folder's `provenance.json` is written before any computation and never
rewritten; a resume whose fingerprint differs fails, naming the fields that differ.
Analysis refuses tables whose kernel fingerprint differs from the runs'.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

SAMPLING_PROTOCOL = (
    "v1: one numpy PCG64 stream per run, SeedSequence([base_seed, arm_id, condition_id, "
    "run]); agents draw in index order; a = H iff u < p_H; on cheap talk each agent then "
    "draws its signal from P(m | h, a) for its sampled a; vLLM's own sampling is not used")


class ProvenanceError(RuntimeError):
    pass


def canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def fingerprint(obj: Any) -> str:
    return hashlib.sha256(canonical(obj).encode()).hexdigest()


def template_hashes() -> dict[str, str]:
    """sha256 of each prompt template file, read now (comments included)."""
    from . import ladder

    return {step: hashlib.sha256((ladder.PROMPT_DIR / f).read_bytes()).hexdigest()
            for step, f in sorted(ladder._TEMPLATE_FILES.items())}


def kernel_identity(backend: Any) -> dict:
    params = getattr(backend, "params", None)
    return {"templates": template_hashes(), "backend": backend.describe(),
            "payoffs": asdict(params) if params is not None else None}


def run_payload(cond: Any, backend: Any) -> dict:
    design = {k: v for k, v in asdict(cond).items() if k not in ("runs", "params")}
    # Preserve existing cycle/complete fingerprints when merging graph parameters.
    # Nonempty parameters still identify the particular random regular graph.
    if not design.get("graph_kwargs"):
        design.pop("graph_kwargs", None)
    return {"kernel": kernel_identity(backend), "design": design,
            "sampling": SAMPLING_PROTOCOL}


def diff_keys(a: Any, b: Any, prefix: str = "") -> list[str]:
    """Dotted paths where two JSON-like objects differ."""
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b)):
            out += diff_keys(a.get(k), b.get(k), f"{prefix}{k}.")
        return out
    return [] if canonical(a) == canonical(b) else [prefix.rstrip(".") or "(root)"]


def bind_run_provenance(folder: Path, cond: Any, backend: Any, commit: str | None) -> dict:
    """Create the condition's provenance.json before computing, or check the existing one.
    Raises ProvenanceError on a mismatch, or if runs exist without a provenance file."""
    import datetime as dt

    folder = Path(folder)
    payload = run_payload(cond, backend)
    fp = fingerprint(payload)
    path = folder / "provenance.json"
    if path.exists():
        rec = json.loads(path.read_text())
        if rec["fingerprint"] != fp:
            raise ProvenanceError(
                f"{folder}: existing runs were made under a different configuration "
                f"(differs in: {', '.join(diff_keys(rec['payload'], payload))}). Keep them as "
                "superseded and use a new condition or arm, or restore the settings.")
        return rec
    if any(folder.glob("run_*.jsonl.gz")):
        raise ProvenanceError(f"{folder}: runs exist without provenance.json; cannot verify "
                              "their configuration, so they cannot be resumed")
    rec = {"fingerprint": fp, "kernel_fingerprint": fingerprint(payload["kernel"]),
           "payload": payload, "created": dt.datetime.now(dt.timezone.utc).isoformat(),
           "git_commit": commit}
    folder.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec, indent=1, default=str) + "\n")
    return rec


def table_provenance(backend: Any, design: dict) -> dict:
    kernel = kernel_identity(backend)
    return {"kernel": kernel, "kernel_fingerprint": fingerprint(kernel), "design": design,
            "fingerprint": fingerprint({"kernel": kernel, "design": design})}


def check_tables_match_runs(run_kernel_fp: str | None, tables: dict[str, Any]) -> None:
    """Every table must have been fitted under the runs' kernel (prompts, model, labels,
    payoffs, serving)."""
    if run_kernel_fp is None:
        raise ProvenanceError("the runs have no provenance.json: cannot check the tables")
    for name, t in tables.items():
        if t is None:
            continue
        tfp = t.meta.get("provenance", {}).get("kernel_fingerprint")
        if tfp is None:
            raise ProvenanceError(f"table {name!r} has no provenance: refit it")
        if tfp != run_kernel_fp:
            raise ProvenanceError(
                f"table {name!r} was fitted under a different kernel than the runs "
                "(prompts, model, labels, payoffs or serving differ): refit it under the "
                "runs' settings")
