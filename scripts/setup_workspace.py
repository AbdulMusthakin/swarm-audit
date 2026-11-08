#!/usr/bin/env python3
"""One-time set-up of a teammate's Modal workspace for the build (Abdul, 7 Nov 2026).

GPU runs use a teammate's Modal workspace, never Abdul's. Before any gate:
1. Sign in to the teammate's workspace: `python scripts/modal_login.py` (the teammate
   opens the printed link while signed in to their own Modal account).
2. In that workspace, create a Modal secret named "huggingface" holding HF_TOKEN, a
   Hugging Face token whose account has accepted the Llama 3.1 licence.
3. Run, with the secret's name in the environment:

       SWARM_AUDIT_HF_SECRET=huggingface python scripts/setup_workspace.py

It checks the workspace (never one in inputs.forbidden_workspaces) and the secret,
downloads the Llama weights once into the workspace's `swarm-audit-weights` volume (a
CPU container; this also proves the token has Llama access), fetches the tokenizer, and
checks on the final prompts that every label is still one token with the configured id.
No GPU is used. Writes results/serving/workspace_setup.json.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from swarm_audit import build  # noqa: E402


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="core.yaml")
    cfg = build.load(ap.parse_args().config)
    m = cfg["model"]
    if os.environ.get("SWARM_AUDIT_HF_SECRET") != "huggingface":
        raise SystemExit("set SWARM_AUDIT_HF_SECRET=huggingface (the secret's name in the workspace)")
    ws = build.check_workspace(cfg)
    import modal

    try:
        modal.Secret.from_name("huggingface").hydrate()
    except Exception as e:  # NotFoundError and friends
        raise SystemExit(f"no Modal secret 'huggingface' in workspace {ws!r}: create it with "
                         f"HF_TOKEN for an account with Llama 3.1 access ({e})")
    from swarm_audit import serving
    from swarm_audit.screening import fetch_tokenizer_via_modal, load_tokenizer

    t0 = time.time()
    with modal.enable_output(), serving.app.run():
        dl = serving.download_weights.remote(m["hf_id"], m["revision"])
        fetch_tokenizer_via_modal(m)
    tok = load_tokenizer(m)
    labels = build.check_final_labels(tok, cfg)
    out = {"workspace": ws, "secret": "huggingface", "model": m["hf_id"], "revision": m["revision"],
           "weights": dl, "seconds": round(time.time() - t0, 1), "label_check": labels}
    path = build.results_root(cfg) / "serving" / "workspace_setup.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1, default=str) + "\n")
    print(json.dumps(out, indent=1, default=str))
    if not labels["pass"]:
        raise SystemExit("label check FAILED on the final prompts: stop and report to Abdul")


if __name__ == "__main__":
    main()
