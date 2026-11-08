#!/usr/bin/env python3
"""Deploy the vLLM server as a persistent Modal app, so consecutive steps reuse a warm GPU.

An app started with `app.run()` stops when its script exits, GPU included, and the next
script queues for a new L4 (10-20 minutes on this account). The deployed app outlives
each script; its GPU container stays up for serving.scaledown_window_s (600 s) after the
last call. With serving.deployed: true, every GPU script uses it and first checks that
it runs this repository's serving code and settings (results/serving/deployed.json).

    SWARM_AUDIT_HF_SECRET=huggingface SWARM_AUDIT_BATCH_INVARIANT=1 \
        python scripts/deploy_server.py

Redeploy after any change to swarm_audit/{serving,tokens,ladder,game}.py or the serving
settings. Stop it when the build is done:

    modal app stop swarm-audit-serving
"""

from __future__ import annotations

import datetime as dt
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from swarm_audit import build  # noqa: E402


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="core.yaml")
    cfg = build.load(ap.parse_args().config)
    srv = cfg["serving"]
    want = {"SWARM_AUDIT_HF_SECRET": "huggingface",
            "SWARM_AUDIT_BATCH_INVARIANT": "1" if srv["batch_invariant"] else "0"}
    bad = [f"{k}={v}" for k, v in want.items() if os.environ.get(k, "0" if "INVARIANT" in k else "") != v]
    if bad:
        raise SystemExit("set " + " ".join(bad) + " before deploying")
    os.environ["SWARM_AUDIT_SCALEDOWN_S"] = str(srv["scaledown_window_s"])
    os.environ["SWARM_AUDIT_MAX_CONTAINERS"] = str(srv["max_containers"])
    os.environ["SWARM_AUDIT_TIMEOUT_MIN"] = str(srv["timeout_minutes"])
    os.environ["SWARM_AUDIT_VLLM_VERSION"] = srv["vllm_version"]
    os.environ["SWARM_AUDIT_GPU"] = srv["gpu"]
    ws = build.check_workspace(cfg)
    build.apply_serving_env(cfg)
    import modal

    from swarm_audit import serving

    with modal.enable_output():
        serving.app.deploy()
    rec = {"app": serving.APP_NAME, "workspace": ws, "code_sha256": build.server_code_sha(),
           "settings": build.deploy_settings(cfg), "git_commit": build_commit(),
           "time": dt.datetime.now(dt.timezone.utc).isoformat()}
    out = build.deploy_file(cfg)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps(rec, indent=1))


def build_commit() -> str | None:
    from swarm_audit.swarm import git_commit

    return git_commit()


if __name__ == "__main__":
    main()
