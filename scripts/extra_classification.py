#!/usr/bin/env python3
"""Derived classification of every extra-arm analysis (independent review M2, 8 Nov 2026).

Original analysis JSONs stay unchanged as provenance; results/extra/analysis/
classification.json records the corrected is_primary_contrast (only the core's Llama
arm core_cheap_talk, arm_id 6, is the primary contrast) and each arm's status.

    python scripts/extra_classification.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from swarm_audit.analysis import is_primary_contrast  # noqa: E402

R = REPO / "results" / "extra"
STATUS = {"extra_rr4": "own arm (scale and topology demonstration), not a primary contrast",
          "olmo_calibration_cheap_talk": "EXPLORATORY (arm 9a; trigger not met, run by Abdul's decision after seeing results)",
          "olmo_core_cheap_talk": "EXPLORATORY (arm 9b; trigger not met, run by Abdul's decision after seeing results)",
          "followup_olmo_n10_last1": "EXPLORATORY overnight follow-up E3 (arm_id 92)",
          "followup_olmo_n4_full": "EXPLORATORY overnight follow-up E3 (arm_id 93)"}


def main() -> None:
    out = {"what": "Derived classification of every extra-arm analysis (independent review M2, 8 Nov 2026). "
                   "Original analysis JSONs are kept unchanged as provenance; where their stored "
                   "is_primary_contrast differs from the corrected value, this file governs.",
           "rule": "swarm_audit.analysis.is_primary_contrast: only the core's Llama arm core_cheap_talk "
                   "(arm_id 6) is the primary contrast",
           "files": {}}
    files = sorted(list((R / "analysis").rglob("*.json")) + list((R / "followup" / "analysis").rglob("*.json")))
    for f in files:
        if f.name.endswith("_G5.json") or f.name == "classification.json":
            continue
        j = json.loads(f.read_text())
        arm = f.parent.name
        runs = R / "runs" / arm / f.stem
        if not runs.exists():
            runs = R / "followup" / "runs" / arm / f.stem
        prov = runs / "provenance.json"
        design = json.loads(prov.read_text())["payload"]["design"] if prov.exists() else {}
        c = j["condition"]
        out["files"][str(f.relative_to(REPO))] = {
            "arm": design.get("arm"), "arm_id": design.get("arm_id"),
            "stored_is_primary_contrast": j.get("is_primary_contrast"),
            "corrected_is_primary_contrast": is_primary_contrast(c["channel"], c["n"], c["graph"], design),
            "status": STATUS.get(arm, "unclassified"),
            "inherited_primary_notes_apply": False}
    (R / "analysis" / "classification.json").write_text(json.dumps(out, indent=1) + "\n")
    for k, v in out["files"].items():
        print(k, v["stored_is_primary_contrast"], "->", v["corrected_is_primary_contrast"])


if __name__ == "__main__":
    main()
