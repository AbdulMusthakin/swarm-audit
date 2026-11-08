#!/usr/bin/env python3
"""Figure 1 of every analysed condition, with the Dobrushin contraction bounds added
(results/followup/dobrushin/README.md): the TV contraction bounds beside Theorem 1, and the
local bound on the mean as a band around the actions-only audit's expected X_t.

    python scripts/plot_contraction.py --preset m70     # 70B, interest/model-70b/figures
    python scripts/plot_contraction.py --preset core    # 8B, writes results/figures/

CPU only; reads the analysis JSONs, the run logs and the audit tables. The figures are
written as fig1_<arm>_<condition>_contraction.png beside the existing fig1 files, which
are left as they are.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from analyse_dobrushin import PRESETS, analyse  # noqa: E402
from swarm_audit import plots  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", choices=sorted(PRESETS), required=True)
    args = ap.parse_args()
    pre = PRESETS[args.preset]
    root = REPO / pre["root"]
    figs = root.parent / "figures"
    rng = np.random.default_rng(20261108)   # the same seed and order as analyse_dobrushin.py
    for arm, name in pre["specs"]:
        path = root / arm / f"{name}.json"
        res = json.loads(path.read_text())
        con = analyse(REPO, path, rng)
        out = figs / f"fig1_{arm}_{name}_contraction.png"
        plots.figure1(res, out, f"{arm}: {name} ({pre['label']})", contraction=con)
        print(f"wrote {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
