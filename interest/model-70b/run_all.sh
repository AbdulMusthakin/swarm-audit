#!/usr/bin/env bash
# Llama 3.1 70B, for interest only (not part of the report or submission).
# Runs every GPU step back to back on one deployed B200, so the GPU never idles between
# steps. Prerequisite (CPU only): scripts/setup_workspace.py --config model-70b.yaml.
# Resume after an interruption: SWARM_AUDIT_RESUME=1 bash interest/model-70b/run_all.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
export SWARM_AUDIT_HF_SECRET=huggingface SWARM_AUDIT_BATCH_INVARIANT=1 SWARM_AUDIT_MAX_CONTAINERS=1
C=model-70b.yaml
R=interest/model-70b
T=$R/audit/llama-3.1-70b
step() { echo; echo "=== $(date -u +%H:%M:%S) $*"; }

step deploy;  python scripts/deploy_server.py --config $C
if [ ! -f $R/serving/g3_batchinv1.json ]; then
  step G3;    python scripts/check_serving.py --config $C --batch-invariant
fi
step floor   # the core's settled G3 rule: the chosen setting's maximum is the floor; > 0.07 stops
python - <<'PY'
import json
from swarm_audit import build
r = json.load(open("interest/model-70b/serving/g3_batchinv1.json"))
floor = max(r["max_abs_diff_p_H"], r["max_abs_diff_signal"])
print("G3 floor (batch invariant):", floor)
if floor > 0.07:
    raise SystemExit("G3 floor above 0.07: stop")
build.set_config_line("numerical_floor", repr(float(floor)), configs=("model-70b.yaml",),
                      note="G3 on the B200 (interest/model-70b/serving/g3_batchinv1.json)")
PY
step audit-enum; python scripts/run_audit.py --config $C --channel counts --enumerated
python scripts/run_audit.py --config $C --channel cheap_talk --enumerated
step calibration; python scripts/run_swarm.py --config $C --arm m70_calibration_counts
python scripts/run_swarm.py --config $C --arm m70_calibration_cheap_talk
step audit-fit; python scripts/run_audit.py --config $C --channel counts
python scripts/run_audit.py --config $C --channel cheap_talk
step core;    python scripts/run_swarm.py --config $C --arm m70_core_counts
python scripts/run_swarm.py --config $C --arm m70_core_cheap_talk
step stop;    modal app stop -y swarm-audit-serving-70b || true

step analysis   # CPU; a G5 FAIL is recorded, not fatal here
python scripts/analyse.py --config $C --runs $R/runs/m70_calibration_counts/counts_n4_cycle_last1 \
  --tables $T --tag counts_d2_enumerated --gate-g5 || echo "G5 counts: FAIL (recorded)"
python scripts/analyse.py --config $C --runs $R/runs/m70_calibration_cheap_talk/cheap_talk_n4_cycle_last1 \
  --tables $T --tag cheap_talk_d2_enumerated --gate-g5 || echo "G5 cheap talk: FAIL (recorded)"
python scripts/analyse.py --config $C --runs $R/runs/m70_core_counts/counts_n10_cycle_full --tables $T --tag counts_d2
python scripts/analyse.py --config $C --runs $R/runs/m70_core_cheap_talk/cheap_talk_n10_cycle_full --tables $T --tag cheap_talk_d2
step done
