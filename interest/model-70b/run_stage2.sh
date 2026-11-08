#!/usr/bin/env bash
# Stage 2 of the 70B runs (for interest only): n = 20 on the cycle and K_10 (arm 7, d = 9,
# actions-only tables), both channels. Run after run_all.sh. Every launch checks the cap.
set -euo pipefail
cd "$(dirname "$0")/../.."
export SWARM_AUDIT_HF_SECRET=huggingface SWARM_AUDIT_BATCH_INVARIANT=1 SWARM_AUDIT_MAX_CONTAINERS=1
C=model-70b.yaml
R=interest/model-70b
T=$R/audit/llama-3.1-70b
step() { echo; echo "=== $(date -u +%H:%M:%S) $*"; }

step deploy;  python scripts/deploy_server.py --config $C
step n20;     python scripts/run_swarm.py --config $C --arm m70_core_counts_n20
python scripts/run_swarm.py --config $C --arm m70_core_cheap_talk_n20
step K10-audit; python scripts/run_audit.py --config $C --channel counts --d 9 --actions-only
python scripts/run_audit.py --config $C --channel cheap_talk --d 9 --actions-only
step K10;     python scripts/run_swarm.py --config $C --arm m70_second_topology_counts
python scripts/run_swarm.py --config $C --arm m70_second_topology_cheap_talk
step stop;    modal app stop -y swarm-audit-serving-70b || true

step analysis
python scripts/analyse.py --config $C --runs $R/runs/m70_core_counts_n20/counts_n20_cycle_full --tables $T --tag counts_d2
python scripts/analyse.py --config $C --runs $R/runs/m70_core_cheap_talk_n20/cheap_talk_n20_cycle_full --tables $T --tag cheap_talk_d2
python scripts/analyse.py --config $C --runs $R/runs/m70_second_topology_counts/counts_n10_complete_full --tables $T --tag counts_d9_actions
python scripts/analyse.py --config $C --runs $R/runs/m70_second_topology_cheap_talk/cheap_talk_n10_complete_full --tables $T --tag cheap_talk_d9_actions
step done
