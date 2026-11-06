#!/usr/bin/env bash
# Sweep part 1: calibration ablation (web-calibrated prune, q06 shape) + "tiny" shape (pruned vs random).
# Same labels and token budget as the pilot. Every step is idempotent; rerun to resume.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
M=~/.cache/prune-lab/models
ANS=${ANS:-15e6}
PROBE=${PROBE:-1.5e6}
export PYTORCH_ALLOC_CONF=expandable_segments:True  # generation KV cache grows by concatenation and fragments memory
log() { echo "[$(date +%H:%M:%S)] $*"; }

log "full run pruned-web q06 lr=3e-4 (pilot's best LR for the code-calibrated prune)"
$PY -u scripts/train.py --init $M/pruned-web-q06 --name full-prunedweb-q06 --lr 3e-4 --answer-tokens $ANS
[ -f $M/random-tiny/prunelab.json ] || { log "make tiny students"; $PY -u scripts/make_students.py --shapes tiny --calib code; }
for lr in 3e-4 6e-4 1.2e-3; do
  log "probe tiny $lr"
  $PY -u scripts/train.py --init $M/pruned-code-tiny --name probe-tiny-$lr --lr $lr --answer-tokens $PROBE --no-eval --val-every 100000
done
lr=$($PY scripts/pick_lr.py tiny 3e-4 6e-4 1.2e-3)
log "full run pruned tiny lr=$lr"
$PY -u scripts/train.py --init $M/pruned-code-tiny --name full-pruned-tiny --lr $lr --answer-tokens $ANS
log "full run random tiny lr=1e-3"
$PY -u scripts/train.py --init $M/random-tiny --name full-random-tiny --lr 1e-3 --answer-tokens $ANS
log "sweep1 done"
touch ~/.cache/prune-lab/logs/sweep1.done
