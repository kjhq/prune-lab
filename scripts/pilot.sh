#!/usr/bin/env bash
# Pilot: 4B -> Qwen3-0.6B shape. Every step is idempotent; rerun to resume.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
M=~/.cache/prune-lab/models
ANS=${ANS:-15e6}
PROBE=${PROBE:-1.5e6}
log() { echo "[$(date +%H:%M:%S)] $*"; }

[ -f $M/random-q06/prunelab.json ] || { log "make students"; $PY -u scripts/make_students.py --shapes q06 --calib code,web; }
log "label"; $PY -u scripts/label.py --answer-tokens $ANS
log "baseline evals"
$PY -u scripts/eval_model.py --model Qwen/Qwen3-4B-Instruct-2507 --name teacher-4b
$PY -u scripts/eval_model.py --model Qwen/Qwen3-0.6B --name qwen06-zeroshot --own-template --thinking false --batch 128
$PY -u scripts/eval_model.py --model $M/pruned-code-q06 --name pruned-code-q06-untrained --batch 128
declare -A GRID=( [pruned]="1e-4 3e-4 1e-3" [qwen06]="2e-5 6e-5 2e-4" [random]="3e-4 1e-3 3e-3" )
declare -A INIT=( [pruned]=$M/pruned-code-q06 [qwen06]=$M/qwen06-vt [random]=$M/random-q06 )
for arm in pruned qwen06 random; do
  for lr in ${GRID[$arm]}; do
    log "probe $arm $lr"
    $PY -u scripts/train.py --init ${INIT[$arm]} --name probe-$arm-$lr --lr $lr --answer-tokens $PROBE --no-eval --val-every 100000
  done
done
for arm in pruned qwen06 random; do
  lr=$($PY scripts/pick_lr.py $arm ${GRID[$arm]})
  log "full run $arm lr=$lr"
  $PY -u scripts/train.py --init ${INIT[$arm]} --name full-$arm-q06 --lr $lr --answer-tokens $ANS
done
log "pilot done"
touch ~/.cache/prune-lab/logs/pilot.done
