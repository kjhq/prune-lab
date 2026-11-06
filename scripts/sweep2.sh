#!/usr/bin/env bash
# Sweep part 2: the Qwen3-1.7B shape. Teacher pruned to Qwen3-1.7B's exact shape vs Qwen3-1.7B itself,
# same labels and token budget as q06. 8-bit AdamW (bitsandbytes) so fp32 weights + grads + optimizer state fit in memory.
# No LR probes (time): LRs carried over from the q06 probes, where 1e-4..3e-4 (pruned) and
# 2e-5..6e-5 (vendor) all landed within 7% of each other. Waits for sweep1 (shared GPU). Idempotent.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
M=~/.cache/prune-lab/models
ANS=${ANS:-15e6}
export PYTORCH_ALLOC_CONF=expandable_segments:True
log() { echo "[$(date +%H:%M:%S)] $*"; }

until [ -f ~/.cache/prune-lab/logs/sweep1.done ]; do sleep 60; done
[ -f $M/random-q17/prunelab.json ] || { log "make q17 students"; $PY -u scripts/make_students.py --shapes q17 --calib code; }
log "eval qwen17 zero-shot"
$PY -u scripts/eval_model.py --model Qwen/Qwen3-1.7B --name qwen17-zeroshot --own-template --thinking false --batch 96

train() {  # micro 1536 without grad checkpointing should fit; fall back to checkpointing if not
  local name=$1; shift
  $PY -u scripts/train.py --name "$name" --answer-tokens "$ANS" --optim adamw8bit --micro 1536 --eval-batch 96 "$@" || {
    log "$name failed at micro 1536; retrying with gradient checkpointing"
    $PY -u scripts/train.py --name "$name" --answer-tokens "$ANS" --optim adamw8bit --micro 4096 --grad-ckpt --eval-batch 96 "$@"; }
}
log "full run pruned q17 lr=2e-4"
train full-pruned-q17 --init $M/pruned-code-q17 --lr 2e-4
log "full run qwen17 lr=4e-5"
train full-qwen17-q17 --init $M/qwen17-vt --lr 4e-5
log "sweep2 done"
touch ~/.cache/prune-lab/logs/sweep2.done
