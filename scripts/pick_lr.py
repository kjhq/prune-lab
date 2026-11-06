"""Print the LR whose probe run reached the lowest final val loss (val_kl + 0.5 * val_ce)."""
import json
import sys

from prunelab.common import RUNS

arm, lrs = sys.argv[1], sys.argv[2:]
best = None
for lr in lrs:
    recs = [json.loads(l) for l in open(RUNS / f"probe-{arm}-{lr}" / "log.jsonl") if "val_kl" in l]
    v = recs[-1]["val_kl"] + 0.5 * recs[-1]["val_ce"]
    print(f"{arm} lr={lr}: val {v:.4f}", file=sys.stderr)
    if best is None or v < best[0]:
        best = (v, lr)
print(best[1])
