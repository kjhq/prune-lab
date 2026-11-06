"""Teacher top-32 log-probs on the answer tokens of the training pool (offline distillation targets).

Shards of 4096 examples in ~/.cache/prune-lab/data/labels/; 'val' shard = last 512 pool examples.
Resumable: finished shards are skipped. Indices are in the compact (kept) vocab.
"""
import argparse
import time

import numpy as np
import torch

from prunelab.common import DATA, ORIG_VOCAB, TEACHER, load_model, train_arrays, vocab_keep

K = 32
SHARD = 4096
OUT = DATA / "labels"


def batches(lens, idx, budget):
    idx = sorted(idx, key=lambda i: lens[i])
    cur, mx = [], 0
    for i in idx:
        m2 = max(mx, lens[i])
        if cur and m2 * (len(cur) + 1) > budget:
            yield cur
            cur, m2 = [], lens[i]
        cur.append(i)
        mx = m2
    if cur:
        yield cur


@torch.no_grad()
def label(model, old2new, tokens, offsets, plen, ex_ids, budget):
    lens = {i: int(offsets[i + 1] - offsets[i]) for i in ex_ids}
    W = model.lm_head.weight
    res = {}
    for b in batches(lens, ex_ids, budget):
        L = max(lens[i] for i in b)
        ids = torch.zeros(len(b), L, dtype=torch.long)
        att = torch.zeros(len(b), L, dtype=torch.long)
        sel = torch.zeros(len(b), L, dtype=torch.bool)
        for r, i in enumerate(b):
            ids[r, : lens[i]] = torch.from_numpy(tokens[offsets[i] : offsets[i + 1]].astype(np.int64))
            att[r, : lens[i]] = 1
            sel[r, plen[i] - 1 : lens[i] - 1] = True
        ids, att, sel = ids.cuda(), att.cuda(), sel.cuda()
        h = model.model(input_ids=ids, attention_mask=att, use_cache=False).last_hidden_state[sel]
        gold = ids[:, 1:][sel[:, :-1]]
        lps, idxs, nll = [], [], []
        for s in range(0, len(h), 4096):
            lg = (h[s : s + 4096] @ W.T).float()
            lse = torch.logsumexp(lg, -1, keepdim=True)
            v, i = lg.topk(K, -1)
            lp = v - lse
            ni = old2new[i]
            lp = lp.masked_fill(ni < 0, float("-inf"))
            lps.append(lp.half())
            idxs.append(ni.clamp_min(0).to(torch.int32))
            nll.append((lse.squeeze(-1) - lg.gather(-1, gold[s : s + 4096, None]).squeeze(-1)).half())
        lps, idxs, nll = torch.cat(lps).cpu(), torch.cat(idxs).cpu(), torch.cat(nll).cpu()
        o = 0
        for i in b:
            n = lens[i] - plen[i]
            res[i] = (lps[o : o + n], idxs[o : o + n], nll[o : o + n])
            o += n
        assert o == len(lps)
    order = sorted(res)
    counts = np.array([len(res[i][0]) for i in order])
    return {
        "ex_ids": np.array(order, dtype=np.int64),
        "pos_offsets": np.concatenate([[0], np.cumsum(counts)]).astype(np.int64),
        "lp": torch.cat([res[i][0] for i in order]).numpy(),
        "idx": torch.cat([res[i][1] for i in order]).numpy().astype(np.uint16 if old2new.max() < 65535 else np.int32),
        "nll": torch.cat([res[i][2] for i in order]).numpy(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--answer-tokens", type=float, default=25e6, help="stop after this many labelled answer tokens")
    ap.add_argument("--budget", type=int, default=24000, help="padded tokens per teacher forward")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    tokens, offsets, plen = train_arrays()
    n = len(plen)
    keep = vocab_keep()
    old2new = torch.full((ORIG_VOCAB,), -1, dtype=torch.long)
    old2new[torch.from_numpy(keep)] = torch.arange(len(keep))
    old2new = old2new.cuda()
    print(f"pool {n} examples, compact vocab {len(keep)}", flush=True)
    model = load_model(TEACHER)
    ans = (np.diff(offsets) - plen)
    jobs = [("val", list(range(n - 512, n)))]
    done_tokens, k = 0, 0
    while done_tokens < a.answer_tokens and k * SHARD < n - 512:
        ex = list(range(k * SHARD, min((k + 1) * SHARD, n - 512)))
        jobs.append((f"{k:04d}", ex))
        done_tokens += int(ans[ex].sum())
        k += 1
    print(f"{len(jobs) - 1} train shards, {done_tokens:,} answer tokens planned", flush=True)
    t0, tok_done = time.time(), 0
    for name, ex in jobs:
        path = OUT / f"shard_{name}.npz"
        if path.exists():
            continue
        t = time.time()
        out = label(model, old2new, tokens, offsets, plen, ex, a.budget)
        tmp = OUT / f".tmp_{name}.npz"
        np.savez(tmp, **out)
        tmp.replace(path)
        tot = int(offsets[ex[-1] + 1] - offsets[ex[0]])
        tok_done += tot
        el = time.time() - t
        print(f"shard {name}: {len(ex)} ex, {tot:,} tok in {el:.0f}s = {tot / el:,.0f} tok/s; "
              f"teacher gold nll {out['nll'].astype(np.float32).mean():.3f}", flush=True)
    (OUT / "DONE").touch()
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
