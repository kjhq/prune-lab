"""Full-parameter training of a student on the coding task by distillation from offline teacher labels.

Every weight trains (no LoRA, nothing frozen). Loss per answer token:
  top-32 KL(teacher || student) with an exact "rest" bucket  +  ce_weight * CE(reference solution token)
fp32 master weights, bf16 autocast, fused AdamW, cosine schedule. Resumable from ckpt.pt.
At the end: saves final/ (bf16) and runs the sandboxed code eval.
"""
import argparse
import json
import math
import random
import shutil
import time
from collections import OrderedDict

import numpy as np
import torch

from prunelab.common import DATA, ORIG_VOCAB, RESULTS, RUNS, VocabMap, load_tokenizer, train_arrays

LABELS = DATA / "labels"


class Shards:
    def __init__(self, cache=3):
        self.c, self.n = OrderedDict(), cache

    def get(self, name):
        if name not in self.c:
            z = np.load(LABELS / f"shard_{name}.npz")
            self.c[name] = {k: z[k] for k in z.files}
            if len(self.c) > self.n:
                self.c.popitem(last=False)
        self.c.move_to_end(name)
        return self.c[name]


def make_plan(names, offsets, plen, answer_budget, micro, step_tokens, seed):
    """List of steps; step = list of micro-batches; micro-batch = (shard, [row indices in shard])."""
    rng = random.Random(seed)
    lens = np.diff(offsets)
    steps, cur, cur_tok, total, epoch = [], [], 0, 0, 0
    while total < answer_budget:
        order = names[:]
        rng.shuffle(order)
        for name in order:
            z = np.load(LABELS / f"shard_{name}.npz")
            ex = z["ex_ids"]
            rows = sorted(range(len(ex)), key=lambda r: lens[ex[r]])
            mbs, mb, mx = [], [], 0
            for r in rows:
                m2 = max(mx, lens[ex[r]])
                if mb and m2 * (len(mb) + 1) > micro:
                    mbs.append(mb)
                    mb, m2 = [], lens[ex[r]]
                mb.append(r)
                mx = m2
            if mb:
                mbs.append(mb)
            rng.shuffle(mbs)
            for mb in mbs:
                t = int(sum(lens[ex[r]] - plen[ex[r]] for r in mb))
                cur.append((name, mb))
                cur_tok += t
                total += t
                if cur_tok >= step_tokens:
                    steps.append(cur)
                    cur, cur_tok = [], 0
                if total >= answer_budget:
                    break
            if total >= answer_budget:
                break
        epoch += 1
    if cur:
        steps.append(cur)
    return steps, epoch


def batch_tensors(shard, rows, tokens, offsets, plen, old2new):
    ex = shard["ex_ids"]
    po = shard["pos_offsets"]
    lens = [int(offsets[ex[r] + 1] - offsets[ex[r]]) for r in rows]
    L = max(lens)
    ids = np.zeros((len(rows), L), dtype=np.int64)
    att = np.zeros((len(rows), L), dtype=np.int64)
    sel = np.zeros((len(rows), L), dtype=bool)
    lp, ix = [], []
    for k, r in enumerate(rows):
        e = ex[r]
        ids[k, : lens[k]] = old2new[tokens[offsets[e] : offsets[e + 1]].astype(np.int64)]
        att[k, : lens[k]] = 1
        sel[k, plen[e] - 1 : lens[k] - 1] = True
        lp.append(shard["lp"][po[r] : po[r + 1]])
        ix.append(shard["idx"][po[r] : po[r + 1]])
    ids_t = torch.from_numpy(ids).cuda(non_blocking=True)
    sel_t = torch.from_numpy(sel).cuda(non_blocking=True)
    gold = ids_t[:, 1:][sel_t[:, :-1]]
    lp = torch.from_numpy(np.concatenate(lp)).cuda(non_blocking=True).float()
    ix = torch.from_numpy(np.concatenate(ix).astype(np.int64)).cuda(non_blocking=True)
    assert len(gold) == len(lp), (len(gold), len(lp))
    return ids_t, torch.from_numpy(att).cuda(non_blocking=True), sel_t, gold, lp, ix


def chunk_loss(lg, t_lp, t_idx, gold, ce_w):
    """lg: (n, V) fp32 student logits. Returns per-position (kl, ce) and student top-1 == teacher top-1."""
    lse = torch.logsumexp(lg, -1)
    valid = torch.isfinite(t_lp)
    idx = torch.where(valid, t_idx, t_idx[:, :1])
    sl = lg.gather(-1, idx) - lse[:, None]
    tp = t_lp.exp()
    kl_top = torch.where(valid, tp * (t_lp - sl), torch.zeros_like(sl)).sum(-1)
    t_rest = (1 - tp.sum(-1)).clamp_min(1e-6)
    s_rest = torch.logsumexp(lg.scatter(-1, idx, float("-inf")), -1) - lse
    kl = kl_top + t_rest * (t_rest.log() - s_rest)
    ce = lse - lg.gather(-1, gold[:, None]).squeeze(-1)
    agree = lg.argmax(-1) == t_idx[:, 0]
    return kl, ce, agree


def loss_backward(model, h, t_lp, t_idx, gold, ce_w, scale, chunk=2048):
    hd = h.detach().requires_grad_()
    W = model.model.embed_tokens.weight
    kl_s = ce_s = 0.0
    for s in range(0, len(hd), chunk):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = hd[s : s + chunk] @ W.T
        kl, ce, _ = chunk_loss(lg.float(), t_lp[s : s + chunk], t_idx[s : s + chunk], gold[s : s + chunk], ce_w)
        ((kl.sum() + ce_w * ce.sum()) * scale).backward()
        kl_s += float(kl.detach().sum())
        ce_s += float(ce.detach().sum())
    h.backward(hd.grad)
    return kl_s, ce_s


@torch.no_grad()
def validate(model, shard, tokens, offsets, plen, old2new, micro):
    model.eval()
    lens = np.diff(offsets)
    ex = shard["ex_ids"]
    rows = sorted(range(len(ex)), key=lambda r: lens[ex[r]])
    kl_s = ce_s = ag_s = n = 0
    mb, mx = [], 0
    groups = []
    for r in rows:
        m2 = max(mx, lens[ex[r]])
        if mb and m2 * (len(mb) + 1) > micro:
            groups.append(mb)
            mb, m2 = [], lens[ex[r]]
        mb.append(r)
        mx = m2
    groups.append(mb)
    W = model.model.embed_tokens.weight
    for g in groups:
        ids, att, sel, gold, lp, ix = batch_tensors(shard, g, tokens, offsets, plen, old2new)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = model.model(input_ids=ids, attention_mask=att, use_cache=False).last_hidden_state[sel]
        for s in range(0, len(h), 2048):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = h[s : s + 2048] @ W.T
            kl, ce, ag = chunk_loss(lg.float(), lp[s : s + 2048], ix[s : s + 2048], gold[s : s + 2048], 0)
            kl_s += float(kl.sum())
            ce_s += float(ce.sum())
            ag_s += float(ag.sum())
            n += len(kl)
    model.train()
    return {"val_kl": kl_s / n, "val_ce": ce_s / n, "val_agree": ag_s / n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--answer-tokens", type=float, default=25e6)
    ap.add_argument("--step-tokens", type=int, default=32768, help="answer tokens per optimizer step")
    ap.add_argument("--micro", type=int, default=4096, help="padded tokens per micro-batch (8192 runs out of memory: ~3 MB activations/token)")
    ap.add_argument("--ce-weight", type=float, default=0.5)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--val-every", type=int, default=100)
    ap.add_argument("--ckpt-min", type=float, default=10)
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--no-eval", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--optim", choices=["adamw", "adamw8bit"], default="adamw",
                    help="adamw8bit (bitsandbytes) keeps optimizer state at ~2 bytes/param; needed for the 1.7B shape")
    ap.add_argument("--eval-batch", type=int, default=128)
    a = ap.parse_args()

    run = RUNS / a.name
    run.mkdir(parents=True, exist_ok=True)
    if (run / "DONE").exists():
        print("already done")
        return
    (run / "args.json").write_text(json.dumps(vars(a), indent=1))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(a.seed)

    from transformers import AutoModelForCausalLM

    if (run / "final" / "vocab_keep.npy").exists():  # training finished earlier; only the eval is missing
        print("final/ exists, running eval only", flush=True)
        model = AutoModelForCausalLM.from_pretrained(run / "final", dtype=torch.bfloat16, attn_implementation="sdpa").cuda()
        finish(model, np.load(run / "final" / "vocab_keep.npy"), a, run)
        return
    model = AutoModelForCausalLM.from_pretrained(a.init, dtype=torch.float32, attn_implementation="sdpa").cuda()
    if a.grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    model.train()
    keep = np.load(f"{a.init}/vocab_keep.npy")
    old2new = np.full(ORIG_VOCAB, 0, dtype=np.int64)
    old2new[keep] = np.arange(len(keep))
    tokens, offsets, plen = train_arrays()
    names = sorted(p.stem.removeprefix("shard_") for p in LABELS.glob("shard_*.npz") if "val" not in p.stem)
    steps, epochs = make_plan(names, offsets, plen, a.answer_tokens, a.micro, a.step_tokens, a.seed)
    n_steps = len(steps)
    warm = max(10, int(a.warmup * n_steps))
    print(f"{a.name}: {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M params, {n_steps} steps, "
          f"{len(names)} label shards, {epochs} epoch(s)", flush=True)

    decay = [p for n, p in model.named_parameters() if p.ndim >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.ndim < 2]
    groups = [{"params": decay, "weight_decay": a.wd}, {"params": no_decay, "weight_decay": 0.0}]
    if a.optim == "adamw8bit":
        import bitsandbytes as bnb

        opt = bnb.optim.AdamW8bit(groups, lr=a.lr, betas=(0.9, 0.95), eps=1e-8)
    else:
        opt = torch.optim.AdamW(groups, lr=a.lr, betas=(0.9, 0.95), eps=1e-8, fused=True)

    def lr_at(s):
        if s < warm:
            return a.lr * (s + 1) / warm
        p = (s - warm) / max(1, n_steps - warm)
        return a.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))

    start = 0
    ck = run / "ckpt.pt"
    if ck.exists():
        state = torch.load(ck, map_location="cuda", weights_only=False)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        start = state["step"] + 1
        print(f"resumed at step {start}", flush=True)

    def save_ckpt(step):
        tmp = run / "ckpt.tmp"
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step}, tmp)
        tmp.replace(ck)

    shards = Shards()
    val_shard = dict(np.load(LABELS / "shard_val.npz"))
    log = open(run / "log.jsonl", "a")
    if start == 0:
        v = validate(model, val_shard, tokens, offsets, plen, old2new, a.micro)
        rec = {"step": 0, **v}
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(rec, flush=True)
    t_last_ck = time.time()
    t0, tok_win, ans_win = time.time(), 0, 0
    for step in range(start, n_steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        plan = steps[step]
        ans_total = 0
        mbs = []
        for name, rows in plan:
            sh = shards.get(name)
            mbs.append(batch_tensors(sh, rows, tokens, offsets, plen, old2new))
            ans_total += len(mbs[-1][3])
        kl_s = ce_s = 0.0
        for ids, att, sel, gold, lp, ix in mbs:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                h = model.model(input_ids=ids, attention_mask=att, use_cache=False).last_hidden_state[sel]
            k, c = loss_backward(model, h, lp, ix, gold, a.ce_weight, 1.0 / ans_total)
            kl_s += k
            ce_s += c
            tok_win += int(att.sum())
        gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        opt.step()
        opt.zero_grad(set_to_none=True)
        ans_win += ans_total
        if step % 10 == 0 or step == n_steps - 1:
            torch.cuda.synchronize()
            el = time.time() - t0
            rec = {"step": step, "lr": lr_at(step), "kl": kl_s / ans_total, "ce": ce_s / ans_total, "gnorm": gn,
                   "tok_s": tok_win / el, "ans_tok_s": ans_win / el,
                   "mem_gb": torch.cuda.max_memory_allocated() / 2**30}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()}), flush=True)
            t0, tok_win, ans_win = time.time(), 0, 0
        if (step + 1) % a.val_every == 0 or step == n_steps - 1:
            v = validate(model, val_shard, tokens, offsets, plen, old2new, a.micro)
            rec = {"step": step + 1, **v}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(rec, flush=True)
            t0, tok_win, ans_win = time.time(), 0, 0
        if time.time() - t_last_ck > a.ckpt_min * 60 and step < n_steps - 1:
            save_ckpt(step)
            t_last_ck = time.time()
    final = run / "final"
    del opt
    model.zero_grad(set_to_none=True)
    model.to(torch.bfloat16).save_pretrained(final)
    torch.cuda.empty_cache()
    shutil.copy(f"{a.init}/vocab_keep.npy", final / "vocab_keep.npy")
    ck.unlink(missing_ok=True)
    finish(model, keep, a, run)


def finish(model, keep, a, run):
    if not a.no_eval:
        from prunelab.codeeval import evaluate

        model.eval()
        model.config.use_cache = True
        tok = load_tokenizer()
        s = evaluate(model, tok, RESULTS / "eval" / a.name, vmap=VocabMap(keep, tok), batch_size=a.eval_batch)
        print(a.name, json.dumps(s), flush=True)
    (run / "DONE").touch()


if __name__ == "__main__":
    main()
