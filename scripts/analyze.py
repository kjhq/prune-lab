"""Look inside the teacher: what does it use for Python-coding vs general web text?

Per dataset (code = KodCode prompt+solution, web = FineWeb-Edu, same token budget):
  hidden channel importance  mean |RMSNorm output| per channel, per layer (attn-in and mlp-in)
  residual max |x| per channel per layer (massive-activation outliers)
  MLP neuron importance       mean |act| * ||W_down[:, i]||
  attention head importance   mean ||W_o[:, head] @ head_out||
  block influence             mean (1 - cos(layer in, layer out))
  drop-one-layer KL           top-64 KL(full || model without layer l), code: answer tokens only
  top contexts                strongest activations of the most code-specific / web-specific neurons
Writes ~/.cache/prune-lab/results/analysis/{code,web}.npz, contexts.json, summary.json
"""
import json
import time

import numpy as np
import torch

from prunelab.common import DATA, RESULTS, TEACHER, load_model, load_tokenizer

OUT = RESULTS / "analysis"
TOK_BUDGET = 12_000  # tokens per forward batch
KL_SEQS = 96


def batches(seqs, budget=TOK_BUDGET):
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
    cur, mx = [], 0
    for i in order:
        mx2 = max(mx, len(seqs[i]))
        if cur and mx2 * (len(cur) + 1) > budget:
            yield cur
            cur, mx2 = [], len(seqs[i])
        cur.append(i)
        mx = mx2
    if cur:
        yield cur


def pad(seqs, idx, dev):
    L = max(len(seqs[i]) for i in idx)
    ids = torch.zeros(len(idx), L, dtype=torch.long)
    m = torch.zeros(len(idx), L, dtype=torch.bool)
    for r, i in enumerate(idx):
        ids[r, : len(seqs[i])] = torch.as_tensor(seqs[i])
        m[r, : len(seqs[i])] = True
    return ids.to(dev), m.to(dev)


class Stats:
    def __init__(self, model):
        c = model.config
        self.L, self.d, self.f, self.nq, self.hd = (c.num_hidden_layers, c.hidden_size, c.intermediate_size,
                                                     c.num_attention_heads, c.head_dim)
        z = lambda *s: torch.zeros(*s, dtype=torch.float64, device="cuda")
        self.attn_in, self.mlp_in = z(self.L, self.d), z(self.L, self.d)
        self.resid_max = torch.zeros(self.L, self.d, device="cuda")
        self.ffn = z(self.L, self.f)
        self.head = z(self.L, self.nq)
        self.bi = z(self.L)
        self.n = 0
        self.mask = None
        self.hooks = []
        self.Wo = [l.self_attn.o_proj.weight.view(self.d, self.nq, self.hd) for l in model.model.layers]
        for i, layer in enumerate(model.model.layers):
            self.hooks += [
                layer.input_layernorm.register_forward_hook(self._norm(self.attn_in, i)),
                layer.post_attention_layernorm.register_forward_hook(self._norm(self.mlp_in, i)),
                layer.mlp.down_proj.register_forward_pre_hook(self._ffn(i)),
                layer.self_attn.o_proj.register_forward_pre_hook(self._head(i)),
                layer.register_forward_hook(self._block(i)),
            ]

    def _sel(self, x):
        return x[self.mask].float()

    def _norm(self, acc, i):
        def h(mod, args, out):
            acc[i] += self._sel(out).abs().sum(0).double()
        return h

    def _ffn(self, i):
        def h(mod, args):
            self.ffn[i] += self._sel(args[0]).abs().sum(0).double()
        return h

    def _head(self, i):
        def h(mod, args):
            x = args[0][self.mask].view(-1, self.nq, self.hd)
            for s in range(0, x.shape[0], 2048):
                c = torch.einsum("nqh,dqh->nqd", x[s : s + 2048], self.Wo[i])
                self.head[i] += c.float().norm(dim=-1).sum(0).double()
        return h

    def _block(self, i):
        def h(mod, args, out):
            a, b = self._sel(args[0]), self._sel(out)
            self.bi[i] += (1 - torch.nn.functional.cosine_similarity(a, b, dim=-1)).sum().double()
            self.resid_max[i] = torch.maximum(self.resid_max[i], b.abs().amax(0))
        return h

    def remove(self):
        for h in self.hooks:
            h.remove()


@torch.no_grad()
def collect(model, seqs):
    st = Stats(model)
    for idx in batches(seqs):
        ids, m = pad(seqs, idx, "cuda")
        st.mask = m
        st.n += int(m.sum())
        model.model(input_ids=ids, attention_mask=m.long(), use_cache=False)
    st.remove()
    wdown = torch.stack([l.mlp.down_proj.weight.float().norm(dim=0) for l in model.model.layers])
    n = st.n
    return {
        "attn_in": (st.attn_in / n).float().cpu().numpy(), "mlp_in": (st.mlp_in / n).float().cpu().numpy(),
        "resid_max": st.resid_max.cpu().numpy(), "ffn_act": (st.ffn / n).float().cpu().numpy(),
        "ffn_imp": ((st.ffn / n).float() * wdown).cpu().numpy(), "head_imp": (st.head / n).float().cpu().numpy(),
        "block_influence": (st.bi / n).float().cpu().numpy(), "tokens": n,
    }


@torch.no_grad()
def layer_drop_kl(model, seqs, loss_from):
    """loss_from[i]: first position whose next-token prediction counts (prompt_len-1 for code, 0 for web)."""
    L = model.config.num_hidden_layers
    W = model.lm_head.weight
    batches_ = list(batches(seqs, 8000))

    def run(drop):
        hk = None
        if drop is not None:
            hk = model.model.layers[drop].register_forward_hook(lambda mod, a, o: a[0])
        outs = []
        for idx in batches_:
            ids, m = pad(seqs, idx, "cuda")
            h = model.model(input_ids=ids, attention_mask=m.long(), use_cache=False).last_hidden_state
            sel = torch.zeros_like(m)
            for r, i in enumerate(idx):
                sel[r, loss_from[i] : len(seqs[i]) - 1] = True
            outs.append(h[sel])
        if hk:
            hk.remove()
        return torch.cat(outs)

    def logp(h):
        return torch.cat([torch.log_softmax((h[s : s + 1024] @ W.T).float(), -1) for s in range(0, len(h), 1024)])

    h_ref = run(None)
    # reference top-64 + rest bucket
    ref_lp, ref_idx = [], []
    for s in range(0, len(h_ref), 1024):
        lp = torch.log_softmax((h_ref[s : s + 1024] @ W.T).float(), -1)
        v, i = lp.topk(64, -1)
        ref_lp.append(v)
        ref_idx.append(i)
    ref_lp, ref_idx = torch.cat(ref_lp), torch.cat(ref_idx)
    tp = ref_lp.exp()
    t_rest = (1 - tp.sum(-1)).clamp_min(1e-12)
    kl, agree = np.zeros(L), np.zeros(L)
    for l in range(L):
        h = run(l)
        k_tot, a_tot = 0.0, 0.0
        for s in range(0, len(h), 1024):
            lp = torch.log_softmax((h[s : s + 1024] @ W.T).float(), -1)
            sl = lp.gather(-1, ref_idx[s : s + 1024])
            s_rest = (1 - sl.exp().sum(-1)).clamp_min(1e-12)
            k = (tp[s : s + 1024] * (ref_lp[s : s + 1024] - sl)).sum(-1) + t_rest[s : s + 1024] * (
                t_rest[s : s + 1024].log() - s_rest.log())
            k_tot += float(k.sum())
            a_tot += float((lp.argmax(-1) == ref_idx[s : s + 1024, 0]).sum())
        kl[l], agree[l] = k_tot / len(h), a_tot / len(h)
        print(f"  drop layer {l:2d}: KL {kl[l]:.4f}  top1 agree {agree[l]:.3f}", flush=True)
    return kl, agree


@torch.no_grad()
def top_contexts(model, tok, seqs, neurons, k=8, win=14):
    """neurons: list of (layer, idx). Returns strongest activations with token context."""
    best = {n: [] for n in neurons}
    by_layer = {}
    for l, i in neurons:
        by_layer.setdefault(l, []).append(i)
    cur = {}
    hooks = []
    for l, idxs in by_layer.items():
        t = torch.tensor(idxs, device="cuda")

        def h(mod, args, l=l, t=t):
            cur[l] = args[0][..., t].float()
        hooks.append(model.model.layers[l].mlp.down_proj.register_forward_pre_hook(h))
    last = max(by_layer) + 1
    for idx in batches(seqs):
        ids, m = pad(seqs, idx, "cuda")
        model.model(input_ids=ids, attention_mask=m.long(), use_cache=False)
        for l, idxs in by_layer.items():
            a = cur[l].masked_fill(~m[..., None], float("-inf"))  # (B,T,n)
            for j, i in enumerate(idxs):
                v, flat = a[..., j].flatten().topk(k)
                for val, f in zip(v.tolist(), flat.tolist()):
                    r, p = divmod(f, a.shape[1])
                    best[(l, i)].append((val, idx[r], p))
    for hk in hooks:
        hk.remove()
    out = []
    for (l, i), lst in best.items():
        lst = sorted(lst, reverse=True)[:k]
        ctx = []
        for val, si, p in lst:
            s = seqs[si]
            left = tok.decode(s[max(0, p - win) : p])
            tokn = tok.decode(s[p : p + 1])
            right = tok.decode(s[p + 1 : p + 4])
            ctx.append({"act": round(val, 2), "left": left, "tok": tokn, "right": right})
        out.append({"layer": l, "neuron": i, "contexts": ctx})
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tok = load_tokenizer()
    model = load_model(TEACHER)
    code = np.load(DATA / "calib_code.npz", allow_pickle=True)
    web = np.load(DATA / "calib_web.npz", allow_pickle=True)
    code_seqs = [list(map(int, s)) for s in code["tokens"]]
    code_plen = [int(p) for p in code["plen"]]
    web_seqs = [list(map(int, s)) for s in web["tokens"]]
    res = {}
    for name, seqs in (("code", code_seqs), ("web", web_seqs)):
        t = time.time()
        r = collect(model, seqs)
        print(f"{name}: stats over {r['tokens']:,} tokens in {time.time() - t:.0f}s", flush=True)
        res[name] = r
    print("layer-drop KL on code (answer tokens):", flush=True)
    res["code"]["drop_kl"], res["code"]["drop_agree"] = layer_drop_kl(
        model, code_seqs[:KL_SEQS], [p - 1 for p in code_plen[:KL_SEQS]])
    print("layer-drop KL on web:", flush=True)
    res["web"]["drop_kl"], res["web"]["drop_agree"] = layer_drop_kl(model, web_seqs[:KL_SEQS], [0] * KL_SEQS)
    for name, r in res.items():
        np.savez(OUT / f"{name}.npz", **{k: np.asarray(v) for k, v in r.items()})

    # code-specific vs web-specific neurons: share of each neuron in its layer's total importance
    c = res["code"]["ffn_imp"] / res["code"]["ffn_imp"].sum(1, keepdims=True)
    w = res["web"]["ffn_imp"] / res["web"]["ffn_imp"].sum(1, keepdims=True)
    ratio = np.log((c + 1e-9) / (w + 1e-9))
    L, F = c.shape
    strong_c = c > np.percentile(c, 99, axis=1, keepdims=True)
    strong_w = w > np.percentile(w, 99, axis=1, keepdims=True)
    cand_c = sorted(((ratio[l, i], l, i) for l, i in zip(*np.nonzero(strong_c))), reverse=True)
    cand_w = sorted(((-ratio[l, i], l, i) for l, i in zip(*np.nonzero(strong_w))), reverse=True)

    def pick(cands, n=8):
        out, layers = [], set()
        for _, l, i in cands:
            if l in layers:
                continue
            out.append((int(l), int(i)))
            layers.add(l)
            if len(out) == n:
                break
        return out

    code_neurons, web_neurons = pick(cand_c), pick(cand_w)
    ctx = {
        "code_specific": top_contexts(model, tok, code_seqs, code_neurons),
        "web_specific": top_contexts(model, tok, web_seqs, web_neurons),
        "web_specific_on_code": top_contexts(model, tok, code_seqs, web_neurons, k=3),
    }
    for g in ctx.values():
        for e in g:
            e["log_ratio_code_vs_web"] = float(ratio[e["layer"], e["neuron"]])
    (OUT / "contexts.json").write_text(json.dumps(ctx, indent=1))
    print("done", flush=True)


if __name__ == "__main__":
    main()
