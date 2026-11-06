"""Structured pruning of a Qwen3 dense model into a smaller, standard Qwen3 config.

Everything is real slicing (no masks): layers, hidden channels, MLP neurons, query heads,
key/value heads and vocabulary rows are removed, and the result loads as a normal
Qwen3ForCausalLM.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class PrunePlan:
    layers: list[int]              # kept layer indices (in order)
    hidden: torch.Tensor           # kept hidden channels (sorted)
    ffn: dict[int, torch.Tensor]   # layer -> kept MLP neurons (sorted), same count for every layer
    q_heads: dict[int, torch.Tensor]  # layer -> kept query heads (original indices), grouped by kept kv head
    kv_heads: dict[int, torch.Tensor]  # layer -> kept kv heads (original indices, sorted)
    vocab: torch.Tensor | None     # kept original token ids (sorted) or None = keep all


def plan_from_scores(
    cfg,
    layer_score: torch.Tensor,     # (L,) higher = more important
    hidden_score: torch.Tensor,    # (d,)
    ffn_score: torch.Tensor,       # (L, ffn)
    head_score: torch.Tensor,      # (L, n_q)
    n_layers: int,
    hidden: int,
    ffn: int,
    n_q: int,
    n_kv: int,
    vocab: torch.Tensor | None = None,
) -> PrunePlan:
    L = cfg.num_hidden_layers
    layers = sorted(torch.topk(layer_score, n_layers).indices.tolist())
    hid = torch.topk(hidden_score, hidden).indices.sort().values
    ffn_keep = {l: torch.topk(ffn_score[l], ffn).indices.sort().values for l in range(L)}
    g_old = cfg.num_attention_heads // cfg.num_key_value_heads
    assert n_q % n_kv == 0
    g_new = n_q // n_kv
    q_keep, kv_keep = {}, {}
    for l in range(L):
        hs = head_score[l].view(cfg.num_key_value_heads, g_old)
        kv = torch.topk(hs.sum(1), n_kv).indices.sort().values
        q = []
        for g in kv.tolist():
            best = torch.topk(hs[g], g_new).indices.sort().values
            q.extend((g * g_old + best).tolist())
        q_keep[l], kv_keep[l] = torch.tensor(q), kv
    return PrunePlan(layers, hid, ffn_keep, q_keep, kv_keep, vocab)


def _heads_rows(idx: torch.Tensor, hd: int) -> torch.Tensor:
    return (idx[:, None] * hd + torch.arange(hd)[None]).reshape(-1)


@torch.no_grad()
def apply_plan(model, plan: PrunePlan):
    """Return a new Qwen3ForCausalLM (on CPU, model's dtype) built by slicing `model`."""
    from transformers import Qwen3Config, Qwen3ForCausalLM

    cfg = model.config
    hd = cfg.head_dim
    ffn_n = len(next(iter(plan.ffn.values())))
    n_q = len(next(iter(plan.q_heads.values())))
    n_kv = len(next(iter(plan.kv_heads.values())))
    vocab = plan.vocab if plan.vocab is not None else torch.arange(cfg.vocab_size)
    d = plan.hidden
    new_cfg_dict = cfg.to_dict()
    new_cfg_dict.update(
        hidden_size=len(d),
        intermediate_size=ffn_n,
        num_attention_heads=n_q,
        num_key_value_heads=n_kv,
        num_hidden_layers=len(plan.layers),
        vocab_size=len(vocab),
        head_dim=hd,
        tie_word_embeddings=True,
        max_window_layers=len(plan.layers),
        layer_types=["full_attention"] * len(plan.layers),
    )
    for k in ("architectures", "_name_or_path", "transformers_version", "dtype", "torch_dtype"):
        new_cfg_dict.pop(k, None)
    new_cfg = Qwen3Config(**new_cfg_dict)
    dtype = model.model.embed_tokens.weight.dtype
    with torch.device("meta"):
        new = Qwen3ForCausalLM(new_cfg)
    sd = {}
    src = {k: v.detach().to("cpu") for k, v in model.state_dict().items()}
    d_cpu, v_cpu = d.cpu(), vocab.cpu()
    sd["model.embed_tokens.weight"] = src["model.embed_tokens.weight"][v_cpu][:, d_cpu]
    sd["model.norm.weight"] = src["model.norm.weight"][d_cpu]
    for new_l, l in enumerate(plan.layers):
        p, q = f"model.layers.{l}.", f"model.layers.{new_l}."
        qr = _heads_rows(plan.q_heads[l].cpu(), hd)
        kvr = _heads_rows(plan.kv_heads[l].cpu(), hd)
        f = plan.ffn[l].cpu()
        sd[q + "input_layernorm.weight"] = src[p + "input_layernorm.weight"][d_cpu]
        sd[q + "post_attention_layernorm.weight"] = src[p + "post_attention_layernorm.weight"][d_cpu]
        sd[q + "self_attn.q_proj.weight"] = src[p + "self_attn.q_proj.weight"][qr][:, d_cpu]
        sd[q + "self_attn.k_proj.weight"] = src[p + "self_attn.k_proj.weight"][kvr][:, d_cpu]
        sd[q + "self_attn.v_proj.weight"] = src[p + "self_attn.v_proj.weight"][kvr][:, d_cpu]
        sd[q + "self_attn.o_proj.weight"] = src[p + "self_attn.o_proj.weight"][d_cpu][:, qr]
        sd[q + "self_attn.q_norm.weight"] = src[p + "self_attn.q_norm.weight"]
        sd[q + "self_attn.k_norm.weight"] = src[p + "self_attn.k_norm.weight"]
        sd[q + "mlp.gate_proj.weight"] = src[p + "mlp.gate_proj.weight"][f][:, d_cpu]
        sd[q + "mlp.up_proj.weight"] = src[p + "mlp.up_proj.weight"][f][:, d_cpu]
        sd[q + "mlp.down_proj.weight"] = src[p + "mlp.down_proj.weight"][d_cpu][:, f]
    sd = {k: v.contiguous().to(dtype) for k, v in sd.items()}
    new.load_state_dict(sd, strict=False, assign=True)
    new.tie_weights()
    missing = [n for n, t in new.named_parameters() if t.is_meta]
    assert not missing, missing
    # Rotary buffers were built on meta; rebuild them on CPU.
    new.model.rotary_emb = type(new.model.rotary_emb)(config=new_cfg)
    return new.eval()


def param_count(model, embeddings: bool = True) -> int:
    n = sum(p.numel() for p in model.parameters())
    if not embeddings:
        n -= model.model.embed_tokens.weight.numel()
    return n
