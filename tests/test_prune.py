import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from prunelab.prune import PrunePlan, apply_plan, plan_from_scores


def tiny(seed=0):
    torch.manual_seed(seed)
    cfg = Qwen3Config(
        vocab_size=97, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
        num_attention_heads=8, num_key_value_heads=2, head_dim=16, tie_word_embeddings=True,
        max_position_embeddings=128,
    )
    m = Qwen3ForCausalLM(cfg).eval()
    # randomise norms so slicing mistakes in them show up
    for n, p in m.named_parameters():
        if "norm" in n:
            p.data.uniform_(0.5, 1.5)
    return m


def full_plan(cfg, vocab=None):
    L = cfg.num_hidden_layers
    return PrunePlan(
        layers=list(range(L)), hidden=torch.arange(cfg.hidden_size),
        ffn={l: torch.arange(cfg.intermediate_size) for l in range(L)},
        q_heads={l: torch.arange(cfg.num_attention_heads) for l in range(L)},
        kv_heads={l: torch.arange(cfg.num_key_value_heads) for l in range(L)}, vocab=vocab,
    )


def test_identity_plan_reproduces_logits():
    m = tiny()
    x = torch.randint(0, 97, (2, 11))
    new = apply_plan(m, full_plan(m.config))
    torch.testing.assert_close(new(x).logits, m(x).logits, rtol=1e-5, atol=1e-5)


def test_vocab_slice_matches_column_subset():
    m = tiny()
    keep = torch.tensor(sorted(set(range(0, 97, 3)) | {5, 7}))
    new = apply_plan(m, full_plan(m.config, vocab=keep))
    old2new = {int(o): i for i, o in enumerate(keep)}
    x_old = keep[torch.randint(0, len(keep), (2, 9))]
    x_new = torch.tensor([[old2new[int(t)] for t in row] for row in x_old])
    torch.testing.assert_close(new(x_new).logits, m(x_old).logits[..., keep], rtol=1e-5, atol=1e-5)


def test_zeroed_units_prune_exactly():
    """Zero out the units a plan removes in the big model; the pruned model must match it exactly
    (except hidden channels, which also change the RMSNorm statistic, so they are kept here)."""
    m = tiny()
    cfg = m.config
    L, hd = cfg.num_hidden_layers, cfg.head_dim
    g = torch.Generator().manual_seed(1)
    plan = plan_from_scores(
        cfg,
        layer_score=torch.tensor([3.0, 1.0, 2.0, 4.0]),
        hidden_score=torch.ones(cfg.hidden_size),
        ffn_score=torch.rand(L, cfg.intermediate_size, generator=g),
        head_score=torch.rand(L, cfg.num_attention_heads, generator=g),
        n_layers=3, hidden=cfg.hidden_size, ffn=48, n_q=4, n_kv=2,
    )
    assert plan.layers == [0, 2, 3]
    ref = tiny()
    with torch.no_grad():
        for l, layer in enumerate(ref.model.layers):
            drop_f = torch.ones(cfg.intermediate_size, dtype=torch.bool)
            drop_f[plan.ffn[l]] = False
            layer.mlp.down_proj.weight[:, drop_f] = 0
            drop_q = torch.ones(cfg.num_attention_heads, dtype=torch.bool)
            drop_q[plan.q_heads[l]] = False
            cols = drop_q.repeat_interleave(hd)
            layer.self_attn.o_proj.weight[:, cols] = 0
        del ref.model.layers[1]
        ref.config.num_hidden_layers = 3
        for i, layer in enumerate(ref.model.layers):
            layer.self_attn.layer_idx = i
    new = apply_plan(m, plan)
    assert new.config.num_attention_heads == 4 and new.config.intermediate_size == 48
    x = torch.randint(0, 97, (2, 13))
    torch.testing.assert_close(new(x).logits, ref(x).logits, rtol=1e-5, atol=1e-5)
