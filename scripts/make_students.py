"""Build the student starting points, all with the same compact vocab.

  pruned-code-<shape>  teacher pruned with importance measured on code
  pruned-web-<shape>   teacher pruned with importance measured on web text (does task calibration matter?)
  qwen06-vt            Qwen3-0.6B, vocab-trimmed only (same shape as q06)
  random-<shape>       same config, random init
"""
import argparse
import json

import numpy as np
import torch

from prunelab.common import MODELS, QWEN06, QWEN17, RESULTS, TEACHER, load_model, vocab_keep
from prunelab.prune import PrunePlan, apply_plan, param_count, plan_from_scores

SHAPES = {
    "q06": dict(n_layers=28, hidden=1024, ffn=3072, n_q=16, n_kv=8),   # = Qwen3-0.6B
    "q17": dict(n_layers=28, hidden=2048, ffn=6144, n_q=16, n_kv=8),   # = Qwen3-1.7B
    "tiny": dict(n_layers=20, hidden=768, ffn=2304, n_q=8, n_kv=4),
}


def scores(name):
    z = np.load(RESULTS / "analysis" / f"{name}.npz")
    hidden = torch.from_numpy(z["attn_in"].sum(0) + z["mlp_in"].sum(0))
    # never drop massive-activation channels in the residual stream
    rm = z["resid_max"].max(0)
    outl = np.nonzero(rm > 20 * np.median(rm))[0]
    hidden[outl] = float("inf")
    return dict(
        layer_score=torch.from_numpy(z["drop_kl"]).float(),
        hidden_score=hidden.float(),
        ffn_score=torch.from_numpy(z["ffn_imp"]).float(),
        head_score=torch.from_numpy(z["head_imp"]).float(),
    ), outl


def save(model, name, info):
    out = MODELS / name
    model.save_pretrained(out)
    np.save(out / "vocab_keep.npy", vocab_keep())
    info.update(params=param_count(model), params_non_embedding=param_count(model, embeddings=False))
    (out / "prunelab.json").write_text(json.dumps(info, indent=1))
    print(f"{name}: {info['params'] / 1e6:.0f}M params ({info['params_non_embedding'] / 1e6:.0f}M non-embedding)",
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", default="q06")
    ap.add_argument("--calib", default="code,web")
    a = ap.parse_args()
    keep = torch.from_numpy(vocab_keep())
    teacher = load_model(TEACHER)
    print(f"teacher: {param_count(teacher) / 1e6:.0f}M params; compact vocab {len(keep)}")
    for shape in a.shapes.split(","):
        for calib in a.calib.split(","):
            sc, outl = scores(calib)
            plan = plan_from_scores(teacher.config, **sc, **SHAPES[shape], vocab=keep)
            m = apply_plan(teacher, plan)
            save(m, f"pruned-{calib}-{shape}", {"kind": "pruned", "calib": calib, "shape": shape,
                                                "layers": plan.layers, "outlier_channels": outl.tolist()})
    for shape, base in (("q06", QWEN06), ("q17", QWEN17)):  # vendor model of the same shape, vocab-trimmed only
        if shape not in a.shapes.split(","):
            continue
        q = load_model(base)
        L = q.config.num_hidden_layers
        full = PrunePlan(list(range(L)), torch.arange(q.config.hidden_size),
                         {l: torch.arange(q.config.intermediate_size) for l in range(L)},
                         {l: torch.arange(q.config.num_attention_heads) for l in range(L)},
                         {l: torch.arange(q.config.num_key_value_heads) for l in range(L)}, keep)
        name = "qwen06" if shape == "q06" else "qwen17"
        save(apply_plan(q, full), f"{name}-vt", {"kind": name, "shape": shape})
        del q
    from transformers import AutoConfig, Qwen3ForCausalLM

    for shape in a.shapes.split(","):
        cfg = AutoConfig.from_pretrained(MODELS / f"pruned-{a.calib.split(',')[0]}-{shape}")
        torch.manual_seed(0)
        m = Qwen3ForCausalLM(cfg).to(torch.bfloat16)
        save(m, f"random-{shape}", {"kind": "random", "shape": shape})


if __name__ == "__main__":
    main()
