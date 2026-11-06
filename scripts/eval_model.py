"""Code eval for one model: HumanEval+, MBPP+, KodCode held-out (pass@1 greedy, sandboxed).

  --model  HF hub id or local dir; a local dir with vocab_keep.npy is run in compact vocab
  --thinking false   pass enable_thinking=False to the chat template (Qwen3 hybrid models)
  --own-template     use the model's own chat template instead of the teacher's
"""
import argparse
import json
from pathlib import Path

import torch

from prunelab.codeeval import evaluate
from prunelab.common import RESULTS, VocabMap, load_model, load_tokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--thinking", choices=["true", "false"], default=None)
    ap.add_argument("--own-template", action="store_true")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--kodcode-n", type=int, default=300)
    a = ap.parse_args()
    out = RESULTS / "eval" / a.name
    if (out / "summary.json").exists():
        print("exists:", json.loads((out / "summary.json").read_text()))
        return
    if a.own_template:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(a.model)
    else:
        tok = load_tokenizer()
    vk = Path(a.model) / "vocab_keep.npy"
    vmap = VocabMap.load(vk, tok) if vk.exists() else None
    model = load_model(a.model, dtype=torch.bfloat16)
    think = None if a.thinking is None else a.thinking == "true"
    s = evaluate(model, tok, out, vmap=vmap, enable_thinking=think, kodcode_n=a.kodcode_n, batch_size=a.batch)
    print(a.name, json.dumps(s), flush=True)


if __name__ == "__main__":
    main()
