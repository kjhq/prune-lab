"""Code-generation eval: HumanEval+, MBPP+ and a held-out KodCode slice, pass@1 greedy, executed in
the sandbox."""
from __future__ import annotations

import glob
import json
import re
import time
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import torch

from .common import DATA, VocabMap, chat_prompt
from .sandbox import run_jobs

HF = Path.home() / ".cache/huggingface/hub"


def _pq(pattern: str) -> pd.DataFrame:
    return pq.read_table(sorted(glob.glob(str(HF / pattern)))[0]).to_pandas()


def load_eval_sets(kodcode_n: int | None = None) -> list[dict]:
    items = []
    he = _pq("datasets--evalplus--humanevalplus/snapshots/*/data/*.parquet")
    for r in he.itertuples():
        items.append({"set": "humaneval+", "id": r.task_id, "question": r.prompt,
                      "test": r.test + f"\n\ncheck({r.entry_point})\n", "kind": "script"})
    mb = _pq("datasets--evalplus--mbppplus/snapshots/*/data/*.parquet")
    for r in mb.itertuples():
        q = f'"""\n{r.prompt}\n{r.test_list[0]}\n"""\n'
        items.append({"set": "mbpp+", "id": f"Mbpp/{r.task_id}", "question": q, "test": r.test, "kind": "script"})
    kc = DATA / "kodcode_heldout.jsonl"
    if kc.exists():
        rows = [json.loads(l) for l in open(kc)]
        for r in rows[:kodcode_n]:
            items.append({"set": "kodcode", "id": r["question_id"], "question": r["question"], "test": r["test"],
                          "kind": "pytest"})
    return items


_FENCE = re.compile(r"```(?:python|py|Python)?[ \t]*\n(.*?)(?:```|$)", re.S)


def extract_code(text: str) -> str:
    blocks = _FENCE.findall(text)
    if blocks:
        # prefer the longest block that defines something
        defs = [b for b in blocks if "def " in b or "class " in b]
        return max(defs or blocks, key=len)
    return text


def make_job(item: dict, code: str, timeout: int = 30) -> dict:
    if item["kind"] == "script":
        files = {"main.py": code + "\n\n" + item["test"]}
        cmd = ["python", "main.py"]
    else:
        files = {"solution.py": code, "test_solution.py": item["test"]}
        cmd = ["python", "-m", "pytest", "-x", "-q", "-p", "no:cacheprovider", "test_solution.py"]
    return {"id": f'{item["set"]}::{item["id"]}', "files": files, "cmd": cmd, "timeout": timeout}


@torch.no_grad()
def generate(model, tok, prompts: list[str], vmap: VocabMap | None = None, max_new_tokens: int = 768,
             batch_size: int = 64, log=print) -> list[str]:
    """Greedy batched generation. With `vmap`, the model works in compact ids."""
    dev = next(model.parameters()).device
    stop_old = [tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")]
    if vmap is not None:
        enc = [vmap.encode_ids(tok(p, add_special_tokens=False).input_ids) for p in prompts]
        stop = [int(vmap.old2new[s]) for s in stop_old if vmap.old2new[s] >= 0]
        pad = stop[0]
    else:
        enc = [tok(p, add_special_tokens=False).input_ids for p in prompts]
        stop, pad = stop_old, stop_old[1]
    order = sorted(range(len(enc)), key=lambda i: len(enc[i]))
    outs: list[str] = [""] * len(enc)
    t0 = time.time()
    for b in range(0, len(order), batch_size):
        idx = order[b : b + batch_size]
        L = max(len(enc[i]) for i in idx)
        ids = torch.full((len(idx), L), pad, dtype=torch.long)
        att = torch.zeros((len(idx), L), dtype=torch.long)
        for r, i in enumerate(idx):
            ids[r, L - len(enc[i]) :] = torch.tensor(enc[i])
            att[r, L - len(enc[i]) :] = 1
        gen = model.generate(
            input_ids=ids.to(dev), attention_mask=att.to(dev), max_new_tokens=max_new_tokens, do_sample=False,
            eos_token_id=stop, pad_token_id=pad, temperature=None, top_p=None, top_k=None,
        )[:, L:].tolist()
        for r, i in enumerate(idx):
            g = gen[r]
            cut = next((k for k, t in enumerate(g) if t in stop), len(g))
            g = g[:cut]
            outs[i] = tok.decode(vmap.to_old(g) if vmap is not None else g, skip_special_tokens=True)
        log(f"gen {min(b + batch_size, len(order))}/{len(order)} {time.time() - t0:.0f}s")
    return outs


def evaluate(model, tok, out_dir: Path, vmap: VocabMap | None = None, enable_thinking: bool | None = None,
             kodcode_n: int | None = 300, batch_size: int = 64, max_new_tokens: int = 768, log=print) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    items = load_eval_sets(kodcode_n)
    prompts = [chat_prompt(tok, it["question"], enable_thinking) for it in items]
    gens = generate(model, tok, prompts, vmap, max_new_tokens, batch_size, log)
    jobs = [make_job(it, extract_code(g)) for it, g in zip(items, gens)]
    res = run_jobs(jobs)
    summary: dict[str, dict] = {}
    with open(out_dir / "samples.jsonl", "w") as f:
        for it, g, j in zip(items, gens, jobs):
            r = res[j["id"]]
            f.write(json.dumps({"set": it["set"], "id": it["id"], "passed": r["passed"], "timeout": r["timeout"],
                                "completion": g, "tail": r["tail"]}) + "\n")
            s = summary.setdefault(it["set"], {"n": 0, "passed": 0})
            s["n"] += 1
            s["passed"] += int(r["passed"])
    for s in summary.values():
        s["pass@1"] = round(100 * s["passed"] / s["n"], 2)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary
