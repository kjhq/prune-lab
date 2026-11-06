"""KodCode -> filtered, tokenized training pool + verified held-out eval slice + calibration sets.

Outputs in ~/.cache/prune-lab/data/:
  train_tokens.npy (uint32, concatenated), train_offsets.npy (int64, n+1), train_plen.npy (int32 prompt lens),
  train_meta.parquet, kodcode_heldout.jsonl, vocab_counts_code.npy, vocab_counts_web.npy,
  calib_code.npz / calib_web.npz (token lists for activation analysis)
"""
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from prunelab.codeeval import make_job
from prunelab.common import DATA, ORIG_VOCAB, answer_text, chat_prompt, load_tokenizer
from prunelab.sandbox import run_jobs

MAX_LEN = 1024
SEED = 0
FINEWEB = Path.home() / ".cache/gemma-kv-retrofit/data/raw/fineweb-edu/sample/10BT/013_00000.parquet"


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    tok = load_tokenizer()
    if (DATA / "vocab_counts_code.npy").exists():
        return calib(tok)
    snap = glob.glob(str(Path.home() / ".cache/huggingface/hub/datasets--KodCode--KodCode-V1/snapshots/*/data"))[0]
    cols = ["style", "subset", "question_id", "question", "solution", "test", "benchmark_similarity", "gpt_difficulty"]
    df = pd.concat([pq.read_table(f, columns=cols).to_pandas() for f in sorted(glob.glob(snap + "/train-*.parquet"))])
    n0 = len(df)
    df = df[(df["style"] != "online_judge") & (df.benchmark_similarity.fillna(0) <= 0.8)]
    df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    print(f"kodcode {n0} -> {len(df)} after dropping online_judge + benchmark_similarity>0.8")

    # Held-out eval slice: stdlib-only tests, reference must pass in the sandbox.
    cand = df[~df.subset.isin(["Package", "Docs"])].head(700)
    jobs = [make_job({"set": "kodcode", "id": r.question_id, "test": r.test, "kind": "pytest"}, r.solution)
            for r in cand.itertuples()]
    res = run_jobs(jobs)
    ok = [r for r in cand.itertuples() if res[f"kodcode::{r.question_id}"]["passed"]][:400]
    print(f"heldout: {len(ok)} verified of {len(cand)} candidates")
    with open(DATA / "kodcode_heldout.jsonl", "w") as f:
        for r in ok:
            f.write(json.dumps({"question_id": r.question_id, "subset": r.subset, "style": r.style,
                                "difficulty": r.gpt_difficulty, "question": r.question, "solution": r.solution,
                                "test": r.test}) + "\n")
    held = {r.question_id for r in ok} | set(cand.question_id[:700])
    train = df[~df.question_id.isin(held)].reset_index(drop=True)

    prompts = [chat_prompt(tok, q) for q in train.question]
    answers = [answer_text(s) + "<|im_end|>" for s in train.solution]
    P = tok(prompts, add_special_tokens=False).input_ids
    A = tok(answers, add_special_tokens=False).input_ids
    keep = [i for i in range(len(P)) if len(P[i]) + len(A[i]) <= MAX_LEN]
    print(f"train: {len(train)} -> {len(keep)} with len <= {MAX_LEN}")
    seqs = [P[i] + A[i] for i in keep]
    lens = np.array([len(s) for s in seqs], dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(lens)])
    tokens = np.fromiter((t for s in seqs for t in s), dtype=np.uint32, count=int(offsets[-1]))
    plen = np.array([len(P[i]) for i in keep], dtype=np.int32)
    np.save(DATA / "train_tokens.npy", tokens)
    np.save(DATA / "train_offsets.npy", offsets)
    np.save(DATA / "train_plen.npy", plen)
    train.iloc[keep][["question_id", "subset", "style", "gpt_difficulty"]].reset_index(drop=True).to_parquet(
        DATA / "train_meta.parquet")
    ans_tokens = int((lens - plen).sum())
    print(f"tokens: {offsets[-1]:,} total, {ans_tokens:,} answer; mean len {lens.mean():.0f}")
    np.save(DATA / "vocab_counts_code.npy", np.bincount(tokens, minlength=ORIG_VOCAB))
    del P, A, seqs, tokens
    calib(tok)


def calib(tok):
    """Calibration: 512 code sequences (prompt+answer) and web chunks with the same token count."""
    tokens = np.load(DATA / "train_tokens.npy", mmap_mode="r")
    offsets = np.load(DATA / "train_offsets.npy")
    plen = np.load(DATA / "train_plen.npy")
    rng = np.random.default_rng(SEED)
    ci = rng.choice(len(plen), 512, replace=False)
    code = [tokens[offsets[i] : offsets[i + 1]].astype(np.int64).tolist() for i in ci]
    code_plen = [int(plen[i]) for i in ci]
    budget = sum(len(s) for s in code)
    web_docs = pq.read_table(FINEWEB, columns=["text"]).to_pandas().text.tolist()
    web, used, wc = [], 0, np.zeros(ORIG_VOCAB, dtype=np.int64)
    for d in web_docs[:20000]:
        ids = tok(d, add_special_tokens=False).input_ids
        wc += np.bincount(ids, minlength=ORIG_VOCAB)
        if used < budget:
            chunk = ids[:512]
            if len(chunk) >= 128:
                web.append(chunk)
                used += len(chunk)
    np.save(DATA / "vocab_counts_web.npy", wc)
    np.savez(DATA / "calib_code.npz", tokens=np.array(code, dtype=object), plen=np.array(code_plen))
    np.savez(DATA / "calib_web.npz", tokens=np.array(web, dtype=object))
    print(f"calib: code {len(code)} seqs {budget:,} tok; web {len(web)} seqs {used:,} tok")


if __name__ == "__main__":
    main()
