"""Shared paths, prompt format, model loading and the compact-vocab map."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch

CACHE = Path(os.environ.get("PRUNELAB_CACHE", Path.home() / ".cache/prune-lab"))
DATA = CACHE / "data"
MODELS = CACHE / "models"
RUNS = CACHE / "runs"
RESULTS = CACHE / "results"

TEACHER = "Qwen/Qwen3-4B-Instruct-2507"
QWEN06 = "Qwen/Qwen3-0.6B"
QWEN17 = "Qwen/Qwen3-1.7B"
ORIG_VOCAB = 151936

# EvalPlus instruct-style prompt; training uses the same wrapper so the task format is fixed.
INSTRUCTION = (
    "Please provide a self-contained Python script that solves the following problem "
    "in a markdown code block:\n```\n{q}\n```"
)


def user_message(question: str) -> str:
    return INSTRUCTION.format(q=question.strip())


def answer_text(solution: str) -> str:
    return "```python\n" + solution.strip() + "\n```"


def chat_prompt(tok, question: str, enable_thinking: bool | None = None) -> str:
    kw = {} if enable_thinking is None else {"enable_thinking": enable_thinking}
    return tok.apply_chat_template(
        [{"role": "user", "content": user_message(question)}], tokenize=False, add_generation_prompt=True, **kw
    )


def load_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(TEACHER)


def load_model(path: str, dtype=torch.bfloat16, device="cuda"):
    from transformers import AutoModelForCausalLM

    m = AutoModelForCausalLM.from_pretrained(path, dtype=dtype, attn_implementation="sdpa")
    return m.to(device).eval()


class VocabMap:
    """Compact vocab: student id j <-> original id keep[j].

    Original ids missing from the kept set are rewritten as a greedy longest-match sequence of kept
    tokens over the token's byte-level string (all 256 byte tokens are always kept), so any text stays
    encodable.
    """

    def __init__(self, keep: np.ndarray, tok):
        self.keep = np.asarray(keep, dtype=np.int64)
        self.old2new = np.full(ORIG_VOCAB, -1, dtype=np.int64)
        self.old2new[self.keep] = np.arange(len(self.keep))
        self.tok = tok
        self._str2new = None
        self._decomp: dict[int, list[int]] = {}

    @property
    def size(self) -> int:
        return len(self.keep)

    def save(self, path: Path):
        np.save(path, self.keep)

    @classmethod
    def load(cls, path: Path, tok):
        return cls(np.load(path), tok)

    def _decompose(self, old_id: int) -> list[int]:
        if old_id in self._decomp:
            return self._decomp[old_id]
        if self._str2new is None:
            toks = self.tok.convert_ids_to_tokens([int(i) for i in self.keep])
            self._str2new = {t: j for j, t in enumerate(toks) if t is not None}
        s = self.tok.convert_ids_to_tokens(int(old_id))
        out, i = [], 0
        while i < len(s):
            for j in range(len(s), i, -1):
                if s[i:j] in self._str2new:
                    out.append(self._str2new[s[i:j]])
                    i = j
                    break
            else:
                raise KeyError(f"cannot decompose token {old_id} {s!r}")
        self._decomp[old_id] = out
        return out

    def encode_ids(self, old_ids) -> list[int]:
        out = []
        for o in old_ids:
            n = self.old2new[o]
            out.extend([int(n)] if n >= 0 else self._decompose(int(o)))
        return out

    def to_old(self, new_ids) -> list[int]:
        return [int(self.keep[i]) for i in new_ids]


def byte_token_ids(tok) -> list[int]:
    """Ids of the 256 single-byte tokens in a byte-level BPE vocab."""
    # GPT-2 byte->unicode table (same as tokenizers' ByteLevel pre-tokenizer)
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    chars = [chr(c) for c in cs]
    ids = tok.convert_tokens_to_ids(chars)
    assert all(i is not None and i < ORIG_VOCAB for i in ids)
    return ids


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    tmp.replace(path)


def vocab_keep(tok=None) -> np.ndarray:
    """Kept original token ids: every id seen in the training pool + 256 byte tokens + special tokens."""
    path = DATA / "vocab_keep.npy"
    if path.exists():
        return np.load(path)
    tok = tok or load_tokenizer()
    counts = np.load(DATA / "vocab_counts_code.npy")
    keep = set(np.nonzero(counts)[0].tolist()) | set(byte_token_ids(tok)) | set(range(151643, ORIG_VOCAB))
    keep = np.array(sorted(keep), dtype=np.int64)
    np.save(path, keep)
    return keep


def train_arrays():
    tokens = np.load(DATA / "train_tokens.npy", mmap_mode="r")
    offsets = np.load(DATA / "train_offsets.npy")
    plen = np.load(DATA / "train_plen.npy")
    return tokens, offsets, plen
