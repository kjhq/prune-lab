# prune-lab

Is a big language model that has been cut down a better starting point than a small one that was trained small?

I took Qwen3-4B, cut it to the exact shape of Qwen3-0.6B (an 8x cut), and then trained three models of that
shape on the same data with the same budget: the cut-down 4B, the real Qwen3-0.6B, and a randomly initialised
one. The task is narrow and hard to fake: write a Python function from a description, scored by running hidden
unit tests.

**Short answer: no.** At this training budget the ready-made small model wins by a wide margin.

## Result

Every trained model saw the same 15M answer tokens (448 steps).

| Model | HumanEval+ | MBPP+ | KodCode (held out) | KL to teacher | Top-1 agreement |
|---|---|---|---|---|---|
| Qwen3-4B, the teacher | 84.1 | 68.0 | 49.0 | 0 | 100% |
| Qwen3-0.6B, untrained | 35.4 | 37.8 | 19.3 | 0.60 | n/a |
| Qwen3-0.6B, trained | **42.7** | **39.7** | **34.7** | 0.153 | 93.6% |
| 4B cut to the 0.6B shape, trained | 7.3 | 9.5 | 16.0 | 0.330 | 89.6% |
| Random init, trained | 0.6 | 0.5 | 2.7 | 1.112 | 75.2% |

- The first three columns are pass@1 in percent: the share of problems where the model's single greedy answer
  passes every hidden test. Higher is better.
- KL to teacher is how far the model's next-token predictions are from the teacher's on 512 held-out solutions.
  Lower is better.

What I take from it:

- **The cut keeps something real.** The cut-down model beats random init by a long way on the same budget
  (7.3 against 0.6 on HumanEval+).
- **Pretraining is worth far more than what the cut keeps.** 15M tokens cannot rebuild what an 8x cut removes.
- **Token agreement hides the gap.** 90% and 94% top-1 agreement look close, but whole-program correctness is
  3 to 6 times apart, because one wrong token in about 150 breaks a program.

### Does it matter which data you measure importance on?

A little. Choosing what to keep from web text instead of code gave HumanEval+ 8.5, MBPP+ 6.6, KodCode 13.3
(79 problems solved, against 96 for the code-calibrated cut) and a KL of 0.350 against 0.330. Most of what
survives a cut is machinery every task shares.

### A smaller shape

Cutting further, to 20 layers and hidden size 768, and training on the same budget gave HumanEval+ 6.1,
MBPP+ 2.1, KodCode 7.7 (KL 0.527).

### Not done

- The random-init baseline for the smaller shape was stopped early, so that row has nothing to compare against.
- The gentler 2.4x cut to Qwen3-1.7B's shape (`scripts/sweep2.sh`) has not been run.
- One training run per model. No seeds were repeated, so small differences between rows are not reliable.

## Setup

- **Teacher:** `Qwen/Qwen3-4B-Instruct-2507`, a plain dense transformer (36 layers, hidden size 2560, tied 152k
  vocabulary). It slices cleanly and has smaller siblings with the same tokenizer to compare against.
- **Scoring:** HumanEval+ (164 problems), MBPP+ (378) and 300 held-out KodCode problems. Greedy decoding, pass@1,
  run in a Docker sandbox with no network, a read-only root, a non-root user and memory and process caps.
- **Training data:** KodCode-V1 (CC BY-NC 4.0) without the `online_judge` style, without rows whose
  `benchmark_similarity` is above 0.8, and without the held-out slice. That leaves 396k problems and 187M tokens,
  72M of them solution tokens.
- **Vocabulary:** Python task text uses 64.6k of the 152k token ids, so every student uses a compact
  65,024-token vocabulary. Ids never seen in training are rewritten as byte-level pieces at evaluation time, so
  any prompt stays encodable.

## Method

1. `scripts/analyze.py` measures how much each channel, MLP neuron and attention head matters, and how much the
   output changes when one layer is dropped, on code and on web text.
2. `scripts/make_students.py` does a one-shot structured cut to an exact target shape, in the style of Minitron.
   This is real slicing of the weight matrices, not masking. For the 0.6B shape it drops the 8 layers that matter
   least for code, then keeps the strongest hidden channels, the strongest MLP neurons in each layer and the top
   2 query heads in each key-value group.
3. `scripts/label.py` stores the teacher's top-32 log-probabilities for every solution token.
4. `scripts/train.py` trains every weight of the student. The loss is the KL divergence to the teacher over
   those top-32 tokens plus a bucket for the rest, plus 0.5 x cross-entropy on the reference solution. The
   learning rate for each model is picked by short probes (`scripts/pick_lr.py`).
5. `scripts/eval_model.py` runs the sandboxed pass@1.

`scripts/pilot.sh` chains steps 2 to 5. Every step can be rerun and picks up where it stopped.

## Running it

You need an NVIDIA GPU with 24 GB of memory, Docker and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
docker build -t prunelab-sandbox docker
uv run pytest -q tests                    # the sliced model reproduces the original maths exactly
uv run python scripts/check_sandbox.py    # reference solutions pass (HumanEval+ 163/164; HumanEval/32's test is broken upstream)
uv run python scripts/prep_data.py
uv run python scripts/analyze.py
scripts/pilot.sh
```

Data, labels, models and results go to `~/.cache/prune-lab/`; set `PRUNELAB_CACHE` to move them.
`prep_data.py` reads one FineWeb-Edu shard (`sample/10BT/013_00000.parquet`) for the web-text comparison; the
path is at the top of the script.

## Notes from running it

- A check on the distillation labels: the teacher scored against its own labels gives KL 0.0000 and 99.8% top-1
  agreement.
- The teacher's own loss on the KodCode reference solutions is about 1.0 nats per token. The references are
  written in a different style from the teacher's.
- Learning-rate probes (1.5M answer tokens each, KL to teacher / top-1 agreement): cut-down model 3e-4 gave
  0.98 / 75% and 1e-3 was unstable; Qwen3-0.6B 6e-5 gave 0.22 / 92%; random init 1e-3 gave 5.35 / 16%.
- `prep_data.py` peaks at about 16 GB of RAM while tokenizing.
- Training the 507M-parameter student takes about 3 MB of activations per padded token on top of about 8 GB of
  weights and optimizer state. The default micro-batch of 4,096 tokens peaks near 22 GB; steps stay at 32k
  answer tokens through gradient accumulation.
- In 8% of entries the teacher's top-32 contains ids outside the compact vocabulary. They are masked and their
  probability goes to the rest bucket.

## Licence

MIT for the code. KodCode-V1 is CC BY-NC 4.0, and the Qwen models carry their own licences.
