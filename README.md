# TinyGPT-Opt: Transformer Training & Inference Optimization in PyTorch

[![tests](https://github.com/KirtiK07/tinygpt-opt/actions/workflows/tests.yml/badge.svg)](https://github.com/KirtiK07/tinygpt-opt/actions/workflows/tests.yml)

A GPT-style transformer written from scratch in PyTorch, then made measurably faster and leaner
one optimization at a time. A test checks each optimization against the unoptimized reference,
so every speedup comes with proof that the outputs didn't change.

Target hardware: Google Colab free tier, NVIDIA T4 (16 GB).

> **Status:** code, configs and the test suite are done, and the tests pass on CPU.
> The GPU runs that produce the numbers below haven't been run yet. Every table cell marked
> *not measured yet* gets filled from `results/results.csv` after running on a T4.
> Only real measured numbers go here.

## What it covers

| Skill | Where it lives |
|---|---|
| Implement a model | Hand-written transformer in [`tinygpt/model.py`](tinygpt/model.py): embeddings, causal multi-head attention, MLP, LayerNorm, residuals, weight tying |
| Reproducible workflow | YAML configs, fixed seeds, metrics CSV, one-command rerun ([`tinygpt/train.py`](tinygpt/train.py)) |
| Optimize memory / throughput | 8-step optimization ladder with before/after numbers |
| Debug numerical failures | fp16 NaN case study ([below](#numerical-debugging-case-study-fp16-nan)) |
| Verify correctness | pytest suite comparing each optimization to the reference ([`tests/`](tests)) |
| Coding agents | [Workflow section](#how-a-coding-agent-was-used) |

## Model

| | |
|---|---|
| Layers / heads / width | 6 / 6 / 384 |
| Context | 256 tokens |
| Parameters | 30.1M (19.3M of them in the tied token embedding) |
| Tokenizer | GPT-2 BPE via `tiktoken` (vocab padded 50257 → 50304) |
| Data | TinyStories (main), Tiny Shakespeare (smoke test) |

Attention has two interchangeable paths, `manual` (matmul → causal mask → softmax → matmul) and
`fused` (`F.scaled_dot_product_attention`). Both support a KV-cache.

## Repo layout

```
configs/      base.yaml (step 0) → amp → fused → ckpt → compile (step 4); shakespeare; nan_repro / nan_fixed
tinygpt/
  model.py    transformer, manual + fused attention, gradient checkpointing, KV-cache
  data.py     download, tokenize, uint16 binary shards, random-batch loader
  train.py    config-driven training loop, AMP + GradScaler, checkpointing, metrics
  generate.py sampling, KV-cache, batched generation
  quantize.py int8 dynamic quantization for CPU inference
  profile.py  inference benchmarks, results report, charts, NaN case-study plots
tests/        correctness tests, one per optimization
results/      results.csv + charts (written by the runs)
notebooks/    colab_run.ipynb, the whole pipeline on a Colab T4
```

## Setup

**Colab (recommended):** open [`notebooks/colab_run.ipynb`](notebooks/colab_run.ipynb), choose
*Runtime → Change runtime type → T4 GPU*, and run the cells top to bottom.

**Local:**

```bash
git clone https://github.com/KirtiK07/tinygpt-opt.git
cd tinygpt-opt
pip install -r requirements.txt     # Python 3.11, PyTorch 2.x
pytest -q                           # correctness suite, about 20 s on CPU
```

## Usage

```bash
# 1. Data
python -m tinygpt.data --dataset shakespeare                 # ~300K tokens, seconds
python -m tinygpt.data --dataset tinystories                 # full set: 2.2 GB download
python -m tinygpt.data --dataset tinystories --max_train_docs 300000   # faster subset

# 2. Baseline: one command reproduces the run
python -m tinygpt.train --config configs/base.yaml

# 3. Training ladder: each config inherits the previous one
python -m tinygpt.train --config configs/amp.yaml
python -m tinygpt.train --config configs/fused.yaml
python -m tinygpt.train --config configs/ckpt.yaml
python -m tinygpt.train --config configs/compile.yaml

# 4. Inference ladder (KV-cache, batching, int8) from the trained model
python -m tinygpt.profile infer --ckpt out/compile/ckpt.pt

# 5. Results table and charts
python -m tinygpt.profile report
python -m tinygpt.profile plot

# Generate text
python -m tinygpt.generate --ckpt out/compile/ckpt.pt --prompt "Once upon a time" --num_samples 3
python -m tinygpt.generate --ckpt out/compile/ckpt.pt --prompt "A dog" --quantize   # int8 on CPU
```

Any config value can be overridden from the command line: `--set max_steps=500 --set model.n_layer=4`.
Each run writes `out/<run_name>/` (resolved config, `metrics.csv`, `ckpt.pt`, `sample.txt`) and
appends a summary row to `results/results.csv`.

## Optimization ladder

Each step is added on top of the previous one, with the same model, data, seed, batch size and
step count (`tests/test_repro.py::test_ladder_configs_stack` enforces this). Training tokens/sec
leaves out the warm-up steps, which also covers `torch.compile`'s compile time.

| # | Technique | Stage | How | tokens/sec | Peak GPU memory | Val loss |
|---|---|---|---|---|---|---|
| 0 | Baseline: fp32, manual attention | Training | plain loop | *not measured yet* | *not measured yet* | *not measured yet* |
| 1 | Mixed precision (fp16 + loss scaling) | Training | `torch.autocast` + `GradScaler` | *not measured yet* | *not measured yet* | *not measured yet* |
| 2 | Fused attention | Training | `F.scaled_dot_product_attention` | *not measured yet* | *not measured yet* | *not measured yet* |
| 3 | Gradient checkpointing | Training | `torch.utils.checkpoint` per block | *not measured yet* | *not measured yet* | *not measured yet* |
| 4 | `torch.compile` | Training | `torch.compile(model)` | *not measured yet* | *not measured yet* | *not measured yet* |

| # | Technique | Stage | How | tokens/sec | ms / token | Model size |
|---|---|---|---|---|---|---|
| 5-ref | No cache (reference) | Inference (GPU) | full recompute each token | *not measured yet* | *not measured yet* | |
| 5 | KV-cache | Inference (GPU) | cache keys/values per layer | *not measured yet* | *not measured yet* | |
| 6 | Batched generation | Inference (GPU) | 16 prompts at once | *not measured yet* | *not measured yet* | |
| 7-ref | fp32 (reference) | Inference (CPU) | | *not measured yet* | *not measured yet* | *not measured yet* |
| 7 | int8 dynamic quantization | Inference (CPU) | `quantize_dynamic` | *not measured yet* | *not measured yet* | *not measured yet* |

`python -m tinygpt.profile report` prints this table filled in, with each step as a change from
its baseline (for example "1.8× tokens/sec, −42% peak memory").

Charts (written by `python -m tinygpt.profile plot`):

- `results/throughput.png`: training tokens/sec per step. *Not generated yet.*
- `results/peak_memory.png`: peak GPU memory per step. *Not generated yet.*

Notes on the int8 step:

- Because of weight tying, quantizing only `nn.Linear` would give `lm_head` its own int8 copy while
  the fp32 embedding stays, and the "quantized" file ends up *larger*. `quantize.py` quantizes the
  embeddings too. The default model's state dict shrinks from about 115 MB to about 48 MB (measured).
- `torch.ao.quantization` is deprecated in recent PyTorch in favour of `torchao`. It still works
  (tested on 2.15).

## Correctness tests

Every optimization has to pass before its numbers count.

| Test | What it checks | Pass condition |
|---|---|---|
| `test_attention.py` | Fused attention vs manual: forward, gradients, and with a KV-cache offset | `torch.allclose`, atol 1e-5 (fp32) |
| `test_kvcache.py` | Greedy generation with vs without KV-cache, including past the context window; incremental vs full logits | Identical token IDs |
| `test_kvcache.py` | Batched vs one-prompt-at-a-time generation | Identical token IDs |
| `test_amp.py` | Mixed-precision loss vs fp32 loss on the same batch | Within 1e-2 (fp16 on GPU; relative 1e-2 for bf16 on CPU) |
| `test_amp.py` | fp16 + GradScaler training stays finite; non-finite loss guard fires | (GPU-only test skips on CPU) |
| `test_repro.py` | Two runs, same seed and config | Identical loss for the first 50 steps |
| `test_repro.py` | Gradient checkpointing vs none | Losses within 1e-5 |
| `test_repro.py` | Ladder configs differ only in the technique being added | Same model, data, steps, seed |
| `test_shapes.py` | Output shapes, parameter count, causal mask | No token attends to future positions |
| `test_quant.py` | int8 vs fp32 next-token prediction; int8 file size | Top-1 agreement > 95%; under half the size |

Current status on CPU (PyTorch 2.15): **27 passed, 1 skipped** (the skip is the GPU-only fp16
training test). `torch.compile` isn't covered by a unit test because compiling on CPU needs a
C++ toolchain. On the T4, its correctness shows as matching validation loss against step 3.

## Numerical debugging case study: fp16 NaN

*Plots and observed numbers get added after running on the T4.*

1. **Break it.** `configs/nan_repro.yaml`: fp16 autocast with `GradScaler` off, 10× learning rate,
   no warm-up, no clipping.
   ```bash
   python -m tinygpt.train --config configs/nan_repro.yaml   # expected: stops with a non-finite loss
   ```
2. **Detect it.** The loop checks every loss and stops on the first `inf`/`NaN` (writes
   `out/nan_repro/nonfinite.txt`). It logs per-layer gradient norms every step
   (`grad_norms.csv`). To find the first bad operation, rerun with `--set detect_anomaly=true`,
   which wraps training in `torch.autograd.set_detect_anomaly`.
3. **Explain it.** fp16 tops out at 65,504, and its smallest normal value is about 6e-5. Without loss
   scaling, small gradients underflow to zero, and with a high learning rate, activations and
   attention scores grow until something overflows to `inf`. `inf - inf` in softmax then gives `NaN`.
4. **Fix it.** `configs/nan_fixed.yaml`: loss scaling back on, gradient clipping at max norm 1.0.
   softmax and LayerNorm already run in fp32 under autocast (the model uses `F.softmax` and
   `F.layer_norm` so that autocast can do this).
   ```bash
   python -m tinygpt.train --config configs/nan_fixed.yaml
   python -m tinygpt.profile nan_plot     # results/nan_loss.png, results/nan_grad_norms.png
   ```
5. **Write it up.** *TODO after the GPU run:* the step where it broke, which layer's gradient norm blew
   up first, what `detect_anomaly` reported, and the before/after loss curves.

## How a coding agent was used

The first version of this repo was generated with Claude Code from the project plan: the package
layout, model, training loop, benchmarks, configs, tests and this README. Running the test suite
during generation caught two real bugs before the first commit:

- `F.cross_entropy` used `.view()` on non-contiguous targets. It now uses `.reshape()`.
- The first int8 version quantized only `nn.Linear`. With tied embeddings, the result was *larger*
  than fp32. Quantizing the embeddings as well fixed it, and a size test now guards against it.

*TODO (author):* note what you reviewed, rewrote by hand and changed after running on the T4.
The plan says the model code should be written and understood by you, so go through
`tinygpt/model.py` line by line.

## Definition of done

- [ ] Model trains from scratch and produces readable sample stories
- [x] One command reproduces the baseline run from a config file
- [ ] All 8 ladder steps measured and logged in `results.csv`
- [x] All tests in `tests/` pass (CPU; rerun on the T4)
- [ ] fp16 NaN case study written up with plots
- [ ] README has the results table, two charts, setup steps and a coding-agent workflow section (charts and numbers pending)
- [x] Repo is public
- [ ] Pinned on GitHub and linked from the resume
