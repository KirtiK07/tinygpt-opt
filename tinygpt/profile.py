"""Measure, report and chart the optimization ladder.

    # Inference ladder (steps 5-7) from a trained checkpoint
    python -m tinygpt.profile infer --ckpt out/compile/ckpt.pt

    # Markdown table of every result vs its baseline, plus the two charts
    python -m tinygpt.profile report
    python -m tinygpt.profile plot

    # fp16 NaN case study plots
    python -m tinygpt.profile nan_plot --broken out/nan_repro --fixed out/nan_fixed

Training steps 0-4 are measured by tinygpt.train itself, which appends a row
to results/results.csv at the end of each run.
"""

from __future__ import annotations

import argparse
import csv
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

RESULT_FIELDS = [
    "timestamp", "ladder_step", "technique", "stage", "device", "tokens_per_sec",
    "peak_mem_mb", "val_loss", "latency_ms_per_token", "model_size_mb", "notes",
]
# Which row each step is compared against when reporting a speedup.
BASELINES = {"0": "0", "1": "0", "2": "0", "3": "0", "4": "0", "5": "5-ref", "6": "5-ref", "7": "7-ref"}


def device_name(device: str) -> str:
    if device.startswith("cuda") and torch.cuda.is_available():
        return torch.cuda.get_device_name(torch.device(device))
    return "cpu"


def append_result(path: str | Path, row: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        if new:
            w.writeheader()
        w.writerow({"timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"), **row})


def read_results(path: str | Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def latest_by_step(rows: list[dict]) -> dict[str, dict]:
    """Keep the most recent row for each ladder step."""
    out: dict[str, dict] = {}
    for row in rows:
        out[row["ladder_step"]] = row
    return out


# --------------------------------------------------------------------------- inference


def _sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def time_generation(model, prompt: torch.Tensor, new_tokens: int, use_cache: bool, device: str,
                    repeats: int) -> tuple[float, float, float]:
    """Return (tokens/sec over the whole batch, ms per generated position, peak MB)."""
    from tinygpt.generate import generate

    generate(model, prompt, min(8, new_tokens), greedy=True, use_cache=use_cache)  # warm-up
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    _sync(device)
    t0 = time.perf_counter()
    for _ in range(repeats):
        generate(model, prompt, new_tokens, greedy=True, use_cache=use_cache)
    _sync(device)
    elapsed = (time.perf_counter() - t0) / repeats
    peak = torch.cuda.max_memory_allocated() / 2**20 if device.startswith("cuda") else float("nan")
    return prompt.size(0) * new_tokens / elapsed, 1000 * elapsed / new_tokens, peak


def bench_inference(args: argparse.Namespace) -> None:
    from tinygpt.generate import load_model
    from tinygpt.quantize import model_size_mb, quantize_model

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    model, _ = load_model(args.ckpt, device)
    gen = torch.Generator().manual_seed(0)
    vocab = min(model.cfg.vocab_size, 50257)

    def prompt(batch: int, dev: str) -> torch.Tensor:
        return torch.randint(vocab, (batch, args.prompt_len), generator=gen).to(dev)

    dev_name = device_name(device)
    runs = [
        ("5-ref", "no KV-cache (reference)", 1, False),
        ("5", "KV-cache", 1, True),
        ("6", f"KV-cache + batch {args.batch}", args.batch, True),
    ]
    for step, name, batch, cache in runs:
        tps, ms, peak = time_generation(model, prompt(batch, device), args.new_tokens, cache, device, args.repeats)
        append_result(args.results, {
            "ladder_step": step, "technique": name, "stage": "inference", "device": dev_name,
            "tokens_per_sec": f"{tps:.0f}", "peak_mem_mb": f"{peak:.0f}", "latency_ms_per_token": f"{ms:.2f}",
            "notes": f"prompt {args.prompt_len}, {args.new_tokens} new tokens, greedy",
        })
        print(f"[{step}] {name}: {tps:,.0f} tok/s, {ms:.2f} ms/token")

    # int8 vs fp32, both on CPU with the KV-cache.
    cpu_model, _ = load_model(args.ckpt, "cpu")
    for step, name, m in [("7-ref", "fp32 CPU", cpu_model), ("7", "int8 dynamic CPU", quantize_model(cpu_model))]:
        tps, ms, _ = time_generation(m, prompt(1, "cpu"), args.cpu_new_tokens, True, "cpu", args.repeats)
        size = model_size_mb(m)
        append_result(args.results, {
            "ladder_step": step, "technique": name, "stage": "inference", "device": "cpu",
            "tokens_per_sec": f"{tps:.0f}", "latency_ms_per_token": f"{ms:.2f}", "model_size_mb": f"{size:.1f}",
            "notes": f"prompt {args.prompt_len}, {args.cpu_new_tokens} new tokens, threads {torch.get_num_threads()}",
        })
        print(f"[{step}] {name}: {tps:,.0f} tok/s, {ms:.2f} ms/token, {size:.1f} MB")


# --------------------------------------------------------------------------- report / plots


def _f(row: dict, key: str) -> float | None:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return None


def report(args: argparse.Namespace) -> None:
    rows = latest_by_step(read_results(args.results))
    print("| # | Technique | Stage | tokens/sec | vs baseline | peak mem (MB) | vs baseline | val loss | ms/token | size (MB) |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    order = ["0", "1", "2", "3", "4", "5-ref", "5", "6", "7-ref", "7"]
    for step in [s for s in order if s in rows] + [s for s in rows if s not in order]:
        r = rows[step]
        base = rows.get(BASELINES.get(step, ""), None)
        speed = mem = ""
        if base is not None and step != BASELINES.get(step):
            t, bt = _f(r, "tokens_per_sec"), _f(base, "tokens_per_sec")
            if t and bt:
                speed = f"{t / bt:.2f}x"
            m, bm = _f(r, "peak_mem_mb"), _f(base, "peak_mem_mb")
            if m and bm and m == m and bm == bm:  # skip NaN
                mem = f"{100 * (m - bm) / bm:+.0f}%"
        print(f"| {step} | {r['technique']} | {r['stage']} | {r['tokens_per_sec']} | {speed} | "
              f"{r['peak_mem_mb']} | {mem} | {r['val_loss']} | {r['latency_ms_per_token']} | {r['model_size_mb']} |")


def plot(args: argparse.Namespace) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = latest_by_step(read_results(args.results))
    train_steps = [s for s in ["0", "1", "2", "3", "4"] if s in rows]
    if not train_steps:
        raise SystemExit("no training rows in results.csv yet")
    labels = [f"{s}: {rows[s]['technique']}" for s in train_steps]
    out = Path(args.results).parent

    for key, ylabel, fname in [
        ("tokens_per_sec", "training tokens / sec", "throughput.png"),
        ("peak_mem_mb", "peak GPU memory (MB)", "peak_memory.png"),
    ]:
        values = [_f(rows[s], key) or 0.0 for s in train_steps]
        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.bar(labels, values, color="#4C78A8")
        ax.bar_label(bars, fmt="%.0f", padding=2)
        ax.set_ylabel(ylabel)
        ax.set_title(f"TinyGPT-Opt training ladder - {ylabel}")
        ax.spines[["top", "right"]].set_visible(False)
        plt.xticks(rotation=15, ha="right")
        fig.tight_layout()
        fig.savefig(out / fname, dpi=150)
        plt.close(fig)
        print(f"wrote {out / fname}")


def nan_plot(args: argparse.Namespace) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    for run, label in [(args.broken, "fp16, no loss scaling"), (args.fixed, "fp16 + loss scaling + clipping")]:
        rows = read_results(Path(run) / "metrics.csv")
        steps = [int(r["step"]) for r in rows]
        loss = [float(r["train_loss"]) for r in rows]
        ax.plot(steps, loss, label=label)
    ax.set_xlabel("step")
    ax.set_ylabel("train loss")
    ax.set_title("fp16 NaN case study - loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "nan_loss.png", dpi=150)
    plt.close(fig)

    gn_path = Path(args.broken) / "grad_norms.csv"
    if gn_path.exists():
        rows = read_results(gn_path)
        fig, ax = plt.subplots(figsize=(8, 4))
        for group in sorted({r["group"] for r in rows}):
            pts = [(int(r["step"]), float(r["norm"])) for r in rows if r["group"] == group]
            ax.plot([p[0] for p in pts], [p[1] for p in pts], label=group)
        ax.set_yscale("log")
        ax.set_xlabel("step")
        ax.set_ylabel("gradient L2 norm (log)")
        ax.set_title("fp16 NaN case study - gradient norm per layer (broken run)")
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(out / "nan_grad_norms.png", dpi=150)
        plt.close(fig)
    print(f"wrote plots to {out}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", default="results/results.csv")
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("infer", help="measure inference ladder steps 5-7")
    pi.add_argument("--ckpt", required=True)
    pi.add_argument("--device", default="auto")
    pi.add_argument("--prompt_len", type=int, default=32)
    pi.add_argument("--new_tokens", type=int, default=200)
    pi.add_argument("--cpu_new_tokens", type=int, default=64)
    pi.add_argument("--batch", type=int, default=16)
    pi.add_argument("--repeats", type=int, default=3)

    sub.add_parser("report", help="print a markdown results table")
    sub.add_parser("plot", help="write throughput and memory charts")

    pn = sub.add_parser("nan_plot", help="plots for the fp16 NaN case study")
    pn.add_argument("--broken", default="out/nan_repro")
    pn.add_argument("--fixed", default="out/nan_fixed")
    pn.add_argument("--out_dir", default="results")

    args = p.parse_args()
    {"infer": bench_inference, "report": report, "plot": plot, "nan_plot": nan_plot}[args.cmd](args)


if __name__ == "__main__":
    main()
