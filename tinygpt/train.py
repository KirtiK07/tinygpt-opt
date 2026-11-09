"""Config-driven training loop with AMP, gradient checkpointing and torch.compile.

    python -m tinygpt.train --config configs/base.yaml
    python -m tinygpt.train --config configs/amp.yaml --set max_steps=500 --set batch_size=64

Writes out/<run_name>/{config.yaml, metrics.csv, ckpt.pt, sample.txt} and
appends one summary row per run to results/results.csv.
"""

from __future__ import annotations

import argparse
import copy
import csv
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from tinygpt.data import get_batch, load_meta, load_split
from tinygpt.model import GPT, GPTConfig
from tinygpt.profile import append_result, device_name

DEFAULTS: dict = {
    "run_name": "base",
    "ladder_step": "0",
    "seed": 1337,
    "dataset": "tinystories",
    "data_dir": "data/tinystories",
    "out_dir": "out",
    "device": "auto",
    "model": {
        "vocab_size": 50304,
        "block_size": 256,
        "n_layer": 6,
        "n_head": 6,
        "n_embd": 384,
        "dropout": 0.0,
        "bias": True,
        "attn_impl": "manual",
        "grad_ckpt": False,
    },
    "batch_size": 32,
    "grad_accum": 1,
    "max_steps": 2000,
    "lr": 6e-4,
    "min_lr": 6e-5,
    "warmup_steps": 100,
    "weight_decay": 0.1,
    "beta1": 0.9,
    "beta2": 0.95,
    "grad_clip": 1.0,  # 0 disables clipping
    "amp": False,
    "amp_dtype": "float16",
    "grad_scaler": True,  # only used with amp + float16
    "compile": False,
    "eval_interval": 250,  # 0 = evaluate only at the end
    "eval_iters": 50,
    "log_interval": 10,
    "ckpt_interval": 0,  # 0 = checkpoint only at the end
    "resume": False,
    "timing_skip_steps": 10,  # warm-up steps left out of tokens/sec
    "detect_anomaly": False,
    "stop_on_nonfinite": True,
    "log_grad_norms": False,
    "deterministic": False,
    "sample_tokens": 200,
    "write_results": True,
    "results_csv": "results/results.csv",
}


class NonFiniteLossError(RuntimeError):
    pass


# --------------------------------------------------------------------------- config


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, val in override.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    """Load a YAML config. `inherit: other.yaml` stacks it on another config."""
    path = Path(path)
    raw = yaml.safe_load(path.read_text()) or {}
    parent = raw.pop("inherit", None)
    base = load_config(path.parent / parent) if parent else copy.deepcopy(DEFAULTS)
    cfg = deep_merge(base, raw)
    for item in overrides or []:
        key, _, value = item.partition("=")
        node = cfg
        *parents, leaf = key.split(".")
        for part in parents:
            node = node.setdefault(part, {})
        node[leaf] = yaml.safe_load(value)
    return cfg


# --------------------------------------------------------------------------- helpers


def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def get_lr(step: int, cfg: dict) -> float:
    """Linear warm-up, then cosine decay to min_lr."""
    if step < cfg["warmup_steps"]:
        return cfg["lr"] * (step + 1) / cfg["warmup_steps"]
    span = max(1, cfg["max_steps"] - cfg["warmup_steps"])
    progress = min(1.0, (step - cfg["warmup_steps"]) / span)
    return cfg["min_lr"] + 0.5 * (1 + math.cos(math.pi * progress)) * (cfg["lr"] - cfg["min_lr"])


def check_finite(loss: float, step: int) -> None:
    if not math.isfinite(loss):
        raise NonFiniteLossError(f"non-finite loss {loss} at step {step}")


def grad_norms_by_layer(model: torch.nn.Module) -> dict[str, float]:
    """L2 gradient norm per top-level group (blocks.0, blocks.1, ..., wte, ln_f)."""
    sq: dict[str, float] = {}
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        parts = name.split(".")
        group = ".".join(parts[:2]) if parts[0] == "blocks" else parts[0]
        sq[group] = sq.get(group, 0.0) + p.grad.detach().float().pow(2).sum().item()
    return {k: math.sqrt(v) for k, v in sq.items()}


@torch.no_grad()
def estimate_loss(model, data, cfg, device, device_type, amp_dtype) -> float:
    """Mean loss over a fixed set of batches, so every eval is comparable."""
    model.eval()
    gen = torch.Generator().manual_seed(cfg["seed"] + 1)
    losses = []
    for _ in range(cfg["eval_iters"]):
        x, y = get_batch(data, cfg["batch_size"], cfg["model"]["block_size"], device, gen)
        with torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=cfg["amp"]):
            _, loss, _ = model(x, y)
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


class CsvLog:
    def __init__(self, path: Path, fields: list[str], append: bool):
        new = not (append and path.exists())
        self.f = open(path, "a" if append else "w", newline="")
        self.w = csv.DictWriter(self.f, fieldnames=fields)
        if new:
            self.w.writeheader()

    def write(self, row: dict) -> None:
        self.w.writerow(row)
        self.f.flush()

    def close(self) -> None:
        self.f.close()


# --------------------------------------------------------------------------- training


def train(cfg: dict) -> dict:
    set_seed(cfg["seed"], cfg["deterministic"])
    device = resolve_device(cfg["device"])
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    out_dir = Path(cfg["out_dir"]) / cfg["run_name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    train_data = load_split(cfg["data_dir"], "train")
    val_data = load_split(cfg["data_dir"], "val")
    block_size = cfg["model"]["block_size"]

    model = GPT(GPTConfig(**cfg["model"])).to(device)
    optimizer = model.configure_optimizers(
        cfg["weight_decay"], cfg["lr"], (cfg["beta1"], cfg["beta2"]), device_type
    )
    amp_dtype = getattr(torch, cfg["amp_dtype"])
    use_scaler = cfg["amp"] and cfg["grad_scaler"] and amp_dtype == torch.float16
    scaler = torch.amp.GradScaler(device_type, enabled=use_scaler)
    batch_gen = torch.Generator().manual_seed(cfg["seed"])  # batch order, separate from init

    start_step = 0
    ckpt_path = out_dir / "ckpt.pt"
    if cfg["resume"] and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        batch_gen.set_state(ckpt["batch_gen"])
        start_step = ckpt["step"]
        print(f"resumed from {ckpt_path} at step {start_step}")

    train_model = torch.compile(model) if cfg["compile"] else model
    model.train()
    print(f"{model.num_params() / 1e6:.2f}M parameters on {device_name(device)}")

    def save_checkpoint(step: int) -> None:
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "batch_gen": batch_gen.get_state(),
                "step": step,
                "config": cfg,
            },
            ckpt_path,
        )

    fields = ["step", "train_loss", "val_loss", "lr", "tokens_per_sec", "peak_mem_mb", "grad_norm", "loss_scale"]
    metrics = CsvLog(out_dir / "metrics.csv", fields, append=start_step > 0)
    gnorm_log = CsvLog(out_dir / "grad_norms.csv", ["step", "group", "norm"], append=start_step > 0) \
        if cfg["log_grad_norms"] else None

    if device_type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    peak_mem = lambda: torch.cuda.max_memory_allocated() / 2**20 if device_type == "cuda" else float("nan")  # noqa: E731

    tokens_per_step = cfg["batch_size"] * block_size * cfg["grad_accum"]
    timed_tokens, timed_seconds = 0, 0.0
    losses: list[float] = []
    val_loss = float("nan")
    step = start_step

    try:
        with torch.autograd.set_detect_anomaly(cfg["detect_anomaly"]):
            for step in range(start_step, cfg["max_steps"]):
                lr = get_lr(step, cfg)
                for group in optimizer.param_groups:
                    group["lr"] = lr

                t0 = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                loss_sum = 0.0
                for _ in range(cfg["grad_accum"]):
                    x, y = get_batch(train_data, cfg["batch_size"], block_size, device, batch_gen)
                    with torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=cfg["amp"]):
                        _, loss, _ = train_model(x, y)
                        loss = loss / cfg["grad_accum"]
                    loss_value = loss.item()
                    if cfg["stop_on_nonfinite"]:
                        check_finite(loss_value, step)
                    scaler.scale(loss).backward()
                    loss_sum += loss_value

                scaler.unscale_(optimizer)  # no-op when the scaler is disabled
                if cfg["grad_clip"] > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"]).item()
                else:
                    grad_norm = math.sqrt(sum(v**2 for v in grad_norms_by_layer(model).values()))
                if gnorm_log is not None and step % cfg["log_interval"] == 0:
                    for group, norm in grad_norms_by_layer(model).items():
                        gnorm_log.write({"step": step, "group": group, "norm": norm})
                scaler.step(optimizer)
                scaler.update()
                if device_type == "cuda":
                    torch.cuda.synchronize()
                dt = time.perf_counter() - t0

                losses.append(loss_sum)
                if step - start_step >= cfg["timing_skip_steps"]:
                    timed_tokens += tokens_per_step
                    timed_seconds += dt

                last = step == cfg["max_steps"] - 1
                do_eval = (cfg["eval_interval"] > 0 and step > 0 and step % cfg["eval_interval"] == 0) or last
                if do_eval:
                    val_loss = estimate_loss(model, val_data, cfg, device, device_type, amp_dtype)
                if step % cfg["log_interval"] == 0 or do_eval:
                    tps = tokens_per_step / dt
                    metrics.write({
                        "step": step, "train_loss": f"{loss_sum:.4f}",
                        "val_loss": f"{val_loss:.4f}" if do_eval else "",
                        "lr": f"{lr:.3e}", "tokens_per_sec": f"{tps:.0f}", "peak_mem_mb": f"{peak_mem():.0f}",
                        "grad_norm": f"{grad_norm:.4f}", "loss_scale": f"{scaler.get_scale():.0f}" if use_scaler else "",
                    })
                    val_txt = f" | val {val_loss:.4f}" if do_eval else ""
                    print(f"step {step:5d} | loss {loss_sum:.4f}{val_txt} | lr {lr:.2e} | "
                          f"{tps:,.0f} tok/s | grad norm {grad_norm:.3f}")
                if cfg["ckpt_interval"] > 0 and step > 0 and step % cfg["ckpt_interval"] == 0:
                    save_checkpoint(step + 1)
    except NonFiniteLossError as err:
        (out_dir / "nonfinite.txt").write_text(f"{err}\n")
        print(f"STOPPED: {err}. Metrics so far are in {out_dir}.")
        raise
    finally:
        metrics.close()
        if gnorm_log is not None:
            gnorm_log.close()

    save_checkpoint(cfg["max_steps"])
    tokens_per_sec = timed_tokens / timed_seconds if timed_seconds > 0 else float("nan")
    summary = {
        "losses": losses,
        "val_loss": val_loss,
        "tokens_per_sec": tokens_per_sec,
        "peak_mem_mb": peak_mem(),
        "out_dir": str(out_dir),
    }
    print(f"done: val loss {val_loss:.4f} | {tokens_per_sec:,.0f} tok/s | peak mem {summary['peak_mem_mb']:.0f} MB")

    if cfg["write_results"]:
        append_result(cfg["results_csv"], {
            "ladder_step": cfg["ladder_step"], "technique": cfg["run_name"], "stage": "train",
            "device": device_name(device), "tokens_per_sec": f"{tokens_per_sec:.0f}",
            "peak_mem_mb": f"{summary['peak_mem_mb']:.0f}", "val_loss": f"{val_loss:.4f}",
            "notes": f"{cfg['max_steps']} steps, batch {cfg['batch_size']}x{block_size}",
        })

    meta = load_meta(cfg["data_dir"])
    if cfg["sample_tokens"] > 0 and meta and meta.get("tokenizer") == "gpt2":
        import tiktoken

        from tinygpt.generate import generate_texts

        enc = tiktoken.get_encoding("gpt2")
        gen = torch.Generator(device=device).manual_seed(cfg["seed"])
        text = generate_texts(model, enc, ["Once upon a time"], cfg["sample_tokens"], device,
                              temperature=0.8, top_k=200, generator=gen)[0]
        (out_dir / "sample.txt").write_text(text, encoding="utf-8")
        print("sample:\n" + text)
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="override a config value, e.g. --set max_steps=100 --set model.n_layer=4")
    args = p.parse_args()
    try:
        train(load_config(args.config, args.set))
    except NonFiniteLossError:
        sys.exit(2)


if __name__ == "__main__":
    main()
