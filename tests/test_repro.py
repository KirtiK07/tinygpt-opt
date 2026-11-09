"""Same seed + config must reproduce the same losses; checkpointing must not change them."""

import copy
from pathlib import Path

import torch

from tinygpt.train import load_config, train

ROOT = Path(__file__).resolve().parents[1]


def _run(cfg, name, **model_overrides):
    cfg = copy.deepcopy(cfg)
    cfg["run_name"] = name
    cfg["model"].update(model_overrides)
    return train(cfg)["losses"]


def test_same_seed_same_losses(tiny_train_cfg):
    a = _run(tiny_train_cfg, "run_a")
    b = _run(tiny_train_cfg, "run_b")
    assert len(a) == 50
    assert a == b  # bit-identical for the first 50 steps


def test_different_seed_different_losses(tiny_train_cfg):
    a = _run(tiny_train_cfg, "seed_a")
    other = copy.deepcopy(tiny_train_cfg)
    other["seed"] = 7
    b = _run(other, "seed_b")
    assert a != b


def test_gradient_checkpointing_matches(tiny_train_cfg):
    base = _run(tiny_train_cfg, "no_ckpt")
    ckpt = _run(tiny_train_cfg, "ckpt", grad_ckpt=True)
    assert torch.allclose(torch.tensor(base), torch.tensor(ckpt), atol=1e-5, rtol=0)


def test_ladder_configs_stack():
    cfg = load_config(ROOT / "configs" / "compile.yaml")
    assert cfg["amp"] and cfg["compile"]
    assert cfg["model"]["attn_impl"] == "fused" and cfg["model"]["grad_ckpt"]
    base = load_config(ROOT / "configs" / "base.yaml")
    for key in ("max_steps", "batch_size", "seed", "data_dir"):
        assert cfg[key] == base[key], key  # same data, steps and seed on every rung
    strip = lambda m: {k: v for k, v in m.items() if k not in ("attn_impl", "grad_ckpt")}  # noqa: E731
    assert strip(cfg["model"]) == strip(base["model"])


def test_config_overrides():
    cfg = load_config(ROOT / "configs" / "base.yaml", ["max_steps=7", "model.n_layer=2"])
    assert cfg["max_steps"] == 7 and cfg["model"]["n_layer"] == 2
