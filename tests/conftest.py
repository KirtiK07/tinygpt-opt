import copy

import numpy as np
import pytest
import torch

from tinygpt.model import GPT, GPTConfig
from tinygpt.train import DEFAULTS

TINY = dict(vocab_size=97, block_size=32, n_layer=2, n_head=4, n_embd=32, dropout=0.0)


@pytest.fixture
def make_model():
    """Build a small model with fixed weights; extra kwargs override GPTConfig."""

    def _make(seed: int = 0, **overrides) -> GPT:
        torch.manual_seed(seed)
        return GPT(GPTConfig(**{**TINY, **overrides})).eval()

    return _make


@pytest.fixture
def tiny_data_dir(tmp_path):
    """train.bin / val.bin of random token ids, enough for a few hundred steps."""
    rng = np.random.default_rng(0)
    for split, n in [("train", 20_000), ("val", 4_000)]:
        rng.integers(0, TINY["vocab_size"], n, dtype=np.uint16).tofile(tmp_path / f"{split}.bin")
    return tmp_path


@pytest.fixture
def tiny_train_cfg(tiny_data_dir, tmp_path):
    cfg = copy.deepcopy(DEFAULTS)
    cfg.update(
        data_dir=str(tiny_data_dir), out_dir=str(tmp_path / "out"), device="cpu",
        batch_size=4, max_steps=50, warmup_steps=5, lr=1e-3, min_lr=1e-4,
        eval_interval=0, eval_iters=2, log_interval=10, write_results=False, sample_tokens=0,
        timing_skip_steps=0,
    )
    cfg["model"] = {**cfg["model"], **TINY}
    return cfg
