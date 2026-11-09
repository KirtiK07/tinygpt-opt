"""Output shapes, parameter count and the causal mask."""

import pytest
import torch

from tinygpt.model import GPT, GPTConfig


def test_output_shapes(make_model):
    model = make_model()
    idx = torch.randint(0, 97, (3, 20))
    logits, loss, present = model(idx, idx)
    assert logits.shape == (3, 20, 97)
    assert loss.dim() == 0
    assert present is None
    logits, loss, present = model(idx, use_cache=True)
    assert loss is None
    assert len(present) == 2 and present[0][0].shape == (3, 4, 20, 8)


def test_rejects_too_long_sequence(make_model):
    with pytest.raises(ValueError):
        make_model()(torch.zeros(1, 33, dtype=torch.long))


@pytest.mark.parametrize("impl", ["manual", "fused"])
def test_no_token_attends_to_the_future(make_model, impl):
    model = make_model(attn_impl=impl)
    g = torch.Generator().manual_seed(0)
    idx = torch.randint(0, 97, (2, 32), generator=g)
    changed = idx.clone()
    changed[:, 16:] = torch.randint(0, 97, (2, 16), generator=g)
    with torch.no_grad():
        a, _, _ = model(idx)
        b, _, _ = model(changed)
    assert torch.equal(a[:, :16], b[:, :16])  # earlier positions never see later tokens
    assert not torch.allclose(a[:, 16:], b[:, 16:])


def test_weight_tying(make_model):
    model = make_model()
    assert model.lm_head.weight is model.wte.weight


def test_default_model_size():
    n = GPT(GPTConfig()).num_params()
    assert 10e6 < n < 31e6, n  # plan: about 10-30M parameters
