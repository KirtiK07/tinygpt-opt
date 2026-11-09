"""Fused attention (F.scaled_dot_product_attention) must match the manual version."""

import copy

import pytest
import torch

ATOL = 1e-5  # fp32


def _pair(make_model):
    manual = make_model(attn_impl="manual")
    fused = copy.deepcopy(manual)
    fused.set_attn_impl("fused")
    return manual, fused


def test_fused_matches_manual_forward(make_model):
    manual, fused = _pair(make_model)
    idx = torch.randint(0, 97, (3, 32), generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        a, _, _ = manual(idx)
        b, _, _ = fused(idx)
    assert torch.allclose(a, b, atol=ATOL), (a - b).abs().max()


def test_fused_matches_manual_backward(make_model):
    manual, fused = _pair(make_model)
    manual.train()
    fused.train()
    g = torch.Generator().manual_seed(2)
    idx = torch.randint(0, 97, (2, 32), generator=g)
    tgt = torch.randint(0, 97, (2, 32), generator=g)
    _, la, _ = manual(idx, tgt)
    _, lb, _ = fused(idx, tgt)
    la.backward()
    lb.backward()
    assert torch.allclose(la, lb, atol=ATOL)
    for (name, pa), (_, pb) in zip(manual.named_parameters(), fused.named_parameters()):
        assert torch.allclose(pa.grad, pb.grad, atol=ATOL), name


@pytest.mark.parametrize("past_len,new_len", [(10, 1), (10, 5), (0, 7)])
def test_attention_module_with_cache_offset(make_model, past_len, new_len):
    """With cached keys, the new queries must use the shifted causal mask."""
    manual, fused = _pair(make_model)
    attn_m, attn_f = manual.blocks[0].attn, fused.blocks[0].attn
    g = torch.Generator().manual_seed(3)
    x = torch.randn(2, past_len + new_len, 32, generator=g)
    with torch.no_grad():
        full, _ = attn_m(x)  # reference: whole sequence at once, no cache
        for attn in (attn_m, attn_f):
            past = attn(x[:, :past_len])[1] if past_len else None
            out, _ = attn(x[:, past_len:], past)
            assert torch.allclose(out, full[:, past_len:], atol=ATOL), attn.attn_impl
