"""int8 dynamic quantization must keep next-token predictions in agreement with fp32."""

import pytest
import torch
import torch.nn as nn

from tinygpt.quantize import model_size_mb, quantize_model

pytestmark = pytest.mark.skipif(
    not set(torch.backends.quantized.supported_engines) & {"fbgemm", "x86", "qnnpack", "onednn"},
    reason="no quantized CPU engine available",
)


def _pattern_batch(g, batch=16, length=32, vocab=97):
    """Sequences where each token is (3 * previous + 1) mod vocab: learnable, so predictions are confident."""
    seq = [torch.randint(0, vocab, (batch, 1), generator=g)]
    for _ in range(length):
        seq.append((3 * seq[-1] + 1) % vocab)
    seq = torch.cat(seq, dim=1)
    return seq[:, :-1], seq[:, 1:]


def test_int8_top1_agreement(make_model):
    model = make_model().train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    g = torch.Generator().manual_seed(0)
    for _ in range(150):
        x, y = _pattern_batch(g)
        _, loss, _ = model(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()

    qmodel = quantize_model(model)
    assert not any(type(m) is nn.Linear for m in qmodel.modules())  # every Linear was swapped

    x, _ = _pattern_batch(torch.Generator().manual_seed(123), batch=32)
    with torch.no_grad():
        ref = model(x)[0].argmax(-1)
        q = qmodel(x)[0].argmax(-1)
    agreement = (ref == q).float().mean().item()
    assert agreement > 0.95, agreement


def test_int8_model_is_smaller():
    from tinygpt.model import GPT, GPTConfig

    torch.manual_seed(0)
    model = GPT(GPTConfig(n_layer=2)).eval()  # real vocab, so the tied embedding dominates
    assert model_size_mb(quantize_model(model)) < 0.5 * model_size_mb(model)
