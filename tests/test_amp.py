"""Mixed precision must give (nearly) the same loss as fp32, and stay finite."""

import math

import pytest
import torch

from tinygpt.train import NonFiniteLossError, check_finite

CUDA = torch.cuda.is_available()


def test_amp_loss_close_to_fp32(make_model):
    device = "cuda" if CUDA else "cpu"
    dtype = torch.float16 if CUDA else torch.bfloat16  # CPU autocast uses bf16
    model = make_model().to(device)
    g = torch.Generator().manual_seed(0)
    x = torch.randint(0, 97, (4, 32), generator=g).to(device)
    y = torch.randint(0, 97, (4, 32), generator=g).to(device)
    with torch.no_grad():
        _, loss32, _ = model(x, y)
        with torch.autocast(device_type=device, dtype=dtype):
            _, loss_amp, _ = model(x, y)
    diff = abs(loss32.item() - loss_amp.item())
    # fp16 on GPU: absolute 1e-2. bf16 keeps fewer mantissa bits, so CPU uses relative 1e-2.
    limit = 1e-2 if CUDA else 1e-2 * loss32.item()
    assert diff < limit, (loss32.item(), loss_amp.item())


@pytest.mark.skipif(not CUDA, reason="fp16 GradScaler training needs a GPU")
def test_fp16_training_with_scaler_stays_finite(make_model):
    model = make_model().cuda().train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda")
    g = torch.Generator().manual_seed(0)
    for _ in range(20):
        x = torch.randint(0, 97, (8, 32), generator=g).cuda()
        with torch.autocast("cuda", dtype=torch.float16):
            _, loss, _ = model(x, x)
        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        assert math.isfinite(loss.item())


def test_nonfinite_guard():
    check_finite(2.5, step=0)
    for bad in (float("nan"), float("inf")):
        with pytest.raises(NonFiniteLossError):
            check_finite(bad, step=7)
