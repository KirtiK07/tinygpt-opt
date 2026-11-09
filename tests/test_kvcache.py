"""KV-cache and batched generation must give exactly the reference tokens."""

import pytest
import torch

from tinygpt.generate import generate


@pytest.mark.parametrize("impl", ["manual", "fused"])
@pytest.mark.parametrize("new_tokens", [20, 45])  # 45 runs past block_size=32
def test_greedy_identical_with_and_without_cache(make_model, impl, new_tokens):
    model = make_model(attn_impl=impl)
    prompt = torch.randint(0, 97, (1, 5), generator=torch.Generator().manual_seed(0))
    cached = generate(model, prompt, new_tokens, greedy=True, use_cache=True)
    uncached = generate(model, prompt, new_tokens, greedy=True, use_cache=False)
    assert cached.shape == (1, 5 + new_tokens)
    assert torch.equal(cached, uncached)


def test_incremental_logits_match_full_forward(make_model):
    model = make_model()
    idx = torch.randint(0, 97, (2, 24), generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        full, _, _ = model(idx)
        logits, _, past = model(idx[:, :10], use_cache=True)
        steps = [logits]
        for t in range(10, 24):
            logits, _, past = model(idx[:, t : t + 1], past_kv=past, use_cache=True)
            steps.append(logits)
    assert torch.allclose(torch.cat(steps, dim=1), full, atol=1e-5)


def test_batched_generation_matches_one_at_a_time(make_model):
    model = make_model()
    prompts = torch.randint(0, 97, (4, 6), generator=torch.Generator().manual_seed(2))
    batched = generate(model, prompts, 15, greedy=True)
    for i in range(prompts.size(0)):
        single = generate(model, prompts[i : i + 1], 15, greedy=True)
        assert torch.equal(batched[i : i + 1], single), i


def test_padding_ids_never_sampled(make_model):
    model = make_model()
    prompt = torch.zeros(8, 1, dtype=torch.long)
    out = generate(model, prompt, 30, valid_vocab=50, generator=torch.Generator().manual_seed(0))
    assert int(out[:, 1:].max()) < 50
