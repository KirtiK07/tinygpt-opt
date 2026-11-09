"""Sampling with optional KV-cache and batched generation.

    python -m tinygpt.generate --ckpt out/base/ckpt.pt --prompt "Once upon a time" --num_samples 3
    python -m tinygpt.generate --ckpt out/base/ckpt.pt --prompt "A cat" --prompt "The sun" --no_cache
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from tinygpt.model import GPT, GPTConfig


def load_model(ckpt_path: str | Path, device: str = "cpu", attn_impl: str | None = None) -> tuple[GPT, dict]:
    # Our own checkpoints also hold optimizer/RNG state, so full unpickling is needed.
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_cfg = dict(ckpt["config"]["model"])
    model_cfg["grad_ckpt"] = False
    if attn_impl is not None:
        model_cfg["attn_impl"] = attn_impl
    model = GPT(GPTConfig(**model_cfg))
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), ckpt


@torch.no_grad()
def generate(
    model: GPT,
    idx: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    greedy: bool = False,
    use_cache: bool = True,
    valid_vocab: int | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Extend idx (batch, T) by max_new_tokens tokens.

    With use_cache, the prompt is run once and each later step feeds only the
    newest token, reusing cached keys/values. Once the sequence outgrows the
    context window the cache no longer applies (the window slides), so it falls
    back to recomputing the last block_size tokens - exactly what the uncached
    path does, which keeps both paths token-for-token identical.
    """
    model.eval()
    block_size = model.cfg.block_size
    past = None
    step_input = idx
    for _ in range(max_new_tokens):
        if use_cache and idx.size(1) <= block_size:
            logits, _, past = model(step_input, past_kv=past, use_cache=True)
        else:
            past = None
            logits, _, _ = model(idx[:, -block_size:])
        logits = logits[:, -1, :].float()
        if valid_vocab is not None:
            logits[:, valid_vocab:] = float("-inf")  # never emit padding ids

        if greedy:
            nxt = logits.argmax(dim=-1, keepdim=True)
        else:
            logits = logits / max(temperature, 1e-8)
            if top_k is not None:
                kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, [-1]]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            probs = torch.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1, generator=generator)

        idx = torch.cat([idx, nxt], dim=1)
        step_input = nxt
    return idx


def generate_texts(
    model: GPT, enc, prompts: list[str], max_new_tokens: int, device: str, **kwargs
) -> list[str]:
    """Batched generation for many prompts.

    Prompts with the same token length run together as one batch, so no
    padding or attention masks are needed and results match one-at-a-time
    generation exactly.
    """
    encoded = [enc.encode_ordinary(p) or [enc.eot_token] for p in prompts]
    groups: dict[int, list[int]] = {}
    for i, ids in enumerate(encoded):
        groups.setdefault(len(ids), []).append(i)

    outputs: list[str] = [""] * len(prompts)
    for members in groups.values():
        batch = torch.tensor([encoded[i] for i in members], dtype=torch.long, device=device)
        out = generate(model, batch, max_new_tokens, valid_vocab=enc.n_vocab, **kwargs)
        for row, i in zip(out.tolist(), members):
            outputs[i] = enc.decode(row)
    return outputs


def main() -> None:
    import tiktoken

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--prompt", action="append", default=None, help="repeat for batched generation")
    p.add_argument("--num_samples", type=int, default=1, help="samples per prompt")
    p.add_argument("--max_new_tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=200)
    p.add_argument("--greedy", action="store_true")
    p.add_argument("--no_cache", action="store_true")
    p.add_argument("--quantize", action="store_true", help="int8 dynamic quantization (CPU only)")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args()

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if args.quantize:
        device = "cpu"
    model, _ = load_model(args.ckpt, device)
    if args.quantize:
        from tinygpt.quantize import quantize_model

        model = quantize_model(model)

    enc = tiktoken.get_encoding("gpt2")
    prompts = [pr for pr in (args.prompt or ["Once upon a time"]) for _ in range(args.num_samples)]
    gen = torch.Generator(device=device).manual_seed(args.seed)
    texts = generate_texts(
        model, enc, prompts, args.max_new_tokens, device,
        temperature=args.temperature, top_k=args.top_k, greedy=args.greedy,
        use_cache=not args.no_cache, generator=gen,
    )
    for text in texts:
        print(text)
        print("-" * 60)


if __name__ == "__main__":
    main()
