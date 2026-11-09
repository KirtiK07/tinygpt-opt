"""GPT-style decoder-only transformer, written from scratch.

Attention runs either as a hand-written matmul + mask + softmax ("manual") or
through PyTorch's fused kernel F.scaled_dot_product_attention ("fused"). Both
paths support a key/value cache for fast autoregressive decoding, and the
blocks can be gradient-checkpointed to trade compute for memory.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

# One (keys, values) pair per layer, each shaped (batch, heads, seq, head_dim).
KVCache = tuple[torch.Tensor, torch.Tensor]

ATTN_IMPLS = ("manual", "fused")


@dataclass
class GPTConfig:
    vocab_size: int = 50304  # GPT-2's 50257 tokens, padded to a multiple of 64
    block_size: int = 256  # maximum context length
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.0
    bias: bool = True
    attn_impl: str = "manual"  # "manual" or "fused"
    grad_ckpt: bool = False  # recompute block activations in backward


class LayerNorm(nn.Module):
    """LayerNorm with an optional bias.

    Uses F.layer_norm so that torch.autocast keeps it in fp32 under mixed
    precision, which matters for the fp16 stability story.
    """

    def __init__(self, ndim: int, bias: bool):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        if cfg.n_embd % cfg.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        if cfg.attn_impl not in ATTN_IMPLS:
            raise ValueError(f"attn_impl must be one of {ATTN_IMPLS}")
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.dropout = cfg.dropout
        self.attn_impl = cfg.attn_impl
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)
        # True where attention is allowed: query i may look at keys 0..i.
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size, dtype=torch.bool))
        self.register_buffer("causal_mask", mask.view(1, 1, cfg.block_size, cfg.block_size), persistent=False)

    def forward(self, x: torch.Tensor, past: KVCache | None = None) -> tuple[torch.Tensor, KVCache]:
        B, T, C = x.shape
        head_dim = C // self.n_head
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_dim).transpose(1, 2)
        if past is not None:
            k = torch.cat([past[0], k], dim=2)
            v = torch.cat([past[1], v], dim=2)
        present = (k, v)

        # With a cache, the T new queries sit at absolute positions
        # offset..Tk-1, so take the matching rows of the causal mask.
        Tk = k.size(2)
        offset = Tk - T
        drop_p = self.dropout if self.training else 0.0

        if self.attn_impl == "fused":
            if past is None:
                y = F.scaled_dot_product_attention(q, k, v, dropout_p=drop_p, is_causal=True)
            else:
                mask = self.causal_mask[:, :, offset:Tk, :Tk]
                y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=drop_p)
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(head_dim))
            att = att.masked_fill(~self.causal_mask[:, :, offset:Tk, :Tk], float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y)), present


class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class Block(nn.Module):
    """Pre-norm transformer block: x + attn(ln(x)), then x + mlp(ln(x))."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln_1 = LayerNorm(cfg.n_embd, cfg.bias)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = LayerNorm(cfg.n_embd, cfg.bias)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor, past: KVCache | None = None) -> tuple[torch.Tensor, KVCache]:
        a, present = self.attn(self.ln_1(x), past)
        x = x + a
        x = x + self.mlp(self.ln_2(x))
        return x, present


def _block_hidden(block: Block, x: torch.Tensor) -> torch.Tensor:
    return block(x)[0]


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.wpe = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = LayerNorm(cfg.n_embd, cfg.bias)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight  # weight tying

        self.apply(self._init_weights)
        # GPT-2 style: scale residual projections by 1/sqrt(2 * n_layer).
        for name, p in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def set_attn_impl(self, impl: str) -> None:
        if impl not in ATTN_IMPLS:
            raise ValueError(f"attn_impl must be one of {ATTN_IMPLS}")
        self.cfg.attn_impl = impl
        for block in self.blocks:
            block.attn.attn_impl = impl

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        past_kv: list[KVCache] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, list[KVCache] | None]:
        """Return (logits, loss, present_kv).

        loss is None without targets; present_kv is None unless use_cache.
        """
        B, T = idx.shape
        past_len = past_kv[0][0].size(2) if past_kv is not None else 0
        if past_len + T > self.cfg.block_size:
            raise ValueError(f"sequence length {past_len + T} exceeds block_size {self.cfg.block_size}")

        pos = torch.arange(past_len, past_len + T, device=idx.device)
        # .contiguous(): sliced inputs are strided, which the int8 embedding kernel rejects.
        x = self.drop(self.wte(idx.contiguous()) + self.wpe(pos))

        presents: list[KVCache] | None = [] if use_cache else None
        ckpt = self.cfg.grad_ckpt and self.training and not use_cache and past_kv is None
        for i, block in enumerate(self.blocks):
            if ckpt:
                x = checkpoint(_block_hidden, block, x, use_reentrant=False)
            else:
                x, present = block(x, past_kv[i] if past_kv is not None else None)
                if presents is not None:
                    presents.append(present)

        logits = self.lm_head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(), targets.reshape(-1), ignore_index=-1
            )
        return logits, loss, presents

    def configure_optimizers(
        self, weight_decay: float, lr: float, betas: tuple[float, float], device_type: str
    ) -> torch.optim.AdamW:
        """AdamW with weight decay on matrices only (not biases, norms)."""
        decay = [p for p in self.parameters() if p.requires_grad and p.dim() >= 2]
        no_decay = [p for p in self.parameters() if p.requires_grad and p.dim() < 2]
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(groups, lr=lr, betas=betas, fused=device_type == "cuda")
