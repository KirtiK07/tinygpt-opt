"""int8 dynamic quantization for CPU inference.

    python -m tinygpt.quantize --ckpt out/compile/ckpt.pt --out out/compile/model_int8.pt

nn.Linear layers get int8 weights with activations quantized on the fly.
The embeddings are quantized too (int8 weights, per-row scales). That matters
because of weight tying: quantizing only the Linears gives lm_head its own int8
copy while the fp32 token embedding stays, and the "quantized" file ends up
larger than the original. With both quantized, the default 30M model goes from
about 115 MB to about 48 MB.

torch.ao.quantization is deprecated in recent PyTorch in favour of torchao's
quantize_ API. It still works (tested on 2.15); the plan names quantize_dynamic,
so it is kept here and the deprecation warnings are silenced.
"""

from __future__ import annotations

import argparse
import copy
import io
import warnings

import torch
import torch.nn as nn

from tinygpt.model import GPT


def quantize_model(model: GPT) -> nn.Module:
    from torch.ao.quantization import default_dynamic_qconfig, float_qparams_weight_only_qconfig, quantize_dynamic

    model = copy.deepcopy(model).cpu().eval()
    spec = {nn.Linear: default_dynamic_qconfig, nn.Embedding: float_qparams_weight_only_qconfig}
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*deprecated.*")
        return quantize_dynamic(model, spec, dtype=torch.qint8)


def model_size_mb(model: nn.Module) -> float:
    """Serialized state_dict size, which counts tied weights once."""
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.getbuffer().nbytes / 2**20


def main() -> None:
    from tinygpt.generate import load_model

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", default=None, help="where to save the quantized model (optional)")
    args = p.parse_args()

    model, _ = load_model(args.ckpt, "cpu")
    qmodel = quantize_model(model)
    print(f"fp32: {model_size_mb(model):.1f} MB  int8: {model_size_mb(qmodel):.1f} MB")
    if args.out:
        torch.save(qmodel, args.out)
        print(f"saved {args.out}")


if __name__ == "__main__":
    main()
