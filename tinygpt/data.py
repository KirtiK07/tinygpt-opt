"""Data pipeline: download, tokenize with tiktoken (GPT-2), write uint16 shards.

    python -m tinygpt.data --dataset shakespeare
    python -m tinygpt.data --dataset tinystories [--max_train_docs 500000]

Each dataset directory ends up with train.bin, val.bin (flat uint16 token
streams, documents separated by <|endoftext|>) and meta.json.
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np
import torch

TOKEN_DTYPE = np.uint16  # GPT-2 vocab (50257) fits in 16 bits

URLS = {
    "shakespeare": {
        "all": "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
    },
    "tinystories": {
        "train": "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt",
        "val": "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt",
    },
}


def download(url: str, dest: Path) -> Path:
    if dest.exists():
        print(f"using cached {dest}")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"downloading {url}")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(dest)
    return dest


def iter_docs(path: Path, max_docs: int | None = None) -> Iterator[str]:
    """Yield stories from a TinyStories file, where <|endoftext|> lines separate them."""
    buf: list[str] = []
    n = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip() == "<|endoftext|>":
                doc = "".join(buf).strip()
                buf = []
                if doc:
                    yield doc
                    n += 1
                    if max_docs is not None and n >= max_docs:
                        return
            else:
                buf.append(line)
    doc = "".join(buf).strip()
    if doc:
        yield doc


def write_tokens(docs: Iterable[str], out_path: Path, enc, batch_docs: int = 2048) -> int:
    """Tokenize docs, append <|endoftext|> after each, stream to a uint16 file."""
    eot = enc.eot_token
    total = 0
    batch: list[str] = []
    with open(out_path, "wb") as f:

        def flush() -> int:
            ids: list[int] = []
            for doc_ids in enc.encode_ordinary_batch(batch):
                ids.extend(doc_ids)
                ids.append(eot)
            np.asarray(ids, dtype=TOKEN_DTYPE).tofile(f)
            return len(ids)

        for doc in docs:
            batch.append(doc)
            if len(batch) >= batch_docs:
                total += flush()
                batch = []
        if batch:
            total += flush()
    return total


def prepare(dataset: str, out_dir: Path, max_train_docs: int | None = None) -> None:
    import tiktoken

    enc = tiktoken.get_encoding("gpt2")
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = out_dir / "raw"

    if dataset == "shakespeare":
        text = download(URLS[dataset]["all"], raw / "input.txt").read_text(encoding="utf-8")
        split = int(0.9 * len(text))
        counts = {
            "train": write_tokens([text[:split]], out_dir / "train.bin", enc),
            "val": write_tokens([text[split:]], out_dir / "val.bin", enc),
        }
    elif dataset == "tinystories":
        train_txt = download(URLS[dataset]["train"], raw / "train.txt")
        val_txt = download(URLS[dataset]["val"], raw / "val.txt")
        counts = {
            "train": write_tokens(iter_docs(train_txt, max_train_docs), out_dir / "train.bin", enc),
            "val": write_tokens(iter_docs(val_txt), out_dir / "val.bin", enc),
        }
    else:
        raise ValueError(f"unknown dataset {dataset!r}")

    meta = {"dataset": dataset, "tokenizer": "gpt2", "vocab_size": enc.n_vocab, "tokens": counts}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


def load_split(data_dir: str | Path, split: str) -> np.memmap:
    path = Path(data_dir) / f"{split}.bin"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run `python -m tinygpt.data` first")
    return np.memmap(path, dtype=TOKEN_DTYPE, mode="r")


def load_meta(data_dir: str | Path) -> dict | None:
    path = Path(data_dir) / "meta.json"
    return json.loads(path.read_text()) if path.exists() else None


def get_batch(
    data: np.ndarray, batch_size: int, block_size: int, device: str, generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample random windows; y is x shifted one token to the right."""
    if len(data) <= block_size:
        raise ValueError(f"dataset has {len(data)} tokens, need more than block_size={block_size}")
    ix = torch.randint(len(data) - block_size, (batch_size,), generator=generator).tolist()
    x = torch.stack([torch.from_numpy(data[i : i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1 : i + 1 + block_size].astype(np.int64)) for i in ix])
    if device.startswith("cuda"):
        return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    return x.to(device), y.to(device)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=sorted(URLS), required=True)
    p.add_argument("--out_dir", type=Path, default=None, help="default: data/<dataset>")
    p.add_argument("--max_train_docs", type=int, default=None, help="cap TinyStories training stories")
    args = p.parse_args()
    prepare(args.dataset, args.out_dir or Path("data") / args.dataset, args.max_train_docs)


if __name__ == "__main__":
    main()
