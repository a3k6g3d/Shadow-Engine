"""Reads tensors out of a Hugging Face safetensors checkpoint by HTTP range requests, so the 360 GB Qwen3.8-Flash-Next
checkpoint never has to be downloaded to run its reference on a few layers.

    from hf_range import Checkpoint
    ck = Checkpoint("Qwen/Qwen3.8-Flash-Next", cache="I:/shadow-ref")
    w = ck.tensor("model.language_model.layers.0.linear_attn.in_proj_qkv.weight")      # whole, as a torch tensor
    e = ck.rows("model.language_model.layers.0.mlp.experts.down_proj", [5, 17])         # rows of dim 0 only

Every byte fetched is kept under `cache`, keyed by file and range, so a second run fetches nothing.  A safetensors file
is 8 bytes (header length N, little-endian u64), N bytes of JSON {name: {dtype, shape, data_offsets}}, then the data;
data_offsets are relative to the end of the header.
"""
import hashlib
import json
import pathlib
import struct
import sys
import time

import requests
import torch

DTYPES = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32, "I64": torch.int64, "I32": torch.int32,
          "U8": torch.uint8, "I8": torch.int8, "BOOL": torch.bool, "F64": torch.float64}
SIZES = {"BF16": 2, "F16": 2, "F32": 4, "I64": 8, "I32": 4, "U8": 1, "I8": 1, "BOOL": 1, "F64": 8}


class Checkpoint:
    def __init__(self, repo: str, cache: str, revision: str = "main"):
        self.repo, self.rev = repo, revision
        self.cache = pathlib.Path(cache)
        (self.cache / "ranges").mkdir(parents=True, exist_ok=True)
        (self.cache / "headers").mkdir(parents=True, exist_ok=True)
        self.s = requests.Session()
        self.fetched = 0          # bytes actually transferred this run
        self.index = json.loads(self.small("model.safetensors.index.json"))["weight_map"]
        self._headers = {}

    def url(self, file: str) -> str:
        return f"https://huggingface.co/{self.repo}/resolve/{self.rev}/{file}"

    def small(self, file: str) -> bytes:
        """A small whole file (index, config, tokenizer), cached."""
        p = self.cache / file.replace("/", "_")
        if not p.exists():
            r = self.s.get(self.url(file), timeout=120)
            r.raise_for_status()
            p.write_bytes(r.content)
            self.fetched += len(r.content)
        return p.read_bytes()

    def _range(self, file: str, start: int, end: int) -> bytes:
        """Bytes [start, end) of a file, from the cache or the network."""
        key = hashlib.sha256(f"{file}:{start}:{end}".encode()).hexdigest()[:32]
        p = self.cache / "ranges" / key
        if p.exists() and p.stat().st_size == end - start:
            return p.read_bytes()
        for attempt in range(6):
            try:
                r = self.s.get(self.url(file), headers={"Range": f"bytes={start}-{end - 1}"}, timeout=300)
                if r.status_code not in (200, 206):
                    raise IOError(f"HTTP {r.status_code}")
                data = r.content
                if r.status_code == 200:      # the server ignored the range: slice it ourselves
                    data = data[start:end]
                if len(data) != end - start:
                    raise IOError(f"short read {len(data)} of {end - start}")
                break
            except Exception as e:           # network hiccups: back off and retry
                if attempt == 5:
                    raise
                print(f"  retry {attempt + 1} for {file}[{start}:{end}]: {e}", file=sys.stderr)
                time.sleep(2 ** attempt)
        tmp = p.with_suffix(".tmp")
        tmp.write_bytes(data)
        tmp.replace(p)
        self.fetched += len(data)
        return data

    def header(self, file: str):
        if file not in self._headers:
            hp = self.cache / "headers" / (file + ".json")
            if hp.exists():
                self._headers[file] = json.loads(hp.read_text())
            else:
                n = struct.unpack("<Q", self._range(file, 0, 8))[0]
                h = json.loads(self._range(file, 8, 8 + n))
                h["__data_start__"] = 8 + n
                hp.write_text(json.dumps(h))
                self._headers[file] = h
        return self._headers[file]

    def info(self, name: str):
        file = self.index[name]
        h = self.header(file)
        t = h[name]
        return file, t["dtype"], t["shape"], h["__data_start__"] + t["data_offsets"][0], t["data_offsets"][1] - t["data_offsets"][0]

    def tensor(self, name: str) -> torch.Tensor:
        file, dt, shape, start, nbytes = self.info(name)
        b = bytearray(self._range(file, start, start + nbytes))
        return torch.frombuffer(b, dtype=DTYPES[dt]).reshape(shape).clone()

    def rows(self, name: str, idx) -> torch.Tensor:
        """Rows `idx` along dimension 0, fetched one contiguous range each (neighbours are merged)."""
        file, dt, shape, start, nbytes = self.info(name)
        row = nbytes // shape[0]
        idx = sorted(set(int(i) for i in idx))
        out = {}
        i = 0
        while i < len(idx):
            j = i
            while j + 1 < len(idx) and idx[j + 1] == idx[j] + 1 and (idx[j + 1] - idx[i] + 1) * row <= 64 << 20:
                j += 1
            lo, hi = idx[i], idx[j] + 1
            b = bytearray(self._range(file, start + lo * row, start + hi * row))
            t = torch.frombuffer(b, dtype=DTYPES[dt]).reshape([hi - lo] + shape[1:])
            for k in range(lo, hi):
                out[k] = t[k - lo].clone()
            i = j + 1
        return torch.stack([out[k] for k in idx]), idx


if __name__ == "__main__":
    ck = Checkpoint("Qwen/Qwen3.8-Flash-Next", cache=sys.argv[1] if len(sys.argv) > 1 else "I:/shadow-ref")
    for n in sys.argv[2:]:
        print(n, ck.info(n))
    print("fetched", ck.fetched, "bytes")
