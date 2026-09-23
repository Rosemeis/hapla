"""Shared small fixtures, temporary files, and CLI calls for unittest."""

__author__ = "Jonas Meisner"

import ctypes as ct
import os
import subprocess
import sys
import tempfile
import unittest
from hashlib import sha256
from pathlib import Path

import numpy as np

import hapla
from hapla import struct, vcf_cy
from hapla.formats import MAGIC
from hapla.identity import writeIdentity


### Give each file-based test its own automatically cleaned directory
class TemporaryTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(prefix="hapla-tests-")
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)


### Run the same package as the parent process, without a checkout shadowing the wheel
def command(*args, success=True, cwd=None):
    package = Path(hapla.__file__).resolve().parent.parent
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", PYTHONPATH=str(package))
    result = subprocess.run(
        [sys.executable, "-m", "hapla.main", *map(str, args)],
        cwd=tempfile.gettempdir() if cwd is None else cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if (result.returncode == 0) != success:
        raise AssertionError(result.stdout + result.stderr)
    return result


### Read named values from the human-readable run log
def readLog(pfx):
    rows = [line.strip().partition(":") for line in Path(f"{pfx}.log").read_text().splitlines()]
    return {key: val.strip() for key, sep, val in rows if sep and val.strip()}


### Write a tiny phased/unphased fixture with optional FORMAT fields
def writeVcf(path, rows, *, fields="GT", samples=("A", "B", "C")):
    header = (
        "##fileformat=VCFv4.3\n##contig=<ID=1,length=10000000>\n"
        "##contig=<ID=2,length=10000000>\n"
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
        '##FORMAT=<ID=DP,Number=1,Type=Integer,Description="Depth">\n'
    )
    header += "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t" + "\t".join(samples) + "\n"
    with open(path, "w") as handle:
        handle.write(header)
        for chrom, pos, calls in rows:
            handle.write(
                f"{chrom}\t{pos}\t.\tA\tG\t.\tPASS\t.\t{fields}\t" + "\t".join(calls) + "\n"
            )


### Build fixtures through the reader's linked HTSlib without shipping a BCF writer
def toBcf(vcf, bcf, index=False):
    lib = ct.CDLL(vcf_cy.__file__)
    ptr, num, txt = ct.c_void_p, ct.c_int, ct.c_char_p
    for name, result, args in (
        ("hts_open", ptr, (txt, txt)),
        ("hts_close", num, (ptr,)),
        ("bcf_hdr_read", ptr, (ptr,)),
        ("bcf_hdr_write", num, (ptr, ptr)),
        ("bcf_hdr_destroy", None, (ptr,)),
        ("bcf_init", ptr, ()),
        ("bcf_destroy", None, (ptr,)),
        ("bcf_read", num, (ptr, ptr, ptr)),
        ("bcf_write", num, (ptr, ptr, ptr)),
        ("bcf_index_build", num, (txt, num)),
    ):
        fun = getattr(lib, name)
        fun.restype, fun.argtypes = result, args
    if Path(bcf).exists() or Path(bcf).is_symlink():
        raise ValueError("BCF fixture must be a new file")
    src = dst = hdr = rec = None
    try:
        src = lib.hts_open(os.fsencode(vcf), b"r")
        if not src:
            raise OSError("Cannot open VCF fixture")
        hdr, rec = lib.bcf_hdr_read(src), lib.bcf_init()
        if not hdr or not rec:
            raise OSError("Cannot read fixture header or allocate record")
        dst = lib.hts_open(os.fsencode(bcf), b"wb")
        if not dst or lib.bcf_hdr_write(dst, hdr) < 0:
            raise OSError("Cannot write BCF fixture header")
        while True:
            status = lib.bcf_read(src, hdr, rec)
            if status == -1:
                break
            if status < -1 or lib.bcf_write(dst, hdr, rec) < 0:
                raise OSError("Cannot convert VCF fixture")
        status = lib.hts_close(dst)
        dst = None
        if status < 0 or (index and lib.bcf_index_build(os.fsencode(bcf), 14) < 0):
            raise OSError("Cannot finish BCF fixture")
    finally:
        if dst:
            lib.hts_close(dst)
        if src:
            lib.hts_close(src)
        if hdr:
            lib.bcf_hdr_destroy(hdr)
        if rec:
            lib.bcf_destroy(rec)


### Write a complete versioned assignment bundle
def writeClusters(root, name, Z, c):
    pfx = root / name
    Path(f"{pfx}.bca").write_bytes(MAGIC + Z.tobytes())
    Path(f"{pfx}.ids").write_text("".join(f"s{i}\n" for i in range(Z.shape[1] // 2)))
    Path(f"{pfx}.win").write_text(
        "#CHROM START END LENGTH SIZE K\n"
        + "".join(f"1 {i + 1} {i + 1} 0 1 {K}\n" for i, K in enumerate(np.diff(c)))
    )
    out = {s: Path(f"{pfx}{s}") for s in (".bca", ".ids", ".win", ".ref.json")}
    writeIdentity(out, sha256(Z.tobytes() + c.tobytes()).hexdigest(), len(Z), int(c[-1]))
    return pfx


### Build fully observed dosages and an independent dense PCA target
def structureFixture(N=17, seed=410):
    rng = np.random.default_rng(seed)
    k = np.array([1, 2, 3, 7, 4, 2, 5, 3, 6, 2, 4, 5], dtype=np.int64)
    c = np.r_[0, np.cumsum(k)]
    Z = np.array([rng.integers(0, K, 2 * N) for K in k], dtype=np.uint8)
    D = np.concatenate([np.eye(K)[z.reshape(N, 2)].sum(axis=1).T for K, z in zip(k, Z)])
    p = D.mean(axis=1) / 2
    a = struct.scale(p)
    return Z, c, p, D, (D - 2 * p[:, None]) * a[:, None]


### Include an empty window, missing sample, and the largest cluster index
def missingFixture():
    rng = np.random.default_rng(116)
    N, k = 19, np.array([0, 2, 4, 255, 3, 2], np.int64)
    c = np.r_[0, np.cumsum(k)]
    Z = np.array([rng.integers(0, C, 2 * N) if C else np.full(2 * N, 255) for C in k], np.uint8)
    Z[:, :2] = 255
    Z[2] = 255
    Z[3, 7] = Z[4, 2] = 255
    parts, means = [], []
    for C, z in zip(k, Z):
        seen = z[z != 255]
        p = np.bincount(seen, minlength=C) / len(seen) if len(seen) else np.zeros(C)
        D = np.zeros((C, N))
        for h, label in enumerate(z):
            if label == 255:
                D[:, h // 2] += p
            else:
                D[int(label), h // 2] += 1
        parts.append(D)
        means.append(p)
    p, D = np.concatenate(means), np.concatenate(parts)
    obs = (Z != 255).sum(axis=1, dtype=np.int64)
    return Z, c, p, D, obs
