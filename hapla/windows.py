"""Bounded variant buffers and chromosome-aware windows."""

__author__ = "Jonas Meisner"

from time import perf_counter

import numpy as np


### Allocate reusable arrays and keep reader state in one dictionary
def createBuffer(read, ids, chroms, limit, *, phase=False, sites=None, b_read=16 * 1024**2):
    H = 2 * len(ids)
    row = H + 13 + (H // 2 if phase else 0)
    n = int(limit) // row
    if n < 1:
        raise ValueError("Input buffer cannot hold one variant. Increase --buffer-mb")
    return dict(
        read=read,
        ids=ids,
        chroms=chroms,
        sites=sites,
        G=np.empty((n, H), np.uint8),
        pos=np.empty(n, np.int64),
        rid=np.empty(n, np.int32),
        miss=np.empty(n, np.uint8),
        phase=np.empty((n, H // 2), np.uint8) if phase else None,
        chunk=max(1, min(n, int(b_read) // row)),
        beg=0,
        end=0,
        idx=0,
        variants=0,
        time=0.0,
        eof=False,
    )


### Refill as needed and allow large windows to exceed the routine read block
def fillBuffer(buf, n):
    beg, end = buf["beg"], buf["end"]
    left = end - beg
    if n > len(buf["pos"]):
        raise ValueError("A window exceeds the input buffer. Increase --buffer-mb")
    if left >= n or buf["eof"]:
        return left
    keys = ("G", "pos", "rid", "miss", "phase")
    if beg:
        for key in keys:
            if buf[key] is not None:
                buf[key][:left] = buf[key][beg:end]
    stop = min(len(buf["pos"]), left + max(n - left, buf["chunk"]))
    start = perf_counter()
    m = buf["read"](*(None if buf[k] is None else buf[k][left:stop] for k in keys))
    buf["eof"] = m < stop - left
    if buf["sites"] is not None:
        buf["sites"](buf["eof"])
    buf["time"] += perf_counter() - start
    buf["beg"], buf["end"] = 0, left + m
    buf["variants"] += m
    return left + m


### Advance while retaining overlapping variants for the next window
def advanceBuffer(buf, n):
    if not 0 <= n <= buf["end"] - buf["beg"]:
        raise ValueError("Invalid buffer advance")
    buf["beg"] += n
    buf["idx"] += n


### Find the first chromosome boundary in sorted contig IDs
def chromosomeEnd(buf):
    rid = buf["rid"][buf["beg"] : buf["end"]]
    if not len(rid) or rid[-1] == rid[0]:
        return len(rid)
    return int(np.searchsorted(rid, rid[0], side="right"))


### Copy one job: ((index, chromosome, start, end, size), GT, missing, phase)
def takeWindow(buf, n):
    beg, end = buf["beg"], buf["beg"] + n
    meta = (
        buf["idx"],
        buf["chroms"][buf["rid"][beg]],
        int(buf["pos"][beg]),
        int(buf["pos"][end - 1]),
        n,
    )
    phase = None if buf["phase"] is None else np.any(buf["phase"][beg:end], axis=0)
    return meta, buf["G"][beg:end].copy(), bool(np.any(buf["miss"][beg:end])), phase


### Slide fixed SNP windows without crossing chromosome boundaries
def fixedWindows(buf, size, step=None, tail="include"):
    step = size if step is None else step
    covered = 0
    while fillBuffer(buf, size):
        n = chromosomeEnd(buf)
        if n >= size:
            yield takeWindow(buf, size)
            covered = buf["idx"] + size
            advanceBuffer(buf, step)
        else:
            if tail == "include" and buf["idx"] + n > covered:
                yield takeWindow(buf, n)
            advanceBuffer(buf, n)


### Extend each window through its physical endpoint
def physicalWindows(buf, length):
    while fillBuffer(buf, 1):
        limit = int(buf["pos"][buf["beg"]]) + length
        while True:
            end = chromosomeEnd(buf)
            pos = buf["pos"][buf["beg"] : buf["beg"] + end]
            n = int(np.searchsorted(pos, limit, side="right"))
            if n < end or end < buf["end"] - buf["beg"] or buf["eof"]:
                break
            fillBuffer(buf, buf["end"] - buf["beg"] + 1)
        yield takeWindow(buf, n)
        advanceBuffer(buf, n)


### Read strictly increasing, zero-based window starts
def readStarts(pth):
    idx = []
    with open(pth) as src:
        for line in src:
            val = line.split("#", 1)[0].strip()
            if not val:
                continue
            if len(val.split()) != 1:
                raise ValueError("--windows requires one zero-based start index per line")
            try:
                val = int(val)
            except ValueError as err:
                raise ValueError("Window start indices must be integers") from err
            if val < 0 or (idx and val <= idx[-1]):
                raise ValueError("Window start indices must be nonnegative and strictly increasing")
            idx.append(val)
    if not idx or idx[0] != 0:
        raise ValueError("Window start indices must begin at zero")
    return idx


### Apply explicit boundaries and extend the final window to EOF
def predefinedWindows(buf, idx):
    for i, beg in enumerate(idx):
        if not fillBuffer(buf, 1):
            if i == len(idx) - 1 and beg == buf["idx"]:
                return  # Optional EOF sentinel
            raise ValueError("Window start index exceeds the genotype record count")
        if beg != buf["idx"]:
            raise ValueError("Window start indices do not match the input")
        if i + 1 < len(idx):
            n = idx[i + 1] - beg
            if fillBuffer(buf, n) < n:
                raise ValueError("Window boundary exceeds the genotype record count")
        else:
            while not buf["eof"]:
                fillBuffer(buf, buf["end"] - buf["beg"] + 1)
            n = buf["end"] - buf["beg"]
        if chromosomeEnd(buf) < n:
            raise ValueError("A predefined window crosses a chromosome boundary")
        yield takeWindow(buf, n)
        advanceBuffer(buf, n)
