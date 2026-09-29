"""Shared genetic map validation and linear interpolation."""

__author__ = "Jonas Meisner"

import gzip
from array import array
from math import isfinite

import numpy as np


### Match common chromosome prefixes without guessing genome builds
def chromosome(chrom):
    return str(chrom).removeprefix("chr")


### Read cumulative cM, with an optional header and gzip compression
def readMap(path):
    data, cols = {}, (0, 1, 2)
    names = (
        {"chr", "chrom", "chromosome"},
        {"bp", "pos", "position", "position(bp)"},
        {"cm", "genetic", "genetic_map(cm)", "map(cm)"},
    )
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt") as src:
        for n, line in enumerate(src, 1):
            row = line.split()
            if not row:
                continue
            if not data:
                keys = [s.lower().lstrip("#") for s in row]
                idx = [[i for i, s in enumerate(keys) if s in group] for group in names]
                if all(len(i) == 1 for i in idx):
                    cols = tuple(i[0] for i in idx)
                    continue
            if row[0].startswith("#"):
                continue
            try:
                chrom = chromosome(row[cols[0]])
                bp, cm = float(row[cols[1]]), float(row[cols[2]])
                if (
                    not chrom
                    or not isfinite(bp)
                    or not isfinite(cm)
                    or bp < 1
                    or bp != int(bp)
                    or cm < 0
                ):
                    raise ValueError
            except (ValueError, IndexError):
                raise ValueError(
                    f"Invalid map line {n}. Use chromosome, position in bp, cumulative cM"
                ) from None
            if chrom not in data:
                data[chrom] = array("d")
            data[chrom].extend((bp, cm))
    if not data:
        raise ValueError("Genetic map is empty")
    for chrom, rows in data.items():
        a = np.asarray(rows).reshape(-1, 2)
        if len(a) < 2 or np.any(np.diff(a[:, 0]) <= 0) or np.any(np.diff(a[:, 1]) < 0):
            raise ValueError(
                f"Map chromosome {chrom} needs increasing positions and nondecreasing cM"
            )
        data[chrom] = a
    return data


### Interpolate window starts, physical midpoints, and ends in Morgans
def coordinates(rows, data):
    out = np.empty((len(rows), 3))
    beg = 0
    while beg < len(rows):
        chrom = chromosome(rows[beg][0])
        end = beg + 1
        while end < len(rows) and chromosome(rows[end][0]) == chrom:
            end += 1
        if chrom not in data:
            raise ValueError(f"Chromosome {chrom} is absent from the genetic map")
        a = data[chrom]
        bp = np.array([(r[1], (r[1] + r[2]) / 2, r[2]) for r in rows[beg:end]])
        if bp.min() < a[0, 0] or bp.max() > a[-1, 0]:
            raise ValueError(
                f"Window coordinates on chromosome {chrom} exceed genetic map coverage"
            )
        out[beg:end] = np.interp(bp, a[:, 0], a[:, 1]) / 100
        beg = end
    return out


### Reset the first transition of each chromosome chain
def distances(rows, regions, data):
    x = coordinates(rows, data)[:, 1]
    d = np.zeros(len(x))
    for beg, end in regions:
        d[beg + 1 : end] = np.diff(x[beg:end])
    if np.any(d < 0):
        raise ValueError("Window midpoints must be ordered within each chromosome")
    return d
