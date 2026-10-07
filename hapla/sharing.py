"""Population structure from matching haplotype cluster sequences."""

__author__ = "Jonas Meisner"

from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from pathlib import Path
from resource import RLIMIT_NOFILE, getrlimit
from time import perf_counter

import numpy as np

from hapla import sharing_cy as cy
from hapla.formats import readWindows
from hapla.runtime import printTiming

##### Inputs


### Join chromosome runs across input files
def chromosomes(paths, data, p):
    parts, seen, off = [], set(), 0
    chrom, rows, zs, live = None, [], [], []
    for path, (Z, c, _) in zip(paths, data):
        meta = readWindows(path)
        cnt = np.r_[0, np.cumsum(p[off : off + c[-1]] > 0, dtype=np.int64)]
        info = np.diff(cnt[c]) > 1
        off += c[-1]
        beg = 0
        for end in range(1, len(meta) + 1):
            if end < len(meta) and meta[end][0] == meta[beg][0]:
                continue
            cur = meta[beg][0]
            if cur != chrom:
                if chrom is not None:
                    parts.append((chrom, rows, zs, np.concatenate(live)))
                if cur in seen:
                    raise ValueError("Sharing chromosomes must be contiguous in input order")
                seen.add(cur)
                chrom, rows, zs, live = cur, [], [], []
            rows.extend(meta[beg:end])
            zs.append(Z[beg:end])
            live.append(info[beg:end])
            beg = end
    parts.append((chrom, rows, zs, np.concatenate(live)))
    return parts


### Bound genomic cells at gaps and constant windows
def geometry(rows, info, gap, gmap=None):
    bp = np.array([(r[1], r[2]) for r in rows], dtype=np.float64)
    if not np.isfinite(bp).all():
        raise ValueError("Sharing window coordinates must be finite")
    if np.any(bp[1:, 0] <= bp[:-1, 1]):
        raise ValueError("Sharing requires ordered, nonoverlapping windows")
    cut = np.r_[True, (bp[1:, 0] - bp[:-1, 1] > gap) | ~info[:-1] | ~info[1:]]
    if gmap is None:
        g = bp / 1e6
        x = g.mean(axis=1)
    else:
        from hapla.maps import coordinates

        pos = coordinates(rows, gmap)
        g, x = pos[:, [0, 2]], pos[:, 1].copy()
    mid = 0.5 * (x[:-1] + x[1:])
    left, right = g[:, 0].copy(), g[:, 1].copy()
    left[1:] = np.where(cut[1:], left[1:], mid)
    right[:-1] = np.where(cut[1:], right[:-1], mid)
    return x, left, right, cut.astype(np.uint8)


##### Sharing profiles


### Map scratch arrays, including empty buffers
def cache(path, dtype, shape):
    if not np.prod(shape, dtype=np.int64):
        Path(path).touch()
        return np.empty(shape, dtype=dtype)
    return np.memmap(path, mode="w+", dtype=dtype, shape=shape)


### Cache sparse chromosome profiles
def paint(path, count, x, left, right, bins=1):
    H = len(count)
    N = H // 2
    ptr = np.r_[0, np.cumsum(count, dtype=np.int64)]
    rows, cov = np.zeros(N + 1, np.int64), np.empty(N)
    sums, seen, hit = np.empty(N), np.full(N, -1, np.int64), np.empty(N, np.int64)
    chunk = (N + bins - 1) // bins
    edge = np.minimum(np.arange(bins + 1) * chunk, N)
    seg = cache(path / "segments", np.uint32, (int(np.diff(ptr[2 * edge]).max()), 3))
    with (path / "columns").open("wb") as cols, (path / "values").open("wb") as vals:
        for slot in range(bins):
            first, last = slot * chunk, min((slot + 1) * chunk, N)
            if first >= last:
                break
            off = ptr[2 * first]
            size = ptr[2 * last] - off
            pth = path / ("matches" if bins == 1 else f"matches.{slot}")
            pos = ptr[:-1] - off
            with pth.open("rb") as src:
                for beg in range(0, size, 1048576):
                    rec = np.fromfile(src, np.uint32, count=4 * min(1048576, size - beg))
                    cy.group(rec.reshape(-1, 4), pos, seg)
            pth.unlink()
            for beg in range(first, last, 128):
                end = min(beg + 128, last)
                b, e = ptr[2 * beg] - off, ptr[2 * end] - off
                if b == e:
                    cov[beg:end] = 0
                    rows[beg + 1 : end + 1] = rows[beg]
                    continue
                sub = ptr[2 * beg : 2 * end + 1] - ptr[2 * beg]
                val = np.empty(e - b)
                cov[beg:end] = cy.paint(seg[b:e], sub, x, left, right, val)
                row = np.empty(end - beg + 1, np.int64)
                col, out = np.empty(e - b, np.uint32), np.empty(e - b)
                n = cy.compact(seg[b:e], sub, val, row, col, out, sums, seen, hit, beg)
                col[:n].tofile(cols)
                out[:n].tofile(vals)
                rows[beg + 1 : end + 1] = rows[beg] + row[1:]
                del val, col, out
    del seg
    (path / "segments").unlink()
    n = int(rows[-1])
    if n == 0:
        return None, cov
    col = np.memmap(path / "columns", mode="r", dtype=np.uint32, shape=(n,))
    val = np.memmap(path / "values", mode="r", dtype=np.float64, shape=(n,))
    return (rows, col, val), cov


### Combine chromosomes and cache normalized profiles in bounded batches
def profiles(data, cov, path, *, root=False, transpose=True):
    path.mkdir()
    N = len(cov)
    chunk = max(1, min(128, 1048576 // N))
    n = min(chunk * N, sum(len(d[1]) for d in data))
    row, rows = np.empty(chunk + 1, np.int64), np.zeros(N + 1, np.int64)
    col, val = np.empty(n, np.uint32), np.empty(n)
    sums, hit = np.zeros(N), np.empty(N, np.uint32)
    with (path / "columns").open("wb") as cols, (path / "values").open("wb") as vals:
        for beg in range(0, N, chunk):
            end = min(beg + chunk, N)
            n = cy.combine(data, cov, beg, row[: end - beg + 1], col, val, sums, hit, root)
            col[:n].tofile(cols)
            val[:n].tofile(vals)
            rows[beg + 1 : end + 1] = rows[beg] + row[1 : end - beg + 1]
    n = int(rows[-1])
    col = np.memmap(path / "columns", mode="r", dtype=np.uint32, shape=(n,))
    val = np.memmap(path / "values", mode="r", dtype=np.float64, shape=(n,))
    if not transpose:
        return ((rows, col, val),)
    back = np.empty_like(rows)
    idx = cache(path / "transpose-columns", np.uint32, (n,))
    out = cache(path / "transpose-values", np.float64, (n,))
    cy.transpose(rows, col, val, back, idx, out)
    return ((rows, col, val), (back, idx, out))


### Match chromosomes in parallel, then paint bounded batches
def build(
    paths,
    data,
    p,
    ids,
    tmp,
    matches=16,
    gap=1000000,
    threads=1,
    seed=42,
    *,
    root=False,
    transpose=True,
    gmap=None,
):
    if matches < 1 or gap < 1:
        raise ValueError("Sharing match count and gap must be positive")
    H = 2 * len(ids)
    if H < 4:
        raise ValueError("Sharing requires at least two samples")
    bins = min(32, (len(ids) + 127) // 128)
    order = sorted(range(len(ids)), key=lambda i: sha256(f"{seed}:{ids[i]}".encode()).digest())
    inv = np.array([(2 * i + h) for i in order for h in (0, 1)], dtype=np.int64)
    rank = np.empty(H, np.int64)
    rank[inv] = np.arange(H)
    jobs, span = [], 0.0
    tick = perf_counter()
    for n, (_, rows, parts, info) in enumerate(chromosomes(paths, data, p)):
        path = Path(tmp) / str(n)
        path.mkdir()
        x, left, right, cut = geometry(rows, info, gap, gmap)
        length = 2 * np.sum((right - left)[info])
        if length <= 0:
            continue
        Z = parts[0]
        if len(parts) > 1:
            Z = cache(path / "labels", np.uint8, (len(rows), H))
            off = 0
            for part in parts:
                Z[off : off + len(part)] = part
                off += len(part)
        if not cy.variable(Z):
            continue
        span += length
        jobs.append((path, Z, info.astype(np.uint8), cut, x, left, right))

    if span <= 0:
        raise ValueError("No usable haplotype sharing: no variable windows with positive length")

    def match(job):
        path, Z, info, cut, *_ = job
        return cy.matches(Z, cut, info, rank, min(matches, H - 2), bytes(path / "matches"), bins)

    limit = getrlimit(RLIMIT_NOFILE)[0]
    nt = min(threads, len(jobs))
    if limit > 0:
        nt = min(nt, max(1, (limit - 64) // bins))
    with ThreadPoolExecutor(max_workers=nt) as pool:
        runs = list(pool.map(match, jobs))
    info = dict(
        chromosomes=len(jobs),
        index_threads=nt,
        matches=sum(r[1] for r in runs),
        capped_matches=sum(r[2] for r in runs),
        index_seconds=perf_counter() - tick,
        distance_unit="Morgans" if gmap is not None else "Mb (physical distance proxy)",
        available_length=float(span),
    )
    printTiming("Matching complete.", info["index_seconds"])
    tick = perf_counter()
    data, cov = [], np.zeros(len(ids))
    for job, (count, total, _) in zip(jobs, runs):
        path, _, _, _, x, left, right = job
        if not total:
            continue
        cur, v = paint(path, count, x, left, right, bins)
        if cur is not None:
            data.append(cur)
        cov += v
    if not np.all(cov > 0):
        absent = ids[cov <= 0]
        raise ValueError(
            f"No usable haplotype sharing for {len(absent)} samples: "
            + ", ".join(map(str, absent[:5]))
        )
    data = profiles(data, cov, Path(tmp) / "profiles", root=root, transpose=transpose)
    info.update(
        links=len(data[0][1]),
        cache_bytes=sum(a.nbytes for side in data for a in side),
        coverage_min=float(np.min(cov / span)),
        coverage_mean=float(np.mean(cov / span)),
        painting_seconds=perf_counter() - tick,
    )
    printTiming("Sharing profiles complete.", info["painting_seconds"])
    return data, cov, info


##### Kernel and components


### Measure column means and total centered variation
def moments(data):
    mean, ss = cy.moments(*data[0])
    if not np.isfinite(ss) or ss <= 0:
        raise ValueError("Sharing requires positive variation in the centered profiles")
    return mean, ss


### Expand and center a bounded block of combined profiles
def rows(data, mean, beg, out):
    out.fill(0)
    cy.rows(*data[0], beg, out)
    out -= mean


### Stream the full lower triangle using bounded BLAS tiles
def grm(path, data, mean, ss, tile=None):
    N = len(data[0][0]) - 1
    if tile is None:
        tile = max(1, min(256, 8 * 1024**2 // (8 * N)))
    tile = min(N, tile)
    X, Y = np.empty((tile, N)), np.empty((tile, N))
    T = np.empty((tile, N), dtype="<f4")
    scale = (N - 1) / ss
    with Path(path).open("wb") as dst:
        for beg in range(0, N, tile):
            end = min(beg + tile, N)
            A = X[: end - beg]
            rows(data, mean, beg, A)
            for b in range(0, end, tile):
                e = min(b + tile, end)
                B = A if b == beg else Y[: e - b]
                if b != beg:
                    rows(data, mean, b, B)
                T[: len(A), b:e] = (A @ B.T) * scale
            for i in range(len(A)):
                T[i, : beg + i + 1].tofile(dst)


### Apply centered sharing or its transpose
def product(data, Q, transpose=False):
    Q = np.ascontiguousarray(Q, dtype=np.float64)
    if Q.ndim != 2 or len(Q) != len(data[0][0]) - 1:
        raise ValueError("Sharing sketch dimensions differ from the sample count")
    if transpose:
        Q = Q - Q.mean(axis=0)
    out = np.zeros_like(Q)
    cy.product(*data[int(transpose)], Q, out)
    if not transpose:
        out -= out.mean(axis=0)
    return out


### Fit randomized SVD and return unit-norm eigenvectors and singular values
def pca(data, K, power, seed):
    N = len(data[0][0]) - 1
    if not 1 <= K < N:
        raise ValueError("Sharing components must be between one and samples minus one")
    L = min(max(K + 10, 20), N - 1)
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(product(data, rng.standard_normal((N, L))), mode="reduced")
    shift = 0.0
    for _ in range(power):
        A = product(data, Q, True)
        Q, R = np.linalg.qr(product(data, A) - shift * Q, mode="reduced")
        low = np.linalg.svd(R, compute_uv=False)[-1]
        if low > shift:
            shift = 0.5 * (low + shift)
    A = product(data, Q, True)
    _, S, R = np.linalg.svd(A, full_matrices=False)
    rank = int(np.count_nonzero(S > np.finfo(float).eps * N * S[0]))
    if K > rank:
        raise ValueError(f"Requested {K} sharing PCs but the profiles support only {rank}")
    V = Q @ R[:K].T
    V *= np.sign(V[np.argmax(np.abs(V), axis=0), np.arange(K)])
    return V, S[:K]
