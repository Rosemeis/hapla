"""Residual correlations from bounded dosage blocks and low-rank projections."""

__author__ = "Thomas Bøggild"

from contextlib import ExitStack
from time import perf_counter

from hapla.runtime import (
    commitOutputs,
    configureThreads,
    printDone,
    printHeader,
    stageOutputs,
    writeLog,
)


### Use the estimable Q subspace without squaring its condition number
def basis(Q):
    import numpy as np

    U, S, _ = np.linalg.svd(Q, full_matrices=False)
    return np.ascontiguousarray(U[:, S > np.finfo(float).eps * max(Q.shape) * S[0]])


### Accumulate empirical and expected covariance with no thread-local square matrices
def covariance(data, Q, chunk=1024):
    import numpy as np

    from hapla import eval_cy as cy
    from hapla.struct import blocks

    N = len(Q)
    chunk = max(255, min(chunk, 64 * 1024**2 // (24 * N)))
    C, E, tmp = np.zeros((N, N)), np.zeros((N, N)), np.empty((N, N))
    U = basis(Q)
    V, full = np.zeros(N), np.zeros(N)
    left, right, rows = [], [], 0
    last, prior, part = None, None, np.zeros(N)

    # E = diag(v) - H diag(v) - diag(v) H + H diag(v) H, H = U U'
    def expected(U, v):
        nonlocal rows
        if not np.any(v):
            return
        V[:] += v
        A = v[:, None] * U
        left.append((0.5 * U @ (U.T @ A) - A).T)
        right.append(U.T)
        rows += U.shape[1]
        if rows >= chunk:
            flush()

    def flush():
        nonlocal rows
        if rows:
            np.matmul(np.concatenate(left).T, np.concatenate(right), out=tmp)
            E[:] += tmp
            E[:] += tmp.T
            left.clear()
            right.clear()
            rows = 0

    for Z, c, _, obs in blocks(data, chunk):
        if not c[-1] or (obs is not None and not np.any(obs)):
            continue
        R = np.empty((c[-1], N))
        if obs is None:
            if U.shape[1] == N:
                continue  # The fitted subspace leaves no residual degrees of freedom.
            np.matmul(cy.project(Z, c, U), U.T, out=R)
            full += cy.residuals(R, Z, c)
        else:
            w = 0
            while w < len(Z):
                if obs[w] == 0:
                    R[c[w] : c[w + 1]] = 0
                    w += 1
                    continue
                end = w + 1
                if obs[w] == 2 * N:
                    while end < len(Z) and obs[end] == 2 * N:
                        end += 1
                    u, h = U, None
                else:
                    h = (Z[w, ::2] != 255).astype(float) + (Z[w, 1::2] != 255)

                    # Reuse one missingness pattern, including across dosage blocks.
                    if last is None or not np.array_equal(h, last):
                        if last is not None:
                            expected(prior, part)
                            part.fill(0)
                        last, prior = h, basis(h[:, None] * Q)
                    u = prior
                a, b = c[w], c[end]
                r = R[a:b]
                z, idx = Z[w:end], c[w : end + 1] - a
                w = end
                if a == b:
                    continue
                if u.shape[1] == (N if h is None else np.count_nonzero(h)):
                    r.fill(0)
                    continue
                np.matmul(cy.project(z, idx, u), u.T, out=r)
                v = cy.residuals(r, z, idx)
                if h is None:
                    full += v
                else:
                    part += v
        np.matmul(R.T, R, out=tmp)
        C += tmp
    if last is not None:
        expected(prior, part)
    expected(U, full)
    flush()
    E.flat[:: N + 1] += V
    return C, E


### Zero undefined correlation rows and discard roundoff at zero variance
def correlation(C):
    import numpy as np

    from hapla import eval_cy as cy

    d = np.diag(C).copy()
    tol = np.finfo(float).eps * len(C) * max(0.0, d.max())
    d[d <= tol] = 0
    cy.correlation(C, np.sqrt(d))
    return C


### Validate sample order and publish all residual diagnostics together
def main(args):
    if (args.filelist is None) == (args.clusters is None):
        raise ValueError("Provide exactly one of --clusters or --filelist")
    if args.threads < 1 or args.qfile is None:
        raise ValueError("Provide --qfile and a positive number of threads")
    configureThreads(args.threads)
    import numpy as np

    from hapla import eval_cy as cy
    from hapla.formats import checkQIds, readMetadata, sampleIndices
    from hapla.struct import readData

    start = perf_counter()
    printHeader("eval", args.threads)
    print("Reading clusters.", flush=True)
    paths, ids, k, sizes = readMetadata(args.clusters, args.filelist)
    data, _ = readData(paths, k, sizes, len(ids), freq=False)
    if args.keep is not None:
        idx = sampleIndices(ids, args.keep, ordered=True)
        h = np.ravel(np.column_stack((2 * idx, 2 * idx + 1)))
        step = max(1, 1024**2 // len(h))
        subset = []
        for Z, c, obs in data:
            Z = np.take(Z, h, axis=1)
            if obs is not None:
                for w in range(0, len(Z), step):
                    obs[w : w + step] = (Z[w : w + step] != 255).sum(axis=1)
            subset.append((Z, c, obs if obs is not None and np.any(obs != len(h)) else None))
        data = subset
        ids = ids[idx]
    q_ids = checkQIds(args.qfile, ids)
    Q = np.loadtxt(args.qfile, ndmin=2)
    if Q.shape[0] != len(ids) or Q.shape[1] < 2 or not np.all(np.isfinite(Q)) or np.any(Q < 0):
        raise ValueError(
            "Q requires one row per sample and at least two finite nonnegative ancestries"
        )
    if not np.allclose(Q.sum(axis=1), 1, rtol=0, atol=1e-5):
        raise ValueError("Q rows must sum to one")
    Q /= Q.sum(axis=1, keepdims=True)
    print(f"Data size: {len(ids):,} samples, {len(k):,} windows", flush=True)
    print("Computing residual correlations.", flush=True)
    inputs = [args.filelist, args.qfile, q_ids, args.keep]
    inputs += [f"{p}{s}" for p in paths for s in (".bca", ".win", ".ids")]
    sfxs = (".bhat", ".chat", ".corres", ".ids", ".log")
    with ExitStack() as stack:
        out = stageOutputs(stack, args.out, sfxs, inputs, stale=())
        C, E = covariance(data, Q)
        correlation(C)
        correlation(E)
        for sfx, A, B in ((".bhat", C, None), (".chat", E, None), (".corres", C, E)):
            with out[sfx].open("wb") as dst:
                cy.writeMatrix(dst.fileno(), A, B)
        np.savetxt(out[".ids"], ids, fmt="%s")
        stats = dict(
            samples=len(ids),
            windows=len(k),
            threads=args.threads,
            elapsed_seconds=perf_counter() - start,
        )
        writeLog(out[".log"], "eval", args, stats)
        commitOutputs(args.out, out, stale=())
    printDone(args.out, out, stats["elapsed_seconds"])
