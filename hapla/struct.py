"""Population structure from direct products of haplotype cluster labels."""

__author__ = "Jonas Meisner"

import json
from contextlib import ExitStack
from pathlib import Path
from time import perf_counter

from hapla.runtime import (
    commitOutputs,
    configureThreads,
    printDone,
    printHeader,
    printMissing,
    printTiming,
    stageOutputs,
    writeLog,
)


### Give constant cluster alleles zero standardization weight
def scale(p):
    import numpy as np

    v = 2.0 * p * (1.0 - p)
    if not np.any(v > 0):
        raise ValueError("Population structure requires variable cluster alleles")
    a = np.zeros_like(p)
    np.divide(1.0, np.sqrt(np.maximum(v, 0)), out=a, where=v > 0)
    return a


### Map assignments and count frequencies without expanding dosages
def readData(paths, k, sizes, N, *, freq=True):
    import numpy as np

    from hapla import struct_cy as cy
    from hapla.formats import mapLabels

    if np.any((k < 0) | (k > 255)):
        raise ValueError("Population structure requires 0..255 clusters per window")
    if len(paths) != len(sizes) or sum(map(int, sizes)) != len(k):
        raise ValueError("Assignment file and window counts differ")
    data, beg, off = [], 0, 0
    p = np.empty(int(k.sum(dtype=np.int64))) if freq else None
    for pfx, W in zip(paths, sizes):
        W = int(W)
        Z = mapLabels(pfx, W, N)
        c = np.r_[0, np.cumsum(k[beg : beg + W], dtype=np.int64)]
        end = off + c[-1]
        obs = np.empty(W, dtype=np.int64)
        cy.frequencies(Z, c, p[off:end] if freq else None, obs)
        data.append((Z, c, obs if np.any(obs != 2 * N) else None))
        beg += W
        off = end
    return data, p


### Measure empirical dosage variation directly, including imputed haplotypes
def variation(data, p):
    import numpy as np

    from hapla import struct_cy as cy

    v, off = np.empty_like(p), 0
    for Z, c, _ in data:
        end = off + c[-1]
        cy.variation(Z, c, p[off:end], v[off:end])
        off = end
    if not np.any(v > 0):
        raise ValueError("Population structure requires positive empirical dosage variation")
    return v


### Keep whole windows within the target cluster block size
def blocks(data, chunk):
    import numpy as np

    off = 0
    for Z, c, obs in data:
        beg = 0
        while beg < len(Z):
            end = max(beg + 1, int(np.searchsorted(c, c[beg] + chunk, side="right")) - 1)
            yield (
                Z[beg:end],
                c[beg : end + 1] - c[beg],
                slice(off + c[beg], off + c[end]),
                None if obs is None else obs[beg:end],
            )
            beg = end
        off += c[-1]


### Apply X' to a random sketch or to X Q, keeping only one block in memory
def product(data, p, a, L, chunk, *, Q=None, rng=None):
    import numpy as np

    from hapla import struct_cy as cy

    H = np.zeros((data[0][0].shape[1] // 2, L), dtype=np.float64)
    shift = np.zeros(L)
    sums = None if Q is None else Q.sum(axis=0)
    for Z, c, s, obs in blocks(data, chunk):
        if Q is None:
            A = rng.standard_normal((int(c[-1]), L), dtype=np.float32).astype(np.float64)
        else:
            A = np.empty((int(c[-1]), L))
            cy.leftProduct(Z, c, p[s], a[s], Q, sums, A, obs)
        A *= a[s, None]
        shift += 2.0 * (p[s] @ A)
        cy.rightProduct(Z, c, A, H, p[s], obs)
    H -= shift
    return H


### Use QR for stable subspace iterations and solve only the small final problem
def pca(data, p, K, chunk, power, seed, v=None):
    import numpy as np

    from hapla import struct_cy as cy

    a = scale(p)
    v = variation(data, p) if v is None else v
    a[v == 0] = 0
    energy = float(np.dot(v, a * a))
    if energy <= 0:
        raise ValueError("PCA requires positive empirical dosage variation")
    N, M = data[0][0].shape[1] // 2, len(p)
    L = min(max(K + 10, 20), N - 1, int(np.count_nonzero(a)))
    if not 1 <= K <= L:
        raise ValueError(
            "PCA components must fit the centered sample and variable-cluster dimensions"
        )
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(product(data, p, a, L, chunk, rng=rng), mode="reduced")
    shift = 0.0
    for it in range(power):
        H = product(data, p, a, L, chunk, Q=np.ascontiguousarray(Q))
        H -= shift * Q
        Q, R = np.linalg.qr(H, mode="reduced")
        low = np.linalg.svd(R, compute_uv=False)[-1]
        if low > shift:
            shift = 0.5 * (low + shift)

    # Accumulate (X Q)' (X Q) without storing all cluster loadings
    Q = np.ascontiguousarray(Q)
    T, sums = np.zeros((L, L)), Q.sum(axis=0)
    for Z, c, s, obs in blocks(data, chunk):
        A = np.empty((int(c[-1]), L))
        cy.leftProduct(Z, c, p[s], a[s], Q, sums, A, obs)
        T += A.T @ A
    vals, R = np.linalg.eigh(T)
    vals, R = vals[::-1], R[:, ::-1]
    tol = np.finfo(float).eps * max(M, N) * energy
    rank = int(np.count_nonzero(vals > tol))
    if K > rank:
        raise ValueError(f"Requested {K} PCs but the data support only {rank} nonzero components")
    V = np.ascontiguousarray(Q @ R[:, :K])
    S = np.sqrt(vals[:K])
    signs = np.sign(V[np.argmax(np.abs(V), axis=0), np.arange(K)])
    V *= signs
    return V, S, a


### Project standardized labels using fixed reference loadings
def project(data, p, U, vals, chunk):
    import numpy as np

    from hapla import struct_cy as cy

    M, K = U.shape
    if M != len(p) or vals.shape != (K,) or K < 1:
        raise ValueError("Reference frequency, loading, and eigenvalue dimensions differ")
    if not np.all(np.isfinite(U)) or not np.all(np.isfinite(vals)) or np.any(vals <= 0):
        raise ValueError("Reference loadings and positive eigenvalues must be finite")
    if not np.all(np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError("Reference frequencies must lie between zero and one")
    a = scale(p)
    V = np.zeros((data[0][0].shape[1] // 2, K))
    shift = np.zeros(K)
    for Z, c, s, obs in blocks(data, chunk):
        A = np.ascontiguousarray(U[s] * (a[s, None] / np.sqrt(vals * M)))
        shift += 2.0 * (p[s] @ A)
        cy.rightProduct(Z, c, A, V, p[s], obs)
    V -= shift
    return V


### Accumulate the same centered GRM with one fewer row per window
def grm(data, p, chunk, center=True, tile=None, info=None):
    import numpy as np

    from hapla import struct_cy as cy

    N = data[0][0].shape[1] // 2
    den = float(np.sum(p * (1.0 - p)))
    if den <= 0:
        raise ValueError("GRM estimation requires variable cluster alleles")
    G = np.zeros(N * (N + 1) // 2)
    if tile is None:
        b_tile = 16 * 1024**2
        tile = max(1, b_tile // (N * np.dtype(np.float32).itemsize))
    tile = min(N, tile)
    T = np.empty(tile * N, dtype=np.float32)
    for Z, c, s, obs in blocks(data, chunk):
        rows = np.r_[0, np.cumsum(np.maximum(np.diff(c) - 1, 0))]
        M = int(rows[-1])
        if M == 0:
            continue
        X = np.empty((M, N), dtype=np.float32)
        cy.contrastBlock(Z, c, p[s], rows, X)
        for beg in range(0, N, tile):
            end = min(N, beg + tile)
            tmp = T[: (end - beg) * end].reshape(end - beg, end)
            np.dot(X[:, beg:end].T, X[:, :end], out=tmp)
            cy.addGram(tmp, G, beg)
    scale = cy.normalizeGram(G, den, np.zeros(N), center)
    if info is not None:
        info.update(
            frequency_sum=den,
            normalization_denominator=2 * den,
            gower_scale=scale,
            centered=bool(center),
        )
    return G, den


### Stream optional loadings from the final sample vectors
def writeLoadings(pth, data, p, a, V, S, chunk):
    import numpy as np

    from hapla import struct_cy as cy

    sums = V.sum(axis=0)
    with Path(pth).open("w", buffering=1024**2) as dst:
        for Z, c, s, obs in blocks(data, chunk):
            A = np.empty((int(c[-1]), V.shape[1]))
            cy.leftProduct(Z, c, p[s], a[s], V, sums, A, obs)
            np.savetxt(dst, A / S, fmt="%.10g")


### Write numeric PC columns directly, with optional sample identifiers
def writeVectors(pth, V, ids, raw, dup):
    import numpy as np

    if raw:
        np.savetxt(pth, V, fmt="%.10g")
        return
    with Path(pth).open("w", buffering=1024**2) as dst:
        dst.write("#FID\tIID\t" + "\t".join(f"PC{k + 1}" for k in range(V.shape[1])) + "\n")
        for name, row in zip(ids, V):
            dst.write(
                f"{name if dup else '0'}\t{name}\t" + "\t".join(f"{v:.10g}" for v in row) + "\n"
            )


### Run requested analyses with one mapped, validated input set
def main(args):
    if (args.clusters is None) == (args.filelist is None):
        raise ValueError("Provide exactly one of --clusters or --filelist")
    if not args.grm and args.pca is None and args.projection is None:
        raise ValueError("Select --grm, --pca, or --projection")
    if args.threads < 1 or args.chunk < 1 or args.power < 1 or args.seed < 0:
        raise ValueError(
            "Threads, chunk size, and power must be positive. Seed must be nonnegative"
        )
    if args.pca is not None and args.pca < 1:
        raise ValueError("PCA components must be positive")
    if args.loadings and args.pca is None:
        raise ValueError("--loadings requires --pca")
    if args.projection == "":
        raise ValueError("Projection prefix must not be empty")
    configureThreads(args.threads, blas=args.threads if args.grm else 1)
    import numpy as np

    from hapla.formats import readMetadata
    from hapla.identity import checkModel, featureKeys, writeModel

    start = perf_counter()
    printHeader("struct", args.threads)
    print("Reading clusters.", flush=True)
    paths, ids, k, sizes = readMetadata(args.clusters, args.filelist)
    M = int(k.sum(dtype=np.int64))
    data, p = readData(paths, k, sizes, len(ids), freq=args.grm or args.pca is not None)
    keys = featureKeys(paths, k, sizes) if args.loadings or args.projection else None
    if args.projection:
        checkModel(args.projection, keys)
    v = variation(data, p) if args.pca is not None or args.grm else None
    stats = dict(
        samples=len(ids),
        windows=len(k),
        clusters=M,
        threads=args.threads,
        missing_assignments=sum(int((2 * len(ids) - o).sum()) for _, _, o in data if o is not None),
        empty_windows=int(np.count_nonzero(k == 0)),
        read_seconds=perf_counter() - start,
    )
    print(f"Data size: {len(ids):,} samples, {len(k):,} windows, {M:,} clusters", flush=True)
    printMissing(stats["missing_assignments"], 2 * len(ids) * len(k))
    sfxs = [".log"]
    if args.grm:
        sfxs += [".grm.bin", ".grm.N.bin", ".grm.id", ".grm.meta.json"]
    stale = (".loadings", ".freqs", ".pca.json") if args.pca is not None else ()
    if args.pca is not None:
        sfxs += [".eigenvecs", ".eigenvals"]
        if args.loadings:
            sfxs += [".loadings", ".freqs", ".pca.json"]
    if args.projection is not None:
        sfxs += [".project.eigenvecs"]
    inputs = [f"{pth}{s}" for pth in paths for s in (".bca", ".win", ".ids")]
    inputs += [args.filelist, *[f"{pth}.ref.json" for pth in paths]]
    if args.projection:
        inputs += [
            f"{args.projection}{s}" for s in (".freqs", ".loadings", ".eigenvals", ".pca.json")
        ]
    with ExitStack() as stack:
        out = stageOutputs(stack, args.out, sfxs, inputs, stale=stale)
        if args.grm:
            tick = perf_counter()
            print("\nComputing GRM.", flush=True)

            # One represented categorical contrast per additional observed cluster allele.
            c = np.r_[0, np.cumsum(k, dtype=np.int64)]
            seen = np.r_[0, np.cumsum(p > 0, dtype=np.int64)]
            levels = np.diff(seen[c])
            count = int(np.maximum(levels - 1, 0).sum())
            info = dict(
                count_unit="categorical_contrasts",
                count=count,
                windows=len(k),
                polymorphic_windows=int(np.count_nonzero(levels > 1)),
                missingness="haplotype_mean_imputation",
                count_is_pairwise_observed=False,
            )
            G, den = grm(data, p, args.chunk, not args.no_centering, info=info)
            with out[".grm.bin"].open("wb") as dst, out[".grm.N.bin"].open("wb") as cnt:
                for beg in range(0, len(G), 262144):
                    n = min(262144, len(G) - beg)
                    dst.write(G[beg : beg + n].astype(np.float32))
                    cnt.write(np.full(n, count, dtype=np.float32))
            with out[".grm.id"].open("w") as dst:
                dst.writelines(f"{s if args.duplicate_fid else '0'}\t{s}\n" for s in ids)
            out[".grm.meta.json"].write_text(json.dumps(info, indent=2) + "\n")
            stats["grm"] = info
            stats["grm_seconds"] = perf_counter() - tick
            printTiming("GRM complete.", stats["grm_seconds"])
            del G
        if args.pca is not None:
            tick = perf_counter()
            print(f"\nComputing {args.pca} principal components.", flush=True)
            V, S, a = pca(data, p, args.pca, args.chunk, args.power, args.seed, v)
            writeVectors(out[".eigenvecs"], V, ids, args.raw, args.duplicate_fid)
            np.savetxt(out[".eigenvals"], S * S / len(p), fmt="%.10g")
            if args.loadings:
                writeLoadings(out[".loadings"], data, p, a, V, S, args.chunk)
                np.savetxt(out[".freqs"], p, fmt="%.10g")
                writeModel(out, keys)
            stats["pca_seconds"] = perf_counter() - tick
            printTiming("PCA complete.", stats["pca_seconds"])
            del V, S, a
        if args.projection is not None:
            tick = perf_counter()
            print("\nProjecting samples.", flush=True)
            vals = np.loadtxt(f"{args.projection}.eigenvals", ndmin=1)
            U = np.loadtxt(f"{args.projection}.loadings", ndmin=2)
            freq = np.loadtxt(f"{args.projection}.freqs", ndmin=1)
            if len(freq) != M:
                raise ValueError("Number of clusters does not match the reference")
            V = project(data, freq, U, vals, args.chunk)
            writeVectors(out[".project.eigenvecs"], V, ids, args.raw, args.duplicate_fid)
            stats["projection_seconds"] = perf_counter() - tick
            printTiming("Projection complete.", stats["projection_seconds"])
        stats["elapsed_seconds"] = perf_counter() - start
        writeLog(out[".log"], "struct", args, stats)
        commitOutputs(args.out, out, stale=stale)
    printDone(args.out, out, stats["elapsed_seconds"])
    return stats
