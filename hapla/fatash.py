"""Bounded local ancestry inference with an exact linear-time haploid HMM."""

from contextlib import ExitStack
from math import isfinite
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


### Reject invalid combinations before reading data or opening outputs
def checkArgs(args):
    if (args.clusters is None) == (args.filelist is None):
        raise ValueError("Provide exactly one of --clusters or --filelist")
    if (args.pfile is None) == (args.pfilelist is None) or args.qfile is None:
        raise ValueError("Provide --qfile and exactly one of --pfile or --pfilelist")
    if args.threads < 1 or args.block < 1 or args.buffer_mb < 1:
        raise ValueError("Threads, block size, and buffer size must be positive")
    if args.phase_correct is not None and args.phase_correct < 0:
        raise ValueError("Phase correction distance must be nonnegative")
    if args.alpha is not None and (not isfinite(args.alpha) or args.alpha <= 0):
        raise ValueError("Alpha must be finite and positive")
    if args.alpha_min >= args.alpha_max or not 1 <= args.alpha_num <= 100000:
        raise ValueError("Use increasing alpha exponents and 1..100000 alpha values")
    if args.viterbi and args.save_posteriors:
        raise ValueError("--save-posteriors requires posterior decoding")
    if args.iter < 1 or not all(
        isfinite(v) and v >= 0 for v in (args.tole, args.p_prior, args.q_prior)
    ):
        raise ValueError("Use positive HMM iterations and finite nonnegative tolerance and priors")
    if not args.fixed_model and (args.medians or args.block != 1 or args.simple):
        raise ValueError(
            "HMM fitting requires hard cluster emissions, --block 1, and standard transitions. Use --fixed-model for --medians, --block, or --simple"
        )
    for value in (args.min_length, args.max_length):
        if value is not None and value < 1:
            raise ValueError("Window length bounds must be positive")
    if (
        args.min_length is not None
        and args.max_length is not None
        and args.min_length > args.max_length
    ):
        raise ValueError("Minimum window length exceeds maximum length")
    if args.quantile is not None and (not isfinite(args.quantile) or not 0 < args.quantile < 1):
        raise ValueError("Window length quantile must lie strictly between zero and one")
    if args.quantile is not None and (args.min_length is not None or args.max_length is not None):
        raise ValueError("Choose either a quantile or explicit window length bounds")
    if not args.prefix or any(x in args.prefix for x in ("/", "\\")):
        raise ValueError("Output chromosome prefix must be a filename component")


### Keep HMM chains and smoothing blocks inside chromosomes
def readWindows(pfx, counts, args):
    import numpy as np

    from hapla.formats import readWindows as readRows

    rows = readRows(pfx)
    if len(rows) != len(counts):
        raise ValueError("Window metadata and cluster counts differ")
    regions, lengths, seen = [], [], set()
    prev, start = None, 0
    for w, row in enumerate(rows):
        chrom, beg, end, length, _, count = row
        if count != counts[w]:
            raise ValueError("Window cluster counts differ")
        if prev is None or chrom != prev[0]:
            if chrom in seen:
                raise ValueError("Chromosome windows must be contiguous")
            if prev is not None:
                regions.append((start, w))
            start = w
            seen.add(chrom)
        elif beg < prev[1]:
            raise ValueError("Windows must be ordered within each chromosome")
        prev = chrom, beg
        lengths.append(length)
    regions.append((start, len(rows)))
    lengths = np.asarray(lengths, np.int64)
    use = np.ones(len(rows), bool)
    if args.quantile is not None:
        lo, hi = np.quantile(lengths, [(1 - args.quantile) / 2, (1 + args.quantile) / 2])
        use &= (lengths >= lo) & (lengths <= hi)
    else:
        if args.min_length is not None:
            use &= lengths >= args.min_length
        if args.max_length is not None:
            use &= lengths <= args.max_length
    return regions, use.astype(np.uint8)


### Read and normalize one ancestry simplex per sample
def readQ(pth, N):
    import numpy as np

    Q = np.loadtxt(pth, ndmin=2)
    if Q.shape[0] != N or not 1 <= Q.shape[1] <= 255 or not np.all(np.isfinite(Q)) or np.any(Q < 0):
        raise ValueError(
            "Q must contain one row per sample and 1..255 finite nonnegative ancestry proportions"
        )
    sums = Q.sum(axis=1)
    if not np.allclose(sums, 1, rtol=0, atol=1e-5):
        raise ValueError("Q rows must sum to one")
    return Q / sums[:, None]


### Read and normalize cluster frequencies within each window and ancestry
def readP(pth, c, K):
    import numpy as np

    if c[-1] == 0:
        if Path(pth).read_text().strip():
            raise ValueError("A file with no cluster alleles requires an empty P file")
        return np.empty(0)
    P = np.loadtxt(pth, ndmin=2)
    if P.shape != (c[-1], K) or not np.all(np.isfinite(P)) or np.any(P < 0):
        raise ValueError("P dimensions or finite nonnegative cluster probabilities do not match")
    from hapla import fatash_cy as cy

    cy.normalizeP(P, c)
    return P.ravel()


### Budget emissions, posterior/traceback scratch, and paired output rows
def batchSize(N, W, regions, K, A, args):
    B = max((end - beg + args.block - 1) // args.block for beg, end in regions)
    viterbi = args.viterbi and args.fixed_model
    row = W * (2 + 8 * args.save_posteriors) + B * K * (8 if viterbi else 16) + 8 * (3 * K + A)
    # Include argmax temporaries and expansion of blocked paths/confidence.
    row += B * (1 if viterbi else 9)
    if args.block > 1:
        row += W * (1 + 8 * args.save_posteriors) * 2
    scratch = B * K * ((5 if A > 1 else 1) if viterbi else 16) + B * 8 + K * 40
    budget = args.buffer_mb * 1024**2
    nt = min(N, args.threads)
    n = (budget - nt * scratch) // row
    if n < nt:
        n = budget // (row + scratch)
    n = min(N, n) // 2 * 2
    if n < 2:
        raise ValueError("The HMM workspace cannot hold two haplotypes. Increase --buffer-mb")
    step = nt if nt % 2 == 0 else 2 * nt
    if n < N and n >= step:
        n = n // step * step
    return n


### Transpose bounded batches while keeping each sample's haplotypes together
def haplotypes(Z, size):
    import numpy as np

    for beg in range(0, Z.shape[1], size):
        end = min(Z.shape[1], beg + size)
        yield beg, end, np.ascontiguousarray(Z[:, beg:end].T)


### Fit window frequencies and one individual Q across all chromosome chains
def refine(data, P, Q, alpha, args):
    import numpy as np

    from hapla import fatash_cy as cy

    tick = perf_counter()
    K = Q.shape[1]
    tp, tq = (0.0, 0.0) if args.baum_welch else (args.p_prior, args.q_prior)
    baseP, baseQ = P, Q
    obs = np.zeros(len(Q), np.int64)
    for Z, _, use, _, _ in data:
        obs += cy.observations(Z, use)
    n = int(obs.sum())
    history, prev = [], None
    lap = tick
    label = "Log-like" if args.baum_welch else "Objective"
    stop = "no_observations" if n == 0 else "iteration_limit"
    for it in range(args.iter + 1 if n else 0):
        finish = it == args.iter
        nextP = []
        countQ = None if finish else np.zeros_like(Q)
        ll, prior = 0.0, cy.penalty(baseQ.ravel(), Q.ravel(), tq)
        for (Z, c, use, regions, size), p, base in zip(data, P, baseP):
            prior += cy.penalty(base, p, tp)
            table = cy.emissionTable(p, c, K)
            countP = None if finish else np.zeros_like(p)
            for i, j, z in haplotypes(Z, size):
                q = np.repeat(Q[i // 2 : j // 2], 2, axis=0)
                for beg, end in regions:
                    E = cy.emissions(z, table, c, use, K, beg, end)
                    G, C, L = cy.posterior(E, q, alpha, resets=not finish, score=finish)
                    ll += float(L.mean(axis=1).sum())
                    if not finish:
                        cy.accumulate(z, G, use, c, beg, end, countP)
                        countQ[i // 2 : j // 2] += C.reshape(-1, 2, K).sum(axis=1)
                    del E, G, C, L
            if not finish:
                nextP.append(cy.refineP(base, countP, c, K, tp))
        obj = ll + prior
        gain = obj - history[-1]["objective"] if history else None
        if not isfinite(obj) or (gain is not None and gain < -1e-10 * max(1.0, abs(obj))):
            if prev is None:
                raise ValueError("Initial HMM objective is not finite")
            P, Q = prev
            stop = "objective_decreased"
            break
        history.append(
            dict(
                iteration=it,
                log_likelihood=ll,
                objective=obj,
                improvement=None if gain is None else gain / n,
            )
        )
        if it == 0:
            print(f"Initial {label.lower()}: {obj:,.1f}", flush=True)
            lap = perf_counter()
        elif it % 5 == 0:
            now = perf_counter()
            printTiming(f"({it:,})  {label}: {obj:,.1f}", now - lap)
            lap = now
        if gain is not None and gain / n <= args.tole:
            stop = "converged"
            break
        if finish:
            break
        prev = P, Q
        P = nextP
        countQ += tq * baseQ
        total = countQ.sum(axis=1, keepdims=True)
        Q = baseQ.copy()
        np.divide(countQ, total, out=Q, where=total > 0)
        Q[obs == 0] = baseQ[obs == 0]
    # Report a final partial block using only accepted updates, including after rollback.
    it = max(0, len(history) - 1)
    if it % 5:
        printTiming(f"({it:,})  {label}: {history[-1]['objective']:,.1f}", perf_counter() - lap)
    info = dict(
        mode="baum-welch" if args.baum_welch else "regularized",
        p_prior=tp,
        q_prior=tq,
        iterations=it,
        stop=stop,
        observed_assignments=n,
        unobserved_samples=int(np.count_nonzero(obs == 0)),
        history=history,
        seconds=perf_counter() - tick,
    )
    status = {
        "converged": "Converged.",
        "iteration_limit": "Iteration limit reached.",
        "no_observations": "No observed assignments. Retained input P/Q.",
        "objective_decreased": "Objective decreased. Restored the last accepted P/Q.",
    }
    print(status[stop], flush=True)
    return P, Q, info


### Decode batches and write haplotypes in sample order without a full posterior cube
def decode(Z, table, Q, c, use, regions, size, alpha, args, path, prob):
    import numpy as np

    from hapla import fatash_cy as cy

    K, n_fix, ll = Q.shape[1], 0, 0.0
    with ExitStack() as stack:
        dst = stack.enter_context(Path(path).open("w", buffering=1024**2))
        conf = stack.enter_context(Path(prob).open("w", buffering=1024**2)) if prob else None
        for i, j, z in haplotypes(Z, size):
            q = np.repeat(Q[i // 2 : j // 2], 2, axis=0)
            D = np.empty_like(z)
            probs = np.empty(z.shape) if conf else None
            for beg, end in regions:
                E = cy.emissions(z, table, c, use, K, beg, end, args.block)
                if args.viterbi:
                    d = cy.viterbi(E, q, alpha, args.simple)
                    p = None
                else:
                    G, _, L = cy.posterior(E, q, alpha, args.simple)
                    d = G.argmax(axis=2).astype(np.uint8)
                    p = G.max(axis=2) if conf else None
                    ll += float(L.mean(axis=1).sum())
                    del G, L
                del E
                if args.block > 1:
                    repeat = np.full(d.shape[1], args.block, dtype=np.intp)
                    repeat[-1] = end - beg - (d.shape[1] - 1) * args.block
                    d = np.repeat(d, repeat, axis=1)
                    if p is not None:
                        p = np.repeat(p, repeat, axis=1)
                    del repeat
                if args.phase_correct is not None:
                    d = np.ascontiguousarray(d)
                    if p is not None:
                        p = np.ascontiguousarray(p)
                    n_fix += cy.phaseCorrect(
                        d, np.empty((0, 0)) if p is None else p, args.phase_correct, p is not None
                    )
                D[:, beg:end] = d
                if conf:
                    probs[:, beg:end] = p
                del d, p
            cy.writeRows(dst.fileno(), D=D)
            if conf:
                cy.writeRows(conf.fileno(), P=probs)
            del D, probs, z
    return dict(phase_corrections=n_fix, mean_alpha_log_likelihood=None if args.viterbi else ll)


### Validate inputs, estimate priors, and publish the complete result set atomically
def main(args):
    checkArgs(args)
    configureThreads(args.threads, blas=1)
    import numpy as np

    from hapla import fatash_cy as cy
    from hapla.formats import MAGIC, readHeader, readMetadata, readPaths
    from hapla.struct import readData

    tick = perf_counter()
    printHeader("fatash", args.threads)
    print("Reading clusters and P/Q estimates.", flush=True)
    paths, ids, counts, sizes = readMetadata(args.clusters, args.filelist, likes=args.medians)
    pfiles = readPaths(args.pfilelist, args.pfile)
    if len(pfiles) != len(paths):
        raise ValueError("Assignment and P file counts differ")
    Q = readQ(args.qfile, len(ids))
    K, N = Q.shape[1], 2 * len(Q)
    with np.errstate(over="ignore", under="ignore"):
        alpha = (
            np.array([args.alpha])
            if args.alpha is not None
            else np.logspace(-args.alpha_max, -args.alpha_min, args.alpha_num)
        )
    if not np.all(np.isfinite(alpha)) or np.any(alpha <= 0):
        raise ValueError("Alpha exponents produce nonfinite or zero rates")
    stems = [f".{args.prefix}{i + 1}" if len(paths) > 1 else "" for i in range(len(paths))]
    sfxs = [".log", ".Q", ".ids"] + [f"{s}{x}" for s in stems for x in (".path", ".P")]
    if len(paths) > 1:
        sfxs.append(".pfilelist")
    stale = [f"{s}.prob" for s in stems]
    if args.save_posteriors:
        sfxs += stale
    if len(paths) == 1:
        stale.append(".pfilelist")
    inputs = [args.filelist, args.pfilelist, args.qfile, *pfiles] + [
        f"{p}{s}"
        for p in paths
        for s in ((".bca", ".ids", ".win", ".blk") if args.medians else (".bca", ".ids", ".win"))
    ]
    stats = dict(
        samples=len(ids),
        ancestries=K,
        threads=args.threads,
        alpha=alpha.tolist(),
        ensemble="Viterbi plurality" if args.viterbi else "mean posterior",
        files=[],
    )
    print(f"Data size: {len(ids):,} samples, {len(counts):,} windows, K={K}", flush=True)
    offset, data, P = 0, [], []
    with ExitStack() as stack:
        out = stageOutputs(stack, args.out, sfxs, inputs, stale=stale)
        for pfx, pfile, W in zip(paths, pfiles, sizes):
            W = int(W)
            k = counts[offset : offset + W]
            offset += W
            mapped, _ = readData([pfx], k, [W], len(ids), freq=False)
            Z, c, obs = mapped[0]
            regions, use = readWindows(pfx, k, args)
            size = batchSize(N, W, regions, K, len(alpha), args)
            data.append((Z, c, use, regions, size))
            P.append(readP(pfile, c, K))
            stats["files"].append(
                dict(
                    input=str(pfx),
                    windows=W,
                    chains=len(regions),
                    batch_haplotypes=size,
                    excluded_windows=int(W - use.sum()),
                    missing_assignments=int((N - obs).sum()) if obs is not None else 0,
                )
            )
        printMissing(sum(row["missing_assignments"] for row in stats["files"]), N * len(counts))
        if args.fixed_model:
            stats["fit"] = dict(mode="fixed", iterations=0, stop="fixed_model")
            print("\nFixed model.", flush=True)
        else:
            mode = "Baum-Welch" if args.baum_welch else "Regularized Baum-Welch"
            print(f"\n{mode}:", flush=True)
            P, Q, stats["fit"] = refine(data, P, Q, alpha, args)
        np.savetxt(out[".Q"], Q, fmt="%.10g")
        np.savetxt(out[".ids"], ids, fmt="%s")
        if len(paths) > 1:
            out[".pfilelist"].write_text(
                "".join(f"{Path(args.out).absolute()}{s}.P\n" for s in stems)
            )
        mode = "Viterbi" if args.viterbi else "Mean posterior"
        print(f"\n{mode} decoding:", flush=True)
        for index, (pfx, stem, (Z, c, use, regions, size), p) in enumerate(
            zip(paths, stems, data, P)
        ):
            start = perf_counter()
            likes = None
            if args.medians:
                pth = Path(f"{pfx}.blk")
                with pth.open("rb") as src:
                    readHeader(src)
                length = int(np.sum(np.diff(c) ** 2))
                if pth.stat().st_size != len(MAGIC) + 4 * length:
                    raise ValueError("Median likelihood payload does not match the windows")
                likes = (
                    np.memmap(pth, mode="r", dtype=np.float32, offset=len(MAGIC), shape=(length,))
                    if length
                    else np.empty(0, np.float32)
                )
            table = cy.emissionTable(p, c, K, likes)
            np.savetxt(out[f"{stem}.P"], p.reshape(-1, K), fmt="%.10g")
            stats["files"][index].update(
                decode(
                    Z,
                    table,
                    Q,
                    c,
                    use,
                    regions,
                    size,
                    alpha,
                    args,
                    out[f"{stem}.path"],
                    out.get(f"{stem}.prob"),
                )
            )
            stats["files"][index]["seconds"] = perf_counter() - start
            label = f"File {index + 1}/{len(paths)}" if len(paths) > 1 else "Decoding complete."
            printTiming(label, stats["files"][index]["seconds"])
            del table, likes
        stats["elapsed_seconds"] = perf_counter() - tick
        writeLog(out[".log"], "fatash", args, stats)
        commitOutputs(args.out, out, stale=stale)
    printDone(args.out, out, stats["elapsed_seconds"])
    return stats
