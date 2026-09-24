"""Ancestry estimation with categorical EM and bounded native updates."""

__author__ = "Jonas Meisner"

import os
from contextlib import ExitStack
from math import isfinite
from pathlib import Path
from time import perf_counter as time

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


### hapla admix
def main(args):
    snp_vcf = getattr(args, "snp_vcf", None)
    # Check input
    if args.supervised is not None and args.projection is not None:
        raise ValueError("Choose either --supervised or --projection")
    if args.projection is not None and args.p_prior:
        raise ValueError("--p-prior cannot update fixed projection frequencies")
    if args.loo and args.projection is not None:
        raise ValueError("--loo requires fitted P, not fixed projection frequencies")
    if snp_vcf is not None and args.loo:
        raise ValueError("Weighted SNP input is not supported with --loo")
    if (args.filelist is None) == (args.clusters is None):
        raise ValueError("Provide exactly one of --clusters or --filelist")
    if snp_vcf is not None and args.projection is not None:
        raise ValueError("Weighted SNP input is not supported in projection mode")
    if not args.prefix or any(x in args.prefix for x in ("/", "\\")):
        raise ValueError("Output chromosome prefix must be a filename component")
    if args.K is None or not 1 < args.K < 100000:
        raise ValueError("Please select 1 < K < 100000 (a 1e-5 probability floor is used)!")
    if args.keep is not None and not os.path.isfile(args.keep):
        raise ValueError("Keep file doesn't exist!")
    if args.threads < 1:
        raise ValueError("Please select a valid number of threads!")
    if args.seed < 0:
        raise ValueError("Please select a valid seed!")
    if args.iter < 1:
        raise ValueError("Please select a valid number of iterations!")
    if not isfinite(args.tole) or args.tole < 0:
        raise ValueError("Please select a valid tolerance!")
    if args.batches < 1:
        raise ValueError("Please select a valid number of mini-batches!")
    if args.check < 1:
        raise ValueError("Please select a valid value for convergence check!")
    if args.power < 1:
        raise ValueError("Please select a valid number of power iterations!")
    if args.chunk < 1:
        raise ValueError("Please select a valid SVD chunk size!")
    if args.als_iter < 1:
        raise ValueError("Please select a valid number of iterations in ALS!")
    if not isfinite(args.als_tole) or args.als_tole < 0:
        raise ValueError("Please select a valid tolerance in ALS!")
    if not isfinite(args.p_prior) or args.p_prior < 0:
        raise ValueError("Please select a finite, nonnegative P prior mass!")
    if args.subsampling < 2:
        raise ValueError("Please select a valid subsampling factor!")
    printHeader("admix", args)
    p_out = f"{args.out}.proj" if args.projection is not None else f"{args.out}"
    f_out = f"{p_out}.K{args.K}.s{args.seed}"
    start = time()

    # Configure numerical pools before importing kernels
    configureThreads(args.threads, blas=1)

    # Import numerical libraries and cython functions
    from math import ceil

    import numpy as np

    from hapla import admix_cy, functions, struct_cy
    from hapla.formats import mapLabels, readMetadata, readPaths, readWindows, sampleIndices

    # Read chromosome metadata once and concatenate cluster counts
    Z_list, z_ids, k_vec, w_vec = readMetadata(args.clusters, args.filelist)
    F = len(Z_list)
    N_all, W = len(z_ids), len(k_vec)
    f_vec = np.insert(np.cumsum(w_vec, dtype=np.uint32), 0, 0)

    M = int(np.sum(k_vec, dtype=np.uint64))
    if M == 0:
        raise ValueError("No observed cluster alleles for ancestry estimation")
    if M * args.K > np.iinfo(np.uint32).max:
        raise ValueError("Too many cluster parameters for the native index range")
    M_clusters = M

    # Select samples from keep file
    if args.keep is not None:
        keep = sampleIndices(z_ids, args.keep)
        hap_idx = np.empty(keep.shape[0] * 2, dtype=np.uint32)
        hap_idx[0::2] = 2 * keep
        hap_idx[1::2] = 2 * keep + 1
        N = keep.shape[0]
        print(f"Selected samples: {N:,}/{N_all:,}", flush=True)
    else:
        keep = None
        hap_idx = None
        N = N_all
    q_ids = z_ids if keep is None else z_ids[keep]
    print("Reading clusters.", flush=True)

    # Keep a single unfiltered chromosome mapped and concatenate only when needed
    B = 0
    Z = None if F == 1 and hap_idx is None else np.empty((W, 2 * N), dtype=np.uint8)
    for z in range(F):
        z_tmp = mapLabels(Z_list[z], w_vec[z], N_all)
        if hap_idx is not None:
            admix_cy.validateLabels(z_tmp, k_vec[B : B + w_vec[z]])
        if Z is None:
            Z = z_tmp
        elif hap_idx is None:
            Z[B : B + w_vec[z]] = z_tmp
        else:
            # Write validated indices directly without an intermediate chromosome copy.
            np.take(z_tmp, hap_idx, axis=1, out=Z[B : B + w_vec[z]], mode="clip")
        B += w_vec[z]
    del z_tmp

    # Give every cluster window one unit of total SNP weight.
    weights = None
    snp_count = 0
    if snp_vcf is not None:
        if F != 1:
            raise ValueError("Weighted SNP input currently requires one cluster prefix")
        from hapla.vcf_cy import Reader

        rows = readWindows(Z_list[0])
        intervals = {}
        for w, row in enumerate(rows):
            intervals.setdefault(row[0].removeprefix("chr"), []).append((row[1], row[2], w))
        intervals = {
            chrom: tuple(np.asarray(x) for x in zip(*values)) for chrom, values in intervals.items()
        }
        with Reader(snp_vcf, max(0, args.threads - 1), phased=True) as src:
            index = {sample: i for i, sample in enumerate(src.samples)}
            if any(sample not in index for sample in q_ids):
                raise ValueError("Cluster samples are missing from the SNP VCF/BCF")
            sel = np.asarray([index[sample] for sample in q_ids], dtype=np.int64)
            hsel = np.ravel(np.column_stack((2 * sel, 2 * sel + 1)))
            chunks, parent = [], []
            while not src.finished:
                raw = np.empty((65536, 2 * len(src.samples)), dtype=np.uint8)
                pos = np.empty(65536, dtype=np.int64)
                chrom = np.empty(65536, dtype=np.int32)
                missing = np.empty(65536, dtype=np.uint8)
                got = src.read_into(raw, pos, chrom, missing)
                if not got:
                    break
                keep_rows, keep_parent = [], []
                for r in range(got):
                    interval = intervals.get(src.contigs[chrom[r]].removeprefix("chr"))
                    if interval is None:
                        continue
                    beg, end, idx = interval
                    j = np.searchsorted(beg, pos[r], side="right") - 1
                    if j >= 0 and pos[r] <= end[j]:
                        keep_rows.append(r)
                        keep_parent.append(idx[j])
                if keep_rows:
                    chunks.append(raw[np.asarray(keep_rows)][:, hsel])
                    parent.extend(keep_parent)
        if not parent:
            raise ValueError("No SNPs overlap the cluster windows")
        parent = np.asarray(parent, dtype=np.int64)
        snp_count = len(parent)
        counts = np.bincount(parent, minlength=W)
        snp_weights = 1.0 / counts[parent]
        Z = np.vstack((Z, np.vstack(chunks)))
        k_vec = np.concatenate((k_vec, np.full(len(parent), 2, dtype=np.uint32)))
        weights = np.concatenate((np.ones(W), snp_weights))
        W = Z.shape[0]
        args.batches = 1
        print(
            f"Added {len(parent):,} SNPs ({snp_weights.sum():,.1f} window equivalents).", flush=True
        )

    M = int(np.sum(k_vec, dtype=np.uint64))
    if M * args.K > np.iinfo(np.uint32).max:
        raise ValueError("Too many cluster parameters for the native index range")

    # Histogram validation also supplies observed-only means and window counts
    c_tmp = np.insert(np.cumsum(k_vec, dtype=np.uint32), 0, 0)
    prior = args.p_prior
    p_vec = (
        np.empty(M)
        if args.loo
        or prior
        or (not args.random_init and args.supervised is None and args.projection is None)
        else None
    )
    obs = np.empty(W, dtype=np.int64)
    struct_cy.frequencies(Z, c_tmp.astype(np.int64), p_vec, obs)
    n_obs = int(obs.sum())
    if n_obs == 0:
        raise ValueError("No observed cluster assignments for ancestry estimation")
    n_miss = W * 2 * N - n_obs
    if n_miss:
        q_obs = admix_cy.observedCounts(Z)
        w_obs = obs.astype(np.uint32)
    else:
        q_obs = w_obs = None
    del obs

    # Count haplotype cluster alleles
    if weights is None:
        L_nrm = 2.0 * float(M) * float(N)
    else:
        q_obs = np.sum((Z.reshape(W, N, 2) != 255) * weights[:, None, None], axis=(0, 2))
        L_nrm = 2.0 * float(N) * float(np.dot(weights, k_vec))
    c_vec = c_tmp * args.K

    # Print information
    print(f"Data size: {N:,} samples, {W:,} windows, {M_clusters:,} clusters", flush=True)
    printMissing(n_miss, W * 2 * N)
    print("\nInitialization:", flush=True)

    t_read = time() - start
    t_init = time()
    y = None

    # Set up parameters
    rng = np.random.default_rng(args.seed)
    if args.supervised is not None:  # Supervised mode
        # Check input of ancestral sources
        if not os.path.isfile(args.supervised):
            raise ValueError("Population assignment file doesn't exist!")
        y = np.loadtxt(args.supervised, ndmin=1)
        if y.ndim != 1 or not np.all(np.isfinite(y)) or np.any(y != np.floor(y)):
            raise ValueError("Population assignments require one integer per sample")
        if args.keep is not None and y.shape[0] == N_all:
            y = y[keep]
        if y.shape[0] != N:
            raise ValueError("Number of samples differ between files!")
        if np.max(y) > args.K:
            raise ValueError("Wrong number of ancestral sources!")
        if np.min(y) < 0:
            raise ValueError("Wrong format for population assignments!")
        if np.any(y > 255):
            raise ValueError("Supervised source labels must fit 0..255")
        y = y.astype(np.uint8)
        print(f"Fixed ancestry: {np.sum(y > 0):,}/{N:,} samples", flush=True)

        # Initialize parameters
        P = rng.random(size=(M, args.K)).clip(min=1e-5, max=1 - (1e-5))
        P[:, np.unique(y[y > 0]) - 1] = 0.0
        admix_cy.superP(Z, P, k_vec, c_tmp, y)
        P = P.ravel()
    elif args.projection is not None:  # Projection mode
        # Load ancestral haplotype cluster frequencies
        print("Projecting onto reference frequencies.", flush=True)
        if not os.path.isfile(args.projection):
            raise ValueError("P matrix file/filelist doesn't exist!")
        if F > 1:  # Load multiple frequency files from filelist
            P_list = readPaths(args.projection)
            if len(P_list) != F:
                raise ValueError("Number of files doesn't match!")

            # Load in files to full matrix
            P = np.empty(M * args.K)
            for p in range(F):
                p_tmp = np.loadtxt(P_list[p], dtype=float, ndmin=2)
                shape = (int(c_vec[f_vec[p + 1]] - c_vec[f_vec[p]]) // args.K, args.K)
                if p_tmp.size == 0 and shape[0] == 0:
                    p_tmp = p_tmp.reshape(shape)
                if p_tmp.shape != shape:
                    raise ValueError("Reference P rows/columns do not match clusters and K")
                p_tmp = p_tmp.ravel()
                P[c_vec[f_vec[p]] : c_vec[f_vec[p + 1]]] = p_tmp
            del p_tmp
        else:  # Load single frequency file
            P = np.loadtxt(args.projection, dtype=float, ndmin=2)
            if P.shape != (M, args.K):
                raise ValueError("Number of haplotype clusters doesn't match!")
            P = P.ravel()

        if not np.all(np.isfinite(P)) or np.any(P < 0):
            raise ValueError("Reference frequencies must be finite and nonnegative")

        # Check that cluster frequencies sum to one
        p_sum = np.zeros((W, args.K))
        admix_cy.checkP(P, p_sum, k_vec, c_vec, args.K)
        if not np.allclose(p_sum[k_vec > 0], 1.0, atol=1e-3):
            raise ValueError("Wrong format for haplotype cluster alleles!")
        admix_cy.normalizeP(P, p_sum, k_vec, c_vec, args.K)
        del p_sum

    elif args.random_init:  # Random initialization
        print("Random initialization.", flush=True)
        P = rng.random(size=(M * args.K)).clip(min=1e-5, max=1 - (1e-5))
    else:  # SVD/ALS initialization
        print("Computing SVD/ALS estimates.", flush=True)
        ts = time()
        W_s = f_vec[ceil(F / args.subsampling)] if F > 1 else W
        try:
            U, S, V = functions.centerSVD(
                Z, p_vec, c_tmp, W_s, args.K, args.chunk, args.power, rng, w_obs
            )
        except ValueError:
            if W_s == W:
                raise
            print("SVD subset lacks rank. Using all windows.", flush=True)
            W_s = W
            U, S, V = functions.centerSVD(
                Z, p_vec, c_tmp, W_s, args.K, args.chunk, args.power, rng, w_obs
            )
        U_r = (
            functions.centerSub(Z, S, V, p_vec, c_tmp, W_s, args.chunk, w_obs) if W_s < W else None
        )
        p = p_vec.astype(np.float32)
        P, Q = functions.factorALS(
            U, S, V, p[: len(U)], k_vec[:W_s], c_tmp[: W_s + 1], args.als_iter, args.als_tole, rng
        )
        if U_r is not None:
            Y = np.ascontiguousarray(np.concatenate((U, U_r), axis=0) * S)
            P, Q = functions.alsStep(Y, V, p, k_vec, c_tmp, Q)
            del Y
        del U, U_r, p
        printTiming("SVD/ALS complete.", time() - ts)
        del S, V

    if args.supervised is not None or args.projection is not None or args.random_init:
        Q = rng.random(size=(N, args.K)).clip(min=1e-5, max=1 - (1e-5))
        Q /= np.sum(Q, axis=1, keepdims=True)

    # Enforce one probability domain for initialization, EM, and acceleration
    P = np.ascontiguousarray(P, dtype=float).ravel()
    Q = np.ascontiguousarray(Q, dtype=float)
    if args.projection is None:
        admix_cy.createP(P, k_vec, c_vec, args.K)
    admix_cy.createQ(Q)
    if q_obs is not None:
        Q[q_obs == 0] = 1.0 / args.K
    if y is not None:
        admix_cy.superQ(Q, y)
    del c_tmp
    if not prior and not args.loo:
        p_vec = None

    # Reuse fixed-partition scratch for every full and mini-batch update
    stats = dict(
        samples=N,
        windows=W,
        clusters=int(M_clusters),
        K=args.K,
        threads=args.threads,
        missing_assignments=int(n_miss),
        empty_windows=int(np.count_nonzero(k_vec == 0)),
        unobserved_samples=0 if q_obs is None else int(np.count_nonzero(q_obs == 0)),
        read_seconds=t_read,
        initialization_seconds=time() - t_init,
    )
    if prior:
        stats["p_prior"] = prior
    if args.loo:
        stats.update(loo=True, convergence="parameter RMSE")
    if snp_count:
        stats["snp_markers"] = snp_count
    Q1, T = np.empty_like(Q), np.empty_like(Q)
    Q2 = None if args.loo else np.empty_like(Q)
    P1 = None if args.projection else np.empty_like(P)
    P2 = None if args.projection or args.loo else np.empty_like(P)
    pt, qt = functions.emWorkspace(N, args.K, k_vec)
    ctx = (Z, k_vec, c_vec, T, pt, qt, w_obs, y, weights)
    em_kw = dict(pool=p_vec, prior=prior, scratch=P2)
    like = np.empty(W)

    def score():
        ll = admix_cy.likelihood(Z, P, Q, c_vec, like, w_obs, weights, L_nrm)
        obj = ll
        if prior:
            obj += admix_cy.priorScore(P, p_vec, k_vec, c_vec, args.K, prior) / L_nrm
        return obj, ll

    L_pre, ll_pre = score()
    if not np.isfinite(L_pre):
        raise ValueError("The initial model assigns zero probability to an observed cluster")
    stats["initial_loglike"] = ll_pre * L_nrm
    if prior:
        stats["initial_objective"] = L_pre * L_nrm
    metric = "Objective" if prior else "Log-like"
    print(f"Initial {metric.lower()}: {L_pre * L_nrm:,.1f}", flush=True)
    n_retry = 0
    if not args.loo:
        ts = time()
        functions.emStep(P, Q, P if P1 is not None else None, Q, ctx, qo=q_obs, **em_kw)
        functions.emQuasi(P, Q, P1, P2, Q1, Q2, ctx, qo=q_obs, **em_kw)
        functions.emStep(P, Q, P if P1 is not None else None, Q, ctx, qo=q_obs, **em_kw)
        L_cur, ll_cur = score()
        if not np.isfinite(L_cur) or L_cur < L_pre:
            # The ordinary EM state survives the accelerated warm-up
            if P1 is not None:
                P[:] = P1
            Q[:] = Q1
            L_cur, ll_cur = score()
            n_retry += 1
        if not np.isfinite(L_cur):
            raise ValueError("Non-finite ancestry objective during warm-up")
        L_pre, ll_pre = L_cur, ll_cur
        stats["priming_seconds"] = time() - ts
        printTiming("Warm-up complete.", stats["priming_seconds"])

    # Keep the batch schedule and checkpoint full-data convergence checks
    batches = 1 if args.loo else min(args.batches, W)
    s_win = np.arange(W, dtype=np.uint32) if batches > 1 else None
    L_bat = L_pre
    P_save = Q_save = None
    if batches == 1 and not args.loo:
        P_save = None if P1 is None else P.copy()
        Q_save = Q.copy()
    conv, stalled = False, False
    history = []
    ts = time()
    lap = ts
    print("\nLOO Q refinement:" if args.loo else "\nAncestry estimation:", flush=True)
    for it in range(1, args.iter + 1):
        if args.loo:
            functions.looStep(P, Q, P1, Q1, ctx, p_vec, prior, q_obs)
            change = max(admix_cy.damp(P, P1), admix_cy.damp(Q.ravel(), Q1.ravel()))
            P, P1 = P1, P
            Q, Q1 = Q1, Q
            conv = change <= args.tole
        elif batches > 1:
            rng.shuffle(s_win)
            step = W // batches
            for b in range(batches):
                rows = s_win[b * step : W if b == batches - 1 else (b + 1) * step]
                qo = (
                    None
                    if q_obs is None
                    else (
                        admix_cy.observedCounts(Z, rows)
                        if weights is None
                        else np.sum(
                            (Z[rows].reshape(len(rows), N, 2) != 255) * weights[rows, None, None],
                            axis=(0, 2),
                        )
                    )
                )
                functions.emQuasi(P, Q, P1, P2, Q1, Q2, ctx, rows, qo, **em_kw)
            functions.emQuasi(P, Q, P1, P2, Q1, Q2, ctx, qo=q_obs, **em_kw)
        else:
            if P1 is None:
                functions.emStep(P, Q, None, Q, ctx, qo=q_obs, **em_kw)
                functions.emQuasi(P, Q, P1, P2, Q1, Q2, ctx, qo=q_obs, **em_kw)
            else:
                functions.emQuasi(P, Q, P1, P2, Q1, Q2, ctx, qo=q_obs, **em_kw)
                functions.emStep(P, Q, P, Q, ctx, qo=q_obs, **em_kw)

        # Always score the actual final parameters, including a partial check interval
        if it % args.check and it != args.iter and not conv:
            continue
        L_cur, ll_cur = score()
        if not np.isfinite(L_cur):
            raise ValueError("Non-finite ancestry objective during estimation")
        if not args.loo and batches > 1:
            if L_cur < L_bat + args.tole:
                batches //= 2
                print(f"Mini-batches: {batches}", flush=True)
                L_bat = float("-inf")
                functions.emQuasi(P, Q, P1, P2, Q1, Q2, ctx, qo=q_obs, **em_kw)
                L_cur, ll_cur = score()
                if batches == 1:
                    P_save = None if P1 is None else P.copy()
                    Q_save = Q.copy()
                    L_pre = L_cur
                    ll_pre = ll_cur
            else:
                L_bat = L_cur
        elif not args.loo:
            if L_cur < L_pre:
                # Reject a decreasing accelerated block and try one ordinary EM step
                if P_save is not None:
                    P[:] = P_save
                Q[:] = Q_save
                functions.emStep(P, Q, P if P1 is not None else None, Q, ctx, qo=q_obs, **em_kw)
                L_cur, ll_cur = score()
                n_retry += 1
                if L_cur < L_pre:
                    if P_save is not None:
                        P[:] = P_save
                    Q[:] = Q_save
                    L_cur, ll_cur, stalled = L_pre, ll_pre, True
            conv = not stalled and 0 <= L_cur - L_pre <= args.tole
            if conv:
                # Confirm the stopping decision with an ordinary, constrained EM step
                if P1 is not None:
                    P1[:] = P
                Q1[:] = Q
                functions.emStep(P, Q, P if P1 is not None else None, Q, ctx, qo=q_obs, **em_kw)
                check, check_ll = score()
                if check < L_cur:
                    if P1 is not None:
                        P[:] = P1
                    Q[:] = Q1
                    stalled = L_cur - check > args.tole
                    conv = not stalled
                else:
                    conv = check - L_cur <= args.tole
                    L_cur = check
                    ll_cur = check_ll
            L_pre = L_cur
            ll_pre = ll_cur
            if P_save is not None:
                P_save[:] = P
            Q_save[:] = Q
        if not np.isfinite(L_cur):
            raise ValueError("Non-finite ancestry objective during estimation")
        row = dict(iteration=it, batches=batches, loglike=ll_cur * L_nrm)
        if args.loo:
            row["rmse"] = change
        if prior:
            row["objective"] = L_cur * L_nrm
        history.append(row)
        now = time()
        printTiming(f"({it:,})  {metric}: {L_cur * L_nrm:,.1f}", now - lap)
        lap = now
        if conv or stalled:
            break
    stats.update(
        iterations=it,
        converged=conv,
        stalled=stalled,
        recoveries=n_retry,
        final_batches=batches,
        em_seconds=time() - ts,
        final_loglike=ll_cur * L_nrm,
        history=history,
    )
    if prior:
        stats["final_objective"] = L_cur * L_nrm
    print(
        "Converged."
        if conv
        else ("Stopped without an improving EM step." if stalled else "Iteration limit reached."),
        flush=True,
    )

    # Publish all requested files together after successful calculation
    sfxs = [".Q", ".ids", ".log"]
    if not args.no_freqs and P1 is not None:
        sfxs += [f".{args.prefix}{f + 1}.P" for f in range(F)] + [".plist"] if F > 1 else [".P"]
    inputs = [f"{p}{s}" for p in Z_list for s in (".bca", ".win", ".ids")]
    inputs += [
        p
        for p in (args.filelist, args.keep, args.supervised, args.projection, snp_vcf)
        if p is not None
    ]
    if args.projection and F > 1:
        inputs += P_list
    stale = (".P", ".plist") + tuple(f".{args.prefix}{f + 1}.P" for f in range(F) if F > 1)
    ts = time()
    with ExitStack() as stack:
        out = stageOutputs(stack, f_out, sfxs, inputs, stale=stale)
        np.savetxt(out[".Q"], Q, fmt="%.10g")
        np.savetxt(out[".ids"], q_ids, fmt="%s")
        if not args.no_freqs and P1 is not None:
            if F > 1:
                for f in range(F):
                    part = P[c_vec[f_vec[f]] : c_vec[f_vec[f + 1]]].reshape(-1, args.K)
                    np.savetxt(out[f".{args.prefix}{f + 1}.P"], part, fmt="%.10g")
                out[".plist"].write_text(
                    "".join(f"{Path(f_out).absolute()}.{args.prefix}{f + 1}.P\n" for f in range(F))
                )
            else:
                np.savetxt(
                    out[".P"], P[: M_clusters * args.K].reshape(M_clusters, args.K), fmt="%.10g"
                )
        stats.update(output_seconds=time() - ts, elapsed_seconds=time() - start)
        writeLog(out, "admix", args, stats, f_out)
        commitOutputs(f_out, out, stale=stale)
    printDone(f_out, out, stats["elapsed_seconds"])
    return stats
