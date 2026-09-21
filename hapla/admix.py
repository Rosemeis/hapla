"""
hapla.
Ancestry estimation using accelerated EM algorithm.
"""

__author__ = "Jonas Meisner"

# Libraries
import os
from datetime import datetime
from time import time
from hapla import __version__


##### hapla admix #####
def main(args, deaf):
    print("-----------------------------------")
    print(f"hapla by Jonas Meisner (v{__version__})")
    print(f"hapla admix using {args.threads} thread(s)")
    print("-----------------------------------")

    # Check input
    assert (args.filelist is not None) or (args.clusters is not None), (
        "No input data (--filelist or --clusters)!"
    )
    assert args.K > 1, "Please select K > 1!"
    if args.keep is not None:
        assert os.path.isfile(args.keep), "Keep file doesn't exist!"
    assert args.threads > 0, "Please select a valid number of threads!"
    assert args.seed >= 0, "Please select a valid seed!"
    assert args.iter > 0, "Please select a valid number of iterations!"
    assert args.tole >= 0.0, "Please select a valid tolerance!"
    assert args.batches > 0, "Please select a valid number of mini-batches!"
    assert args.check > 0, "Please select a valid value for convergence check!"
    assert args.power > 0, "Please select a valid number of power iterations!"
    assert args.chunk > 0, "Please select a valid SVD chunk size!"
    assert args.als_iter > 0, "Please select a valid number of iterations in ALS!"
    assert args.als_tole >= 0.0, "Please select a valid tolerance in ALS!"
    assert args.subsampling > 1, "Please select a valid subsampling factor!"
    print("Estimating ancestry proportions.")
    print(f"K={args.K}, seed={args.seed}, batches={args.batches}\n")
    p_out = f"{args.out}.project" if (args.projection is not None) else f"{args.out}"
    f_out = f"{p_out}.K{args.K}.s{args.seed}"
    start = time()

    # Create log-file of used arguments
    full = vars(args)
    mand = ["seed", "batches"]
    with open(f"{args.out}.K{args.K}.s{args.seed}.log", "w") as log:
        log.write(f"hapla v{__version__}\n")
        log.write("hapla admix\n")
        log.write(f"Time: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")
        log.write(f"Directory: {os.getcwd()}\n")
        log.write("Options:\n")
        for key in full:
            if full[key] != deaf[key]:
                log.write(f"\t--{key}\n") if (type(full[key]) is bool) else log.write(
                    f"\t--{key} {full[key]}\n"
                )
            elif key in mand:
                log.write(f"\t--{key} {full[key]}\n")
    del full, deaf, mand

    # Control threads of external numerical libraries
    os.environ["MKL_NUM_THREADS"] = str(args.threads)
    os.environ["MKL_MAX_THREADS"] = str(args.threads)
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["OMP_MAX_THREADS"] = str(args.threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(args.threads)
    os.environ["NUMEXPR_MAX_THREADS"] = str(args.threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(args.threads)
    os.environ["OPENBLAS_MAX_THREADS"] = str(args.threads)

    # Import numerical libraries and cython functions
    import numpy as np
    from math import ceil
    from hapla import functions
    from hapla import admix_cy
    from hapla import shared_cy

    def missing_file(z):
        for suffix in (".bcmis", ".bca.miss", ".miss"):
            m_file = f"{z}{suffix}"
            if os.path.isfile(m_file):
                return m_file
        return None

    def read_keep_ids(k_file):
        ids = np.genfromtxt(k_file, dtype=np.str_, comments=None)
        ids = np.asarray(ids)
        if ids.ndim == 0:
            ids = ids.reshape(1)
        assert ids.ndim == 1, "Keep file must contain one sample ID per line!"
        return np.atleast_1d(ids).astype(np.str_)

    # Prepare list of data files
    if args.filelist is not None:
        W = 0  # Counter for windows
        Z_list = []  # List of filenames
        with open(args.filelist) as f:
            for z_file in f:
                # Check input across files (chromosomes) and count windows
                z = z_file.strip("\n")
                Z_list.append(z)
                assert os.path.isfile(f"{z}.bca"), "bca file doesn't exist!"
                assert os.path.isfile(f"{z}.ids"), "ids file doesn't exist!"
                assert os.path.isfile(f"{z}.win"), "win file doesn't exist!"
                if W == 0:  # First file
                    z_ids = np.atleast_1d(np.genfromtxt(f"{z}.ids", dtype=np.str_))
                    k_vec = np.genfromtxt(f"{z}.win", dtype=np.uint32, usecols=[5])
                    N_all = z_ids.shape[0]
                    W = k_vec.shape[0]
                    w_list = [W]
                else:  # Loop files
                    t_ids = np.atleast_1d(np.genfromtxt(f"{z}.ids", dtype=np.str_))
                    assert np.sum(z_ids != t_ids) == 0, (
                        "Samples don't match across files!"
                    )
                    k_tmp = np.genfromtxt(f"{z}.win", dtype=np.uint32, usecols=[5])
                    k_vec = np.append(k_vec, k_tmp)
                    W += k_tmp.shape[0]
                    w_list.append(k_tmp.shape[0])
        F = len(Z_list)
        w_vec = np.array(w_list, dtype=np.uint32)
        f_vec = np.insert(np.cumsum(w_vec, dtype=np.uint32), 0, 0)
        del w_list
    else:  # Single file (chromosome)
        F = 1
        Z_list = [args.clusters]
        assert os.path.isfile(f"{Z_list[0]}.bca"), "bca file doesn't exist!"
        assert os.path.isfile(f"{Z_list[0]}.ids"), "ids file doesn't exist!"
        assert os.path.isfile(f"{Z_list[0]}.win"), "win file doesn't exist!"
        k_vec = np.genfromtxt(f"{Z_list[0]}.win", dtype=np.uint32, usecols=[5])
        z_ids = np.atleast_1d(np.genfromtxt(f"{Z_list[0]}.ids", dtype=np.str_))
        N_all = z_ids.shape[0]
        W = k_vec.shape[0]
        w_vec = np.array([W], dtype=np.uint32)

    # Select samples from keep file
    if args.keep is not None:
        keep_ids = read_keep_ids(args.keep)
        keep_idx = np.flatnonzero(np.isin(z_ids, keep_ids)).astype(np.uint32)
        assert keep_idx.shape[0] > 0, "No samples from keep file found in ids file!"
        hap_idx = np.empty(keep_idx.shape[0] * 2, dtype=np.uint32)
        hap_idx[0::2] = 2 * keep_idx
        hap_idx[1::2] = 2 * keep_idx + 1
        N = keep_idx.shape[0]
        print(f"Keeping {N}/{N_all} samples from {args.keep}.")
        del keep_ids
    else:
        keep_idx = None
        hap_idx = None
        N = N_all
    q_ids = z_ids if keep_idx is None else z_ids[keep_idx]
    print(f"Parsing {F} file(s).")

    # Load haplotype cluster assignments from binary hapla format
    B = 0
    Z = np.zeros((W, 2 * N), dtype=np.uint8)
    Z_miss = None
    miss_count = 0
    for z in np.arange(F):
        with open(f"{Z_list[z]}.bca", "rb") as f:
            # Check magic numbers
            magic = np.fromfile(f, dtype=np.uint8, count=3)
            assert np.allclose(magic, np.array([7, 9, 13], dtype=np.uint8)), (
                "Magic number doesn't match file format!"
            )

            # Add haplotype cluster assignments to container
            z_tmp = np.fromfile(f, dtype=np.uint8)
            z_tmp = z_tmp.reshape(w_vec[z], 2 * N_all)
            if hap_idx is not None:
                z_tmp = z_tmp[:, hap_idx]
            Z[B : (B + w_vec[z]), :] = z_tmp

            # Add missingness sidecar mask when present
            m_file = missing_file(Z_list[z])
            if m_file is not None:
                m_tmp = np.fromfile(m_file, dtype=np.uint8)
                m_size = w_vec[z] * 2 * N_all
                if m_tmp.shape[0] == (m_size + 3):
                    assert np.allclose(
                        m_tmp[:3], np.array([7, 9, 14], dtype=np.uint8)
                    ), "Magic number doesn't match missingness mask format!"
                    m_tmp = m_tmp[3:]
                assert m_tmp.shape[0] == m_size, (
                    "Missingness mask doesn't match cluster assignments!"
                )
                m_tmp = m_tmp.reshape(w_vec[z], 2 * N_all)
                if hap_idx is not None:
                    m_tmp = m_tmp[:, hap_idx]
                m_tmp = (m_tmp > 0).astype(np.uint8)
                if np.any(m_tmp):
                    if Z_miss is None:
                        Z_miss = np.zeros_like(Z)
                    Z_miss[B : (B + w_vec[z]), :] = m_tmp
                    miss_count += np.sum(m_tmp, dtype=np.uint64)
            B += w_vec[z]
        print(f"\rParsed file {z + 1}/{F}", end="")
    del magic, z_tmp
    has_missing = Z_miss is not None
    if has_missing:
        H_obs = (Z_miss.reshape(W, N, 2) == 0).sum(axis=2, dtype=np.uint32)
        q_obs = np.sum(H_obs, axis=0, dtype=np.uint32)
        w_obs = np.sum(Z_miss == 0, axis=1, dtype=np.uint32)
        obs_count = np.sum(w_obs, dtype=np.uint64)
    else:
        H_obs = None
        q_obs = None
        w_obs = None
        obs_count = np.uint64(W * 2 * N)

    # Count haplotype cluster alleles
    M = np.sum(k_vec, dtype=np.uint32)
    L_nrm = 2.0 * float(M) * float(N)
    c_vec = np.insert(np.cumsum(k_vec * args.K, dtype=np.uint32), 0, 0)

    # Print information
    print(
        "\rLoaded haplotype cluster assignments:\n"
        + f"- {N} samples\n"
        + f"- {W} windows\n"
        + f"- {M} clusters\n"
    )
    if has_missing:
        print(
            "Detected missingness sidecar mask:\n"
            + f"- {miss_count} missing assignments\n"
            + f"- {obs_count} observed assignments\n"
        )

    # Set up parameters
    rng = np.random.default_rng(args.seed)
    if args.supervised is not None:  # Supervised mode
        # Check input of ancestral sources
        print("Ancestry estimation in supervised mode!")
        assert os.path.isfile(args.supervised), (
            "Population assignment file doesn't exist!"
        )
        y = np.genfromtxt(args.supervised, dtype=np.uint8).reshape(-1)
        if args.keep is not None and y.shape[0] == N_all:
            y = y[keep_idx]
        assert y.shape[0] == N, "Number of samples differ between files!"
        assert np.max(y) <= args.K, "Wrong number of ancestral sources!"
        assert np.min(y) >= 0, "Wrong format for population assignments!"
        print(f"{np.sum(y > 0)}/{N} samples with fixed ancestry.")

        # Count ancestral sources
        x = np.unique(y[y > 0])
        x -= 1
        x = np.sort(x)

        # Initialize parameters
        P = rng.random(size=(M, args.K)).clip(min=1e-5, max=1 - (1e-5))
        Q = rng.random(size=(N, args.K)).clip(min=1e-5, max=1 - (1e-5))
        Q /= np.sum(Q, axis=1, keepdims=True)
        P[:, x] = 0.0
        c_tmp = np.insert(np.cumsum(k_vec, dtype=np.uint32), 0, 0)
        if has_missing:
            admix_cy.superPMiss(Z, Z_miss, P, k_vec, c_tmp, y)
        else:
            admix_cy.superP(Z, P, k_vec, c_tmp, y)
        admix_cy.superQ(Q, y)
        P = P.flatten()
        del x, c_tmp
    elif args.projection is not None:  # Projection mode
        # Load ancestral haplotype cluster frequencies
        print("Ancestry estimation in projection mode!")
        assert os.path.isfile(args.projection), "P matrix file/filelist doesn't exist!"
        if F > 1:  # Load multiple frequency files from filelist
            P_list = []
            with open(args.projection) as f:
                for p, p_file in enumerate(f):
                    assert os.path.isfile(p_file.strip("\n")), (
                        f"The {p + 1}/{F} matrix file doesn't exist!"
                    )
                    P_list.append(p_file.strip("\n"))
            assert len(P_list) == F, "Number of files doesn't match!"

            # Load in files to full matrix
            P = np.zeros(M * args.K)
            M_tmp = 0
            for p in np.arange(F):
                p_tmp = np.genfromtxt(P_list[p], dtype=float).reshape(-1)
                M_tmp += p_tmp.shape[0]
                P[c_vec[f_vec[p]] : c_vec[f_vec[p + 1]]] = p_tmp
            assert np.any(P > 0.0), "Wrong format for haplotype cluster alleles!"
            assert M_tmp == (M * args.K), "Number of haplotype clusters doesn't match!"
            del P_list, p_tmp
        else:  # Load single frequency file
            P = np.genfromtxt(args.projection, dtype=float).reshape(-1)
            assert P.shape[0] == (M * args.K), (
                "Number of haplotype clusters doesn't match!"
            )

        # Check that cluster frequencies sum to one
        p_sum = np.zeros((W, args.K))
        admix_cy.checkP(P, p_sum, k_vec, c_vec, args.K)
        assert np.allclose(p_sum, 1.0, atol=1e-3), (
            "Wrong format for haplotype cluster alleles!"
        )
        del p_sum

        # Initialize Q matrix
        Q = rng.random(size=(N, args.K)).clip(min=1e-5, max=1 - (1e-5))
        Q /= np.sum(Q, axis=1, keepdims=True)
    else:
        if args.random_init:  # Random initialization
            print("Random initialization.")
            P = rng.random(size=(M * args.K)).clip(min=1e-5, max=1 - (1e-5))
            Q = rng.random(size=(N, args.K)).clip(min=1e-5, max=1 - (1e-5))
            Q /= np.sum(Q, axis=1, keepdims=True)
            admix_cy.createP(P, k_vec, c_vec, args.K)
        else:  # SVD/ALS initialization
            print("SVD/ALS initialization.", end="", flush=True)
            ts = time()
            p_vec = np.zeros(M, dtype=np.float32)
            c_tmp = np.insert(np.cumsum(k_vec, dtype=np.uint32), 0, 0)
            if has_missing:
                shared_cy.estimateFreqMiss(Z, Z_miss, p_vec, k_vec, c_tmp)
            else:
                shared_cy.estimateFreq(Z, p_vec, k_vec, c_tmp)
            if F > 1:  # Initialize based on subsampled chromosomes/files
                W_s = f_vec[ceil(F / args.subsampling)]
                if has_missing:
                    U_s, S, V = functions.centerSVDMiss(
                        Z,
                        Z_miss,
                        p_vec,
                        k_vec,
                        c_tmp,
                        W_s,
                        args.K,
                        args.chunk,
                        args.power,
                        rng,
                    )
                    U_r = functions.centerSubMiss(
                        Z, Z_miss, S, V, p_vec, k_vec, c_tmp, W_s, args.chunk
                    )
                else:
                    U_s, S, V = functions.centerSVD(
                        Z, p_vec, k_vec, c_tmp, W_s, args.K, args.chunk, args.power, rng
                    )
                    U_r = functions.centerSub(
                        Z, S, V, p_vec, k_vec, c_tmp, W_s, args.chunk
                    )
                P, Q = functions.factorSub(
                    U_s,
                    U_r,
                    S,
                    V,
                    p_vec,
                    k_vec,
                    c_tmp,
                    W_s,
                    args.als_iter,
                    args.als_tole,
                    rng,
                )
                del U_s, U_r
            else:
                if has_missing:
                    U, S, V = functions.centerSVDMiss(
                        Z,
                        Z_miss,
                        p_vec,
                        k_vec,
                        c_tmp,
                        W,
                        args.K,
                        args.chunk,
                        args.power,
                        rng,
                    )
                else:
                    U, S, V = functions.centerSVD(
                        Z, p_vec, k_vec, c_tmp, W, args.K, args.chunk, args.power, rng
                    )
                P, Q = functions.factorALS(
                    U, S, V, p_vec, k_vec, c_tmp, args.als_iter, args.als_tole, rng
                )
                del U
            print(f"\rSVD/ALS initialization.\t\t({time() - ts:.1f}s)")
            del p_vec, c_tmp, S, V
        y = None

    # Set up containers for EM algorithm
    Q1 = np.zeros_like(Q)
    Q2 = np.zeros_like(Q)
    Q_tmp = np.zeros_like(Q)
    if args.projection is None:
        L = np.max(k_vec)  # Max clusters for inner dynamic arrays
        P1 = np.zeros_like(P)
        P2 = np.zeros_like(P)

    # Estimate initial log-likelihood
    if has_missing:
        L_pre = admix_cy.loglikeMiss(Z, Z_miss, P, Q, c_vec)
    else:
        L_pre = admix_cy.loglike(Z, P, Q, c_vec)
    print(f"Initial log-like: {L_pre * L_nrm:.1f}")

    # Accelerated priming iteration
    ts = time()
    print("Priming iteration.", end="", flush=True)
    if args.projection is None:
        if has_missing:
            functions.stepsMiss(Z, Z_miss, P, Q, Q_tmp, k_vec, c_vec, y, L, w_obs, q_obs)
            functions.quasiMiss(
                Z, Z_miss, P, Q, Q_tmp, P1, P2, Q1, Q2, k_vec, c_vec, y, L, w_obs, q_obs
            )
            functions.stepsMiss(Z, Z_miss, P, Q, Q_tmp, k_vec, c_vec, y, L, w_obs, q_obs)
        else:
            functions.steps(Z, P, Q, Q_tmp, k_vec, c_vec, y, L)
            functions.quasi(Z, P, Q, Q_tmp, P1, P2, Q1, Q2, k_vec, c_vec, y, L)
            functions.steps(Z, P, Q, Q_tmp, k_vec, c_vec, y, L)
    else:
        if has_missing:
            functions.proStepsMiss(Z, Z_miss, P, Q, Q_tmp, c_vec, q_obs)
            functions.proQuasiMiss(Z, Z_miss, P, Q, Q_tmp, Q1, Q2, c_vec, q_obs)
            functions.proStepsMiss(Z, Z_miss, P, Q, Q_tmp, c_vec, q_obs)
        else:
            functions.proSteps(Z, P, Q, Q_tmp, c_vec)
            functions.proQuasi(Z, P, Q, Q_tmp, Q1, Q2, c_vec)
            functions.proSteps(Z, P, Q, Q_tmp, c_vec)
    print(f"\rPriming iteration.\t\t({time() - ts:.1f}s)\n")

    # Set up mini-batch parameters
    if args.batches > 1:
        L_bat = L_pre
        s_win = np.arange(W, dtype=np.uint32)
        W_bat = W // args.batches
        print("Mini-batch accelerated EM algorithm.")
        print(f"Using {args.batches} mini-batches.")
    else:
        print("Standard accelerated EM algorithm.")

    # Accelerated EM algorithm (fastmixture)
    ts = time()
    for it in range(args.iter):
        if args.batches > 1:  # Quasi-Newton mini-batch updates
            rng.shuffle(s_win)  # Shuffle window order
            for b in np.arange(args.batches):
                s_beg = b * W_bat
                s_end = W if b == (args.batches - 1) else (b + 1) * W_bat
                s_bat = s_win[s_beg:s_end]
                if args.projection is None:
                    if has_missing:
                        functions.batQuasiMiss(
                            Z,
                            Z_miss,
                            P,
                            Q,
                            Q_tmp,
                            P1,
                            P2,
                            Q1,
                            Q2,
                            k_vec,
                            c_vec,
                            s_bat,
                            y,
                            L,
                            w_obs,
                            H_obs,
                        )
                    else:
                        functions.batQuasi(
                            Z, P, Q, Q_tmp, P1, P2, Q1, Q2, k_vec, c_vec, s_bat, y, L
                        )
                else:
                    if has_missing:
                        functions.proBatchMiss(
                            Z, Z_miss, P, Q, Q_tmp, Q1, Q2, c_vec, s_bat, H_obs
                        )
                    else:
                        functions.proBatch(Z, P, Q, Q_tmp, Q1, Q2, c_vec, s_bat)
            if args.projection is None:
                if has_missing:
                    functions.quasiMiss(
                        Z,
                        Z_miss,
                        P,
                        Q,
                        Q_tmp,
                        P1,
                        P2,
                        Q1,
                        Q2,
                        k_vec,
                        c_vec,
                        y,
                        L,
                        w_obs,
                        q_obs,
                    )
                else:
                    functions.quasi(Z, P, Q, Q_tmp, P1, P2, Q1, Q2, k_vec, c_vec, y, L)
            else:
                if has_missing:
                    functions.proQuasiMiss(Z, Z_miss, P, Q, Q_tmp, Q1, Q2, c_vec, q_obs)
                else:
                    functions.proQuasi(Z, P, Q, Q_tmp, Q1, Q2, c_vec)
        else:  # Full updates
            if args.projection is None:
                if has_missing:
                    functions.quasiMiss(
                        Z,
                        Z_miss,
                        P,
                        Q,
                        Q_tmp,
                        P1,
                        P2,
                        Q1,
                        Q2,
                        k_vec,
                        c_vec,
                        y,
                        L,
                        w_obs,
                        q_obs,
                    )
                    functions.stepsMiss(
                        Z, Z_miss, P, Q, Q_tmp, k_vec, c_vec, y, L, w_obs, q_obs
                    )
                else:
                    functions.quasi(Z, P, Q, Q_tmp, P1, P2, Q1, Q2, k_vec, c_vec, y, L)
                    functions.steps(Z, P, Q, Q_tmp, k_vec, c_vec, y, L)
            else:
                if has_missing:
                    functions.proStepsMiss(Z, Z_miss, P, Q, Q_tmp, c_vec, q_obs)
                    functions.proQuasiMiss(Z, Z_miss, P, Q, Q_tmp, Q1, Q2, c_vec, q_obs)
                else:
                    functions.proSteps(Z, P, Q, Q_tmp, c_vec)
                    functions.proQuasi(Z, P, Q, Q_tmp, Q1, Q2, c_vec)

        # Log-likelihood convergence check
        if ((it + 1) % args.check) == 0:
            if has_missing:
                L_cur = admix_cy.loglikeMiss(Z, Z_miss, P, Q, c_vec)
            else:
                L_cur = admix_cy.loglike(Z, P, Q, c_vec)
            print(
                f"({it + 1})\tLog-like: {L_cur * L_nrm:.1f}\t({time() - ts:.1f}s)",
                flush=True,
            )
            if args.batches > 1:
                if L_cur < (L_bat + args.tole):
                    args.batches = args.batches // 2  # Halve number of batches
                    if args.batches > 1:
                        print(f"Halving mini-batches to {args.batches}.")
                        L_bat = float("-inf")
                        W_bat = W // args.batches
                    else:  # Turn off mini-batch acceleration
                        print("Running standard updates.")
                    L_pre = L_cur
                    if args.projection is None:
                        if has_missing:
                            functions.quasiMiss(
                                Z,
                                Z_miss,
                                P,
                                Q,
                                Q_tmp,
                                P1,
                                P2,
                                Q1,
                                Q2,
                                k_vec,
                                c_vec,
                                y,
                                L,
                                w_obs,
                                q_obs,
                            )
                        else:
                            functions.quasi(
                                Z, P, Q, Q_tmp, P1, P2, Q1, Q2, k_vec, c_vec, y, L
                            )
                    else:
                        if has_missing:
                            functions.proQuasiMiss(
                                Z, Z_miss, P, Q, Q_tmp, Q1, Q2, c_vec, q_obs
                            )
                        else:
                            functions.proQuasi(Z, P, Q, Q_tmp, Q1, Q2, c_vec)
                else:
                    L_bat = L_cur
            else:  # Check for convergence
                if L_cur < (L_pre + args.tole):
                    print("Converged!\n")
                    break
                L_pre = L_cur
            ts = time()
    L_cur *= L_nrm
    print(f"Final log-likelihood: {L_cur:.1f}")

    # Save output
    np.savetxt(f"{f_out}.Q", Q, fmt="%.6f")
    np.savetxt(f"{f_out}.ids", q_ids, fmt="%s")
    print(f"Saved Q matrix as {f_out}.Q")
    print(f"Saved Q sample IDs as {f_out}.ids")
    if not args.no_freqs and (args.projection is None):
        if F > 1:  # Save P file for each file (chromosome)
            for p in np.arange(F):
                P_tmp = P[c_vec[f_vec[p]] : c_vec[f_vec[p + 1]]].reshape(-1, args.K)
                np.savetxt(
                    f"{f_out}.{args.prefix}{p + 1}.P",
                    P_tmp,
                    fmt="%.6f",
                )
            del P_tmp
            with open(f"{f_out}.pfilelist", "w") as f:
                for p in np.arange(F):
                    f.write(f"{f_out}.{args.prefix}{p + 1}.P\n")
            print(f"Saved P matrices as {f_out}.{args.prefix}{{1..{F}}}.P")
            print(f"Saved P matrix filelist as {f_out}.pfilelist")
        else:  # Single file (chromosome)
            np.savetxt(f"{f_out}.P", P.reshape(M, args.K), fmt="%.6f")
            print(f"Saved P matrix as {f_out}.P")

    # Print elapsed time for computation
    t_tot = time() - start
    t_min = int(t_tot // 60)
    t_sec = int(t_tot - t_min * 60)
    print(f"\nTotal elapsed time: {t_min}m{t_sec}s")

    # Write to log-file
    with open(f"{f_out}.log", "a") as log:
        log.write(f"\nFinal log-likelihood: {L_cur:.1f}\n")
        log.write(f"Converged in {it + 1} iterations.\n")
        log.write(f"\nSaved Q matrix as {args.out}.{p_out}K{args.K}.s{args.seed}.Q\n")
        log.write(f"Saved Q sample IDs as {f_out}.ids\n")
        if not args.no_freqs and (args.projection is None):
            if F > 1:
                log.write(f"Saved P matrices as {f_out}.{args.prefix}{{1..{F}}}.P\n")
                log.write(f"Saved P matrix filelist as {f_out}.pfilelist")
            else:
                log.write(f"Saved P matrix as {f_out}.P\n")
        log.write(f"\nTotal elapsed time: {t_min}m{t_sec}s\n")


##### Main exception #####
assert __name__ != "__main__", "Please use the 'hapla admix' command!"
