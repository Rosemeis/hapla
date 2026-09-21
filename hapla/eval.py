"""
hapla.
Evaluate fit of admix model by computing correlations of residuals.
"""

__author__ = "Thomas Bøggild"

# Libraries
import os
from datetime import datetime
from time import time
from hapla import __version__


##### hapla eval #####
def main(args, deaf):
    print("-----------------------------------")
    print(f"hapla by Jonas Meisner (v{__version__})")
    print(f"hapla eval using {args.threads} thread(s)")
    print("-----------------------------------\n")

    # Check input
    assert (args.filelist is not None) or (args.clusters is not None), (
        "No input data (--filelist or --clusters)!"
    )
    assert args.threads > 0, "Please select a valid number of threads!"
    assert args.qfile is not None, "No Q file provided (--qfile)!"
    if args.keep is not None:
        assert os.path.isfile(args.keep), "Keep file doesn't exist!"

    start = time()

    # Control threads of external numerical libraries
    os.environ["MKL_NUM_THREADS"] = str(args.threads)
    os.environ["MKL_MAX_THREADS"] = str(args.threads)
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["OMP_MAX_THREADS"] = str(args.threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(args.threads)
    os.environ["NUMEXPR_MAX_THREADS"] = str(args.threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(args.threads)
    os.environ["OPENBLAS_MAX_THREADS"] = str(args.threads)

    # Create log-file of used arguments
    full = vars(args)
    with open(f"{args.out}.log", "w") as log:
        log.write(f"hapla v{__version__}\n")
        log.write("hapla eval\n")
        log.write(f"Time: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")
        log.write(f"Directory: {os.getcwd()}\n")
        log.write("Options:\n")
        for key in full:
            if full[key] != deaf[key]:
                log.write(f"\t--{key}\n") if (type(full[key]) is bool) else log.write(
                    f"\t--{key} {full[key]}\n"
                )
    del full, deaf

    # Import numerical libraries and cython functions
    import numpy as np
    from hapla import eval_cy

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
        Z_list = []  # List of filenames
        with open(args.filelist) as f:
            for z_file in f:
                # Check input across files and count windows
                z = z_file.strip("\n")
                Z_list.append(z)
                assert os.path.isfile(f"{z}.bca"), "bca file doesn't exist!"
                assert os.path.isfile(f"{z}.ids"), "ids file doesn't exist!"
                assert os.path.isfile(f"{z}.win"), "win file doesn't exist!"
                if len(Z_list) == 1:  # First file
                    z_ids = np.atleast_1d(np.genfromtxt(f"{z}.ids", dtype=np.str_))
                    k_vec = np.genfromtxt(
                        f"{z}.win", dtype=np.uint32, usecols=[5]
                    ).reshape(-1)
                    N_all = z_ids.shape[0]
                    w_list = [k_vec.shape[0]]
                else:  # Loop files
                    t_ids = np.atleast_1d(np.genfromtxt(f"{z}.ids", dtype=np.str_))
                    assert np.sum(z_ids != t_ids) == 0, (
                        "Samples do not match across files!"
                    )
                    k_tmp = np.genfromtxt(
                        f"{z}.win", dtype=np.uint32, usecols=[5]
                    ).reshape(-1)
                    k_vec = np.append(k_vec, k_tmp)
                    w_list.append(k_tmp.shape[0])
        F = len(Z_list)
        w_vec = np.array(w_list, dtype=np.uint32)
        del w_list
    else:  # Single file (chromosome)
        F = 1
        Z_list = [args.clusters]
        assert os.path.isfile(f"{Z_list[0]}.bca"), "bca file doesn't exist!"
        assert os.path.isfile(f"{Z_list[0]}.ids"), "ids file doesn't exist!"
        assert os.path.isfile(f"{Z_list[0]}.win"), "win file doesn't exist!"
        k_vec = np.genfromtxt(
            f"{Z_list[0]}.win", dtype=np.uint32, usecols=[5]
        ).reshape(-1)
        w_vec = np.array([k_vec.shape[0]], dtype=np.uint32)
        z_ids = np.atleast_1d(np.genfromtxt(f"{Z_list[0]}.ids", dtype=np.str_))
        N_all = z_ids.shape[0]

    # Select samples from keep file
    if args.keep is not None:
        keep_ids = read_keep_ids(args.keep)
        id_to_idx = {sample_id: i for i, sample_id in enumerate(z_ids)}
        keep_idx = np.array(
            [id_to_idx[sample_id] for sample_id in keep_ids if sample_id in id_to_idx],
            dtype=np.uint32,
        )
        assert keep_idx.shape[0] > 0, "No samples from keep file found in ids file!"
        assert np.unique(keep_idx).shape[0] == keep_idx.shape[0], (
            "Keep file contains duplicate sample IDs!"
        )
        hap_idx = np.empty(keep_idx.shape[0] * 2, dtype=np.uint32)
        hap_idx[0::2] = 2 * keep_idx
        hap_idx[1::2] = 2 * keep_idx + 1
        z_ids = z_ids[keep_idx]
        N = keep_idx.shape[0]
        print(f"Keeping {N}/{N_all} samples from {args.keep}.")
        del keep_ids
    else:
        keep_idx = None
        hap_idx = None
        N = N_all
    print(f"Parsing {F} file(s).")
    miss_files = [missing_file(z) for z in Z_list]
    has_missing = any(m_file is not None for m_file in miss_files)

    # Load Q matrix
    Q = np.ascontiguousarray(np.genfromtxt(args.qfile, dtype=float))
    if Q.ndim == 1:
        Q = Q.reshape(-1, 1)
    assert Q.shape[0] == N, "Number of samples doesn't match!"
    assert Q.shape[1] > 1, "Please provide at least two ancestral components!"
    A = np.ascontiguousarray(np.linalg.pinv(np.dot(Q.T, Q)))
    np.savetxt(f"{args.out}.ids", z_ids, fmt="%s")

    # Covariance containers
    C = np.zeros((N, N), dtype=np.float64)
    V = np.zeros(N, dtype=np.float64)

    # Loop over chromosomes
    print("Computing correlations of residuals.")
    w_cnt = 0
    for z in np.arange(F):  # Loop through files
        print(f"Processing file {z + 1}/{F}")
        t_chr = time()
        W_chr = w_vec[z]
        k_chr = k_vec[w_cnt : (w_cnt + W_chr)]
        w_cnt += W_chr

        # Load haplotype cluster assignment file
        with open(f"{Z_list[z]}.bca", "rb") as f:
            # Check magic numbers
            magic = np.fromfile(f, dtype=np.uint8, count=3)
            assert np.allclose(magic, np.array([7, 9, 13], dtype=np.uint8)), (
                "Magic number doesn't match file format!"
            )

            # Add haplotype cluster assignments to container
            Z_chr = np.fromfile(f, dtype=np.uint8)
            Z_chr.shape = (W_chr, 2 * N_all)
            if hap_idx is not None:
                Z_chr = Z_chr[:, hap_idx]
            Z_chr = np.ascontiguousarray(Z_chr)
        assert np.all(np.max(Z_chr, axis=1) < k_chr), "Number of clusters doesn't match!"

        # Project cluster counts onto the ADMIXTURE Q-space and accumulate residual covariances.
        m_file = miss_files[z]
        if m_file is None:
            eval_cy.covar(C, V, Q, A, Z_chr, k_chr)
        else:
            Z_miss = np.fromfile(m_file, dtype=np.uint8)
            m_size = W_chr * 2 * N_all
            if Z_miss.shape[0] == (m_size + 3):
                assert np.allclose(
                    Z_miss[:3], np.array([7, 9, 14], dtype=np.uint8)
                ), "Magic number doesn't match missingness mask format!"
                Z_miss = Z_miss[3:]
            assert Z_miss.shape[0] == m_size, (
                "Missingness mask doesn't match cluster assignments!"
            )
            Z_miss.shape = (W_chr, 2 * N_all)
            if hap_idx is not None:
                Z_miss = Z_miss[:, hap_idx]
            Z_miss = np.ascontiguousarray((Z_miss > 0).astype(np.uint8))
            H_obs = (Z_miss.reshape(W_chr, N, 2) == 0).sum(axis=2)
            A_chr = np.zeros((W_chr, Q.shape[1], Q.shape[1]), dtype=np.float64)
            for w in range(W_chr):
                D = Q * H_obs[w, :, None]
                A_chr[w] = np.linalg.pinv(np.dot(D.T, D))
            A_chr = np.ascontiguousarray(A_chr)
            eval_cy.covarMiss(C, V, Q, A_chr, Z_chr, Z_miss, k_chr)
            del Z_miss, H_obs, A_chr

        if F > 1:
            # Print elapsed time of chromosome
            t_tmp = time() - t_chr
            t_min = int(t_tmp // 60)
            t_sec = int(t_tmp - t_min * 60)
            print(f"Elapsed time: {t_min}m{t_sec}s\n")
    # Estimate model-expected covariance of residuals and convert into correlations
    C_exp = np.zeros_like(C)
    if has_missing:
        w_cnt = 0
        for z in np.arange(F):
            W_chr = w_vec[z]
            k_chr = k_vec[w_cnt : (w_cnt + W_chr)]
            w_cnt += W_chr
            m_file = miss_files[z]
            with open(f"{Z_list[z]}.bca", "rb") as f:
                magic = np.fromfile(f, dtype=np.uint8, count=3)
                assert np.allclose(magic, np.array([7, 9, 13], dtype=np.uint8)), (
                    "Magic number doesn't match file format!"
                )
                Z_chr = np.fromfile(f, dtype=np.uint8)
                Z_chr.shape = (W_chr, 2 * N_all)
                if hap_idx is not None:
                    Z_chr = Z_chr[:, hap_idx]
                Z_chr = np.ascontiguousarray(Z_chr)
            if m_file is None:
                Z_miss = np.zeros((W_chr, 2 * N), dtype=np.uint8)
            else:
                Z_miss = np.fromfile(m_file, dtype=np.uint8)
                m_size = W_chr * 2 * N_all
                if Z_miss.shape[0] == (m_size + 3):
                    assert np.allclose(
                        Z_miss[:3], np.array([7, 9, 14], dtype=np.uint8)
                    ), "Magic number doesn't match missingness mask format!"
                    Z_miss = Z_miss[3:]
                Z_miss.shape = (W_chr, 2 * N_all)
                if hap_idx is not None:
                    Z_miss = Z_miss[:, hap_idx]
                Z_miss = np.ascontiguousarray((Z_miss > 0).astype(np.uint8))
            H_obs = (Z_miss.reshape(W_chr, N, 2) == 0).sum(axis=2)
            A_chr = np.zeros((W_chr, Q.shape[1], Q.shape[1]), dtype=np.float64)
            for w in range(W_chr):
                D = Q * H_obs[w, :, None]
                A_chr[w] = np.linalg.pinv(np.dot(D.T, D))
            A_chr = np.ascontiguousarray(A_chr)
            eval_cy.expectedMiss(C_exp, Q, A_chr, Z_chr, Z_miss, k_chr)
            del Z_chr, Z_miss, H_obs, A_chr
    else:
        S = np.dot(Q.T, V.reshape(-1, 1) * Q)
        B = np.ascontiguousarray(np.dot(A, np.dot(S, A)))
        QA = np.ascontiguousarray(np.dot(Q, A))
        QB = np.ascontiguousarray(np.dot(Q, B))
        eval_cy.expected(C_exp, Q, QA, QB, V)
    b_hat = np.zeros_like(C)
    c_hat = np.zeros_like(C)
    eval_cy.corr(C, b_hat)
    eval_cy.corr(C_exp, c_hat)
    cor = b_hat - c_hat

    # Write correlations to files
    np.savetxt(f"{args.out}.bhat", b_hat, fmt="%.4f")
    np.savetxt(f"{args.out}.chat", c_hat, fmt="%.4f")
    np.savetxt(f"{args.out}.corres", cor, fmt="%.4f")

    # Clean
    del C, C_exp, b_hat, c_hat, cor

    # Print elapsed time for computation
    t_tot = time() - start
    t_min = int(t_tot // 60)
    t_sec = int(t_tot - t_min * 60)
    print(f"Total elapsed time: {t_min}m{t_sec}s")

    # Write to log-file
    with open(f"{args.out}.log", "a") as log:
        log.write(f"\nSaved empirical correlations of residuals as {args.out}.bhat\n")
        log.write(f"Saved model-expected correlations of residuals as {args.out}.chat\n")
        log.write(f"Saved corrected correlations of residuals as {args.out}.corres\n")
        log.write(f"\nTotal elapsed time: {t_min}m{t_sec}s\n")


##### Main exception #####
assert __name__ != "__main__", "Please use the 'hapla eval' command!"
