"""Independent leave-pair-out counts, HMM updates, and command contracts."""

__author__ = "Jonas Meisner"

from argparse import Namespace
from contextlib import redirect_stdout
from io import StringIO
from itertools import product
from unittest.mock import patch

import numpy as np
from helpers import TemporaryTests, command, readLog, writeClusters
from test_admix import simplex
from test_fatash import enumerateHMM

from hapla import admix_cy, fatash, fatash_cy, functions


### Delete donor observations explicitly rather than subtracting their counts
class LeaveOneOutTests(TemporaryTests):
    def test_admix_matches_deleted_individual_reference_and_sample_tiles(self):
        rng = np.random.default_rng(617)
        Z = np.array(
            [[255, 255, 2, 2, 0, 1, 0, 1], [255] * 8, [255, 255, 0, 255, 1, 1, 0, 1]],
            dtype=np.uint8,
        )
        W, N = len(Z), Z.shape[1] // 2
        obs = (Z != 255).sum(axis=1).astype(np.uint32)
        qo = (Z != 255).reshape(W, N, 2).sum(axis=(0, 2), dtype=np.uint32)
        for C, K in product((3, 255), (2, 5, 6)):
            pool = np.array(
                [
                    np.bincount(z[z != 255], minlength=C) / n if n else np.zeros(C)
                    for z, n in zip(Z, obs)
                ]
            )
            P = rng.dirichlet(np.ones(C), (W, K)).transpose(0, 2, 1).copy()
            Q = rng.dirichlet(np.ones(K), N)
            k = np.full(W, C, np.uint32)
            c = np.arange(W + 1, dtype=np.uint32) * C * K
            for mass in (0, 2, 1e308):
                expected = P.copy()
                qcounts = np.zeros_like(Q)
                for w, z in enumerate(Z):
                    counts = np.zeros((2 * N, C, K))
                    for h in np.flatnonzero(z != 255):
                        x = P[w, z[h]] * Q[h // 2]
                        counts[h, z[h]] = x / x.sum()
                    if obs[w]:
                        expected[w] = simplex(counts.sum(axis=0) + mass * pool[w, :, None])
                    for i in range(N):
                        donors = np.arange(2 * N) // 2 != i
                        seen = z[donors & (z != 255)]
                        f = np.bincount(seen, minlength=C).astype(float)
                        f = f / f.sum() if len(seen) else np.full(C, 1 / C)
                        p = simplex(counts[donors].sum(axis=0) + mass * f[:, None])
                        for label in z[2 * i : 2 * i + 2]:
                            if label == 255:
                                continue
                            x = Q[i] * p[label] if label in seen else Q[i]
                            qcounts[i] += x / x.sum()
                expected_q = Q.copy()
                expected_q[qo > 0] = simplex(qcounts[qo > 0].T).T
                fits = []
                for tile in (1, N):
                    ctx = (
                        Z,
                        k,
                        c,
                        np.empty_like(Q),
                        np.empty((W, C * K)),
                        np.empty((W, tile, K)),
                        obs,
                        None,
                    )
                    A, B = np.empty(P.size), np.empty_like(Q)
                    functions.looStep(P.ravel(), Q, A, B, ctx, pool.ravel(), mass, qo)
                    np.testing.assert_allclose(A.reshape(P.shape), expected, atol=3e-14)
                    np.testing.assert_allclose(B, expected_q, atol=3e-14)
                    np.testing.assert_array_equal(B[0], Q[0])
                    fits.append((A, B))
                for a, b in zip(*fits):
                    np.testing.assert_array_equal(a, b)

    def test_hmm_q_update_matches_deleted_individual_enumeration(self):
        rng = np.random.default_rng(914)
        K, N, W, C = 3, 4, 4, 3
        Z = rng.integers(0, 2, (W, 2 * N), dtype=np.uint8)
        Z[:, :2] = 255
        Z[0, 2:4] = 2
        Z[2, 3] = 255
        use = np.array([1, 0, 1, 1], np.uint8)
        c = np.arange(W + 1, dtype=np.int64) * C
        regions = [(0, 2), (2, W)]
        P = rng.dirichlet(np.ones(C), (W, K)).transpose(0, 2, 1).copy()
        Q = rng.dirichlet(np.ones(K), N)
        alpha = np.array([0.03, 0.2])
        donors = np.zeros((N, W, C, K))
        for h in range(2 * N):
            for beg, end in regions:
                E = np.array(
                    [
                        np.log(P[w, Z[w, h]]) if use[w] and Z[w, h] != 255 else np.zeros(K)
                        for w in range(beg, end)
                    ]
                )
                G = np.mean([enumerateHMM(E, Q[h // 2], a)[0] for a in alpha], axis=0)
                for w in range(beg, end):
                    if use[w] and Z[w, h] != 255:
                        donors[h // 2, w, Z[w, h]] += G[w - beg]
        counts = donors.sum(axis=0)
        for bw, mass in ((True, 0), (False, 2)):
            cq = np.zeros_like(Q)
            for i in range(N):
                other = donors[np.arange(N) != i].sum(axis=0)
                p = other + mass * P
                total = p.sum(axis=1, keepdims=True)
                for w in range(W):
                    hist = np.bincount(
                        Z[w, (np.arange(2 * N) // 2 != i) & (Z[w] != 255)], minlength=C
                    ).astype(float)
                    hist = hist / hist.sum() if hist.sum() else np.full(C, 1 / C)
                    p[w] = np.divide(
                        p[w],
                        total[w],
                        out=np.broadcast_to(hist[:, None], (C, K)).copy(),
                        where=total[w] > 0,
                    )
                for h in (2 * i, 2 * i + 1):
                    for beg, end in regions:
                        E = np.zeros((end - beg, K))
                        for w in range(beg, end):
                            label = Z[w, h]
                            if use[w] and label != 255 and np.dot(Q[i], p[w, label]) > 0:
                                with np.errstate(divide="ignore"):
                                    E[w - beg] = np.log(p[w, label])
                        cq[i] += np.mean([enumerateHMM(E, Q[i], a)[1] for a in alpha], axis=0)
            q = cq + mass * Q
            q /= q.sum(axis=1, keepdims=True)
            q[0] = Q[0]
            expected_q = 0.5 * (Q + q)
            p = counts + mass * P
            total = p.sum(axis=1, keepdims=True)
            p = np.divide(p, total, out=P.copy(), where=total > 0)
            expected_p = 0.5 * (P + p)
            args = Namespace(baum_welch=bw, p_prior=mass, q_prior=mass, iter=1, tole=0, loo=True)
            with redirect_stdout(StringIO()):
                pp, qq, info = fatash.refine([(Z, c, use, regions, 4)], [P.ravel()], Q, alpha, args)
            np.testing.assert_allclose(pp[0].reshape(P.shape), expected_p, atol=3e-13)
            np.testing.assert_allclose(qq, expected_q, atol=3e-13)
            self.assertEqual(info["iterations"], 1)
            self.assertTrue(info["loo"])
            self.assertEqual(info["convergence"], "parameter RMSE")

    def test_hmm_loo_accepts_a_likelihood_decrease(self):
        Z = np.array([[0, 1, 0, 0], [1, 0, 1, 1]], np.uint8)
        c = np.array([0, 2, 4], np.int64)
        P = np.array([0.8, 0.2, 0.2, 0.8] * 2)
        Q = np.array([[0.7, 0.3], [0.4, 0.6]])
        args = Namespace(baum_welch=True, p_prior=0, q_prior=0, iter=1, tole=0, loo=True)
        posterior = fatash_cy.posterior
        calls = [0]

        def lower(*args, **kwargs):
            G, C, L = posterior(*args, **kwargs)
            calls[0] += 1
            return G, C, L - 1000 * calls[0]

        with patch.object(fatash_cy, "posterior", side_effect=lower), redirect_stdout(StringIO()):
            _, _, info = fatash.refine(
                [(Z, c, np.ones(2, np.uint8), [(0, 2)], 4)], [P], Q, [0.1], args
            )
        self.assertEqual(info["iterations"], 1)
        self.assertEqual(info["stop"], "iteration_limit")
        self.assertLess(info["history"][1]["objective"], info["history"][0]["objective"])

    def test_admix_loo_accepts_a_likelihood_decrease(self):
        from hapla.main import main

        Z = np.array([[0, 1, 0, 0], [1, 0, 1, 1]], np.uint8)
        ref = writeClusters(self.root, "input", Z, np.array([0, 2, 4]))
        out = self.root / "fit"
        argv = [
            "hapla",
            "admix",
            "--clusters",
            str(ref),
            "--K",
            "2",
            "--random-init",
            "--loo",
            "--iter",
            "1",
            "--out",
            str(out),
        ]
        with (
            patch("sys.argv", argv),
            redirect_stdout(StringIO()),
            patch.object(admix_cy, "likelihood", side_effect=[-1.0, -2.0]),
        ):
            main()
        log = readLog(f"{out}.K2.s42")
        self.assertEqual(log["Iterations"], "1")
        self.assertNotIn("Recoveries", log)
        self.assertNotIn("Stalled", log)
        self.assertLess(float(log["Final log-like"]), float(log["Initial log-like"]))

    def test_single_sample_zero_ancestry_and_empty_hmm(self):
        Q = np.array([[1.0, 0.0]])
        P = np.array([1.0, 0.0, 0.0, 1.0] * 2)
        c = np.array([0, 2, 4], np.int64)
        args = Namespace(baum_welch=True, p_prior=0, q_prior=0, iter=2, tole=0, loo=True)
        for Z in (np.zeros((2, 2), np.uint8), np.full((2, 2), 255, np.uint8)):
            with redirect_stdout(StringIO()):
                _, q, info = fatash.refine(
                    [(Z, c, np.ones(2, np.uint8), [(0, 2)], 2)], [P], Q, [0.01, 0.1], args
                )
            np.testing.assert_array_equal(q, Q)
            self.assertEqual(info["stop"], "no_observations" if Z[0, 0] == 255 else "converged")

    def test_flags_threads_missingness_and_shared_final_decoding(self):
        rng = np.random.default_rng(514)
        W, N, C, K = 1024, 12, 3, 5
        Z = rng.integers(C, size=(W, 2 * N), dtype=np.uint8)
        Z[:, :2] = 255
        Z[3] = 255
        Z[::5, 7] = 255
        refs = [
            writeClusters(self.root, f"chr{j}", z, np.arange(len(z) + 1) * C)
            for j, z in enumerate(np.split(Z, 2), 1)
        ]
        models = []
        for threads in (1, 4):
            out = self.root / f"adm{threads}"
            command(
                "admix",
                "--clusters",
                *refs,
                "--K",
                K,
                "--random-init",
                "--loo",
                "--p-prior",
                2,
                "--iter",
                3,
                "--threads",
                threads,
                "--out",
                out,
            )
            pfx = f"{out}.K{K}.s42"
            models.append([np.loadtxt(pfx + s) for s in (".Q", ".chr1.P", ".chr2.P")])
            log = readLog(pfx)
            self.assertEqual(log["LOO"], "yes")
            self.assertEqual(log["Convergence"], "parameter RMSE")
        for a, b in zip(*models):
            np.testing.assert_array_equal(a, b)
        qfile = f"{self.root}/adm1.K{K}.s42.Q"
        pfiles = [f"{self.root}/adm1.K{K}.s42.chr{j}.P" for j in (1, 2)]
        fits = []
        for threads in (1, 4):
            out = self.root / f"hmm{threads}"
            command(
                "fatash",
                "--clusters",
                *refs,
                "--qfile",
                qfile,
                "--pfile",
                *pfiles,
                "--loo",
                "--iter",
                2,
                "--tole",
                0,
                "--threads",
                threads,
                "--buffer-mb",
                256 if threads == 1 else 1,
                "--save-posteriors",
                "--out",
                out,
            )
            fits.append(
                [
                    np.loadtxt(str(out) + s)
                    for s in (
                        ".Q",
                        ".chr1.P",
                        ".chr2.P",
                        ".chr1.path",
                        ".chr2.path",
                        ".chr1.prob",
                        ".chr2.prob",
                    )
                ]
            )
            np.testing.assert_allclose(fits[-1][0][0], models[0][0][0], atol=2e-10)
        for a, b in zip(*fits):
            np.testing.assert_array_equal(a, b)
        out = self.root / "reuse"
        command(
            "fatash",
            "--clusters",
            *refs,
            "--qfile",
            self.root / "hmm1.Q",
            "--pfile",
            self.root / "hmm1.chr1.P",
            self.root / "hmm1.chr2.P",
            "--fixed-model",
            "--save-posteriors",
            "--out",
            out,
        )
        for s, a in zip((".chr1.path", ".chr2.path", ".chr1.prob", ".chr2.prob"), fits[0][3:]):
            np.testing.assert_allclose(np.loadtxt(str(out) + s), a, atol=1e-8)
        for mode, opts in (
            ("admix", ["--K", K, "--projection", pfiles[0]]),
            ("fatash", ["--qfile", qfile, "--pfile", pfiles[0], "--fixed-model"]),
        ):
            res = command(mode, "--clusters", refs[0], "--loo", *opts, success=False)
            self.assertIn("--loo requires", res.stderr)
        for mode, opts in (
            ("admix", ["--K", K, "--random-init"]),
            ("fatash", ["--qfile", qfile, "--pfile", *pfiles]),
        ):
            out = self.root / f"stop-{mode}"
            command(
                mode, "--clusters", *refs, "--loo", "--tole", 1, "--iter", 3, "--out", out, *opts
            )
            log = readLog(f"{out}.K{K}.s42" if mode == "admix" else out)
            self.assertEqual(log["Iterations"], "1")
