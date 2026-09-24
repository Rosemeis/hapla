"""Independent categorical-EM references and ancestry command regressions."""

__author__ = "Jonas Meisner"

import sys
from argparse import Namespace
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import numpy as np
from helpers import TemporaryTests, command, missingFixture, readLog, writeClusters

from hapla import admix_cy as cy
from hapla import functions


### Solve the constrained M-step by independent bisection
def simplex(A):
    A = np.maximum(A, 0.0)
    out = np.empty_like(A)
    C = len(A)
    for k in range(A.shape[1]):
        x = A[:, k]
        if C == 1 or x.sum() == 0:
            out[:, k] = 1 / C
            continue
        lo, hi = x.max(), x.sum() / (1 - C * 1e-5)
        for _ in range(80):
            mid = lo + (hi - lo) / 2
            if np.maximum(1e-5, x / mid).sum() > 1:
                lo = mid
            else:
                hi = mid
        out[:, k] = np.maximum(1e-5, x / hi)
    return out


### Build deterministic categorical mixtures with optional missingness
def fixture(K=5, missing=False):
    rng = np.random.default_rng(891)
    k = np.array([1, 2, 7, 255, 4, 3], dtype=np.uint32)
    c = np.r_[0, np.cumsum(k * K)].astype(np.uint32)
    Z = np.array([rng.integers(0, C, 38) for C in k], dtype=np.uint8)
    Z[:, 1::6] = Z[:, ::6]
    Z[3, 2] = 254
    P = np.concatenate([rng.dirichlet(np.ones(C), size=K).T for C in k]).ravel()
    Q = rng.dirichlet(np.ones(K), size=19)
    mask = np.zeros_like(Z) if missing else None
    if missing:
        mask[:, :2] = 1
        mask[2, :] = 1
        mask[4, 7] = 1
        Z[mask != 0] = 255
    wo = None if mask is None else (mask == 0).sum(axis=1, dtype=np.uint32)
    T = np.empty_like(Q)
    pt = np.full((6, int(k.max()) * K), np.nan)
    qt = np.full((6, len(Q), K), np.nan)
    return P, Q, (Z, k, c, T, pt, qt, wo, None)


### Compute dense EM updates and the observed-data likelihood
def reference(P, Q, ctx, rows=None):
    Z, k, c, _, _, _, _, _ = ctx
    rows = range(len(Z)) if rows is None else rows
    K = Q.shape[1]
    A, B, qo = np.zeros_like(P).reshape(-1, K), np.zeros_like(Q), np.zeros(len(Q))
    out = P.copy().reshape(-1, K)
    ll = 0.0
    for w in rows:
        beg, end = int(c[w] // K), int(c[w + 1] // K)
        obs = 0
        for h, z in enumerate(Z[w]):
            if z == 255:
                continue
            i, r = h // 2, beg + int(z)
            prob = P.reshape(-1, K)[r] * Q[i]
            ll += np.log(prob.sum())
            posterior = prob / prob.sum()
            A[r] += posterior
            B[i] += posterior
            qo[i] += 1
            obs += 1
        if obs:
            out[beg:end] = simplex(A[beg:end])
    q = Q.copy()
    seen = qo > 0
    q[seen] = simplex(B[seen].T).T
    return out.ravel(), q, qo.astype(np.uint32), ll / (2 * len(Q) * len(A))


### Check ancestry updates, initialization, and command output
class AdmixtureTests(TemporaryTests):
    def test_em_workspace_is_bounded_for_large_cohorts(self):
        k = np.full(129, 255, dtype=np.uint32)
        for N, K in ((500_000, 10), (1_000_000, 20)):
            pt, qt = functions.emWorkspace(N, K, k)
            self.assertEqual(qt.dtype, np.float64)
            self.assertEqual(qt.shape[0], 64)
            self.assertEqual(qt.shape[2], K)
            self.assertLess(qt.shape[1], N)
            self.assertLessEqual(qt.nbytes, 8 * 1024**2)
            self.assertEqual(pt.shape, (64, 255 * K))

    def test_sample_tiles_preserve_em_results_and_reduction_order(self):
        rng = np.random.default_rng(952)
        N, W = 79, 67
        k = np.resize(np.array([1, 2, 0, 7, 4, 3, 5], np.uint32), W)
        for K in (3, 5, 6):
            P = np.concatenate([rng.dirichlet(np.ones(C), size=K).T.ravel() for C in k if C])
            Q = rng.dirichlet(np.ones(K), size=N)
            c = np.r_[0, np.cumsum(k * K)].astype(np.uint32)
            for missing in (False, True):
                Z = np.array(
                    [rng.integers(C, size=2 * N) if C else np.full(2 * N, 255) for C in k], np.uint8
                )
                if missing:
                    Z[:, :2] = 255
                    Z[5] = 255
                    Z[::4, 17] = 255
                obs = (Z != 255).sum(axis=1, dtype=np.uint32)
                pool = np.concatenate(
                    [
                        np.bincount(z[z != 255], minlength=C) / n if n else np.zeros(C)
                        for z, C, n in zip(Z, k, obs)
                    ]
                )
                for rows in (None, np.array([66, 0, 5, 2, 1, 32, 17], np.uint32)):
                    for mass in (0.0, 0.75):
                        for mode in ("update", "inplace", "projection"):
                            fits = []
                            for tile in (N, 1, 32, N - 1):
                                with self.subTest(
                                    K=K, missing=missing, mass=mass, mode=mode, tile=tile
                                ):
                                    p = P.copy()
                                    out = P.copy()
                                    if mode == "inplace":
                                        out = p
                                    elif mode == "projection":
                                        out = None
                                    T = np.full_like(Q, np.nan)
                                    pt = np.full((64, int(k.max()) * K), np.nan)
                                    qt = np.full((64, tile, K), np.nan)
                                    cy.em(
                                        Z,
                                        p,
                                        out,
                                        Q,
                                        T,
                                        k,
                                        c,
                                        pt,
                                        qt,
                                        rows,
                                        obs if missing else None,
                                        pool,
                                        mass,
                                    )
                                    fit = (p if out is None else out, T)
                                    if fits:
                                        for actual, expected in zip(fit, fits[0]):
                                            np.testing.assert_array_equal(actual, expected)
                                    fits.append(fit)

    def test_tiled_quasi_newton_reuses_spare_p_and_preserves_checkpoint(self):
        fits = []
        for tile in (19, 7):
            P, Q, ctx = fixture(5, missing=True)
            ctx = (*ctx[:5], ctx[5][:, :tile].copy(), *ctx[6:])
            P1, P2 = np.empty_like(P), np.empty_like(P)
            Q1, Q2 = np.empty_like(Q), np.empty_like(Q)
            qo = cy.observedCounts(ctx[0])
            functions.emQuasi(P, Q, P1, P2, Q1, Q2, ctx, qo=qo, scratch=P2)
            P1[:] = P
            saved = P1.copy()
            functions.emStep(P, Q, P, Q, ctx, qo=qo, scratch=P2)
            np.testing.assert_array_equal(P1, saved)
            fits.append((P, Q))
        for actual, expected in zip(fits[1], fits[0]):
            np.testing.assert_array_equal(actual, expected)

    def test_tiled_inplace_batch_preserves_unselected_windows(self):
        fits = []
        rows = np.array([5, 2, 0], np.uint32)
        for tile in (19, 7):
            P, Q, ctx = fixture(5, missing=True)
            ctx = (*ctx[:5], ctx[5][:, :tile].copy(), *ctx[6:])
            saved = P.copy()
            qo = cy.observedCounts(ctx[0], rows)
            functions.emStep(P, Q, P, Q, ctx, rows, qo, scratch=np.full_like(P, np.nan))
            for w in (1, 3, 4):
                a, b = ctx[2][w : w + 2]
                np.testing.assert_array_equal(P[a:b], saved[a:b])
            fits.append((P, Q))
        for actual, expected in zip(fits[1], fits[0]):
            np.testing.assert_array_equal(actual, expected)

    def test_observed_counts_cover_sample_tiles_and_window_subsets(self):
        rng = np.random.default_rng(406)
        Z = rng.integers(0, 255, (17, 2 * 513), dtype=np.uint8)
        Z[::3, ::7] = 255
        Z[:, :2] = 255
        Z[5] = 255
        for rows in (None, np.array([16, 5, 2, 0], np.uint32), np.empty(0, np.uint32)):
            z = Z if rows is None else Z[rows]
            expected = (z != 255).reshape(len(z), 513, 2).sum(axis=(0, 2), dtype=np.uint32)
            np.testing.assert_array_equal(cy.observedCounts(Z, rows), expected)

    def test_rare_ancestry_mstep_and_likelihood_ascent_at_boundaries(self):
        for K in (3, 5, 6):
            P, Q, ctx = fixture(K, missing=True)
            Z, k, c = ctx[:3]
            for beg, end in zip(c[:-1] // K, c[1:] // K):
                P.reshape(-1, K)[beg:end] = simplex(P.reshape(-1, K)[beg:end])
            Q[:] = 1e-5
            Q[:, 0] = 1 - (K - 1) * 1e-5
            qo = (Z != 255).reshape(len(Z), len(Q), 2).sum(axis=(0, 2), dtype=np.uint32)
            old = cy.likelihood(Z, P, Q, c, np.empty(len(Z)), ctx[6])
            for _ in range(8):
                functions.emStep(P, Q, P, Q, ctx, qo=qo)
                now = cy.likelihood(Z, P, Q, c, np.empty(len(Z)), ctx[6])
                self.assertGreaterEqual(now, old - 2e-15)
                self.assertGreaterEqual(P.min(), 1e-5)
                self.assertGreaterEqual(Q.min(), 1e-5)
                old = now

        # A minor ancestry must retain its supported allele
        N, K = 1000, 5
        Z, k, c = (
            np.zeros((1, 2 * N), np.uint8),
            np.array([2], np.uint32),
            np.array([0, 10], np.uint32),
        )
        P = np.tile([[1 - 1e-5], [1e-5]], (1, K)).ravel()
        Q = np.tile([1 - 4e-5, 1e-5, 1e-5, 1e-5, 1e-5], (N, 1))
        ctx = (Z, k, c, np.empty_like(Q), np.empty((1, 10)), np.empty((1, N, K)), None, None)
        A = P.copy()
        functions.emStep(P, Q, A, Q, ctx)
        np.testing.assert_allclose(A, P, atol=3e-15)

    def test_missing_svd_mean_imputation_and_small_cohorts(self):
        Z, c, p, D, obs = missingFixture()
        U, S, V = functions.centerSVD(
            Z, p, c.astype(np.uint32), len(Z), 5, 7, 2, np.random.default_rng(9), obs
        )
        X = D - 2 * p[:, None]
        _, exact, R = np.linalg.svd(X, full_matrices=False)
        np.testing.assert_allclose(S, exact[:4], rtol=2e-6)
        np.testing.assert_allclose(V @ V.T, R[:4].T @ R[:4], atol=2e-6)
        np.testing.assert_allclose(U * S, X @ V, atol=2e-6)
        for N in (8, 19):
            Z = np.random.default_rng(219).integers(0, 4, (60, 2 * N), dtype=np.uint8)
            Z[0, 0] = 255
            ref = writeClusters(self.root, f"small{N}", Z, np.arange(61) * 4)
            for K in (5, 6):
                out = self.root / f"fit{N}_{K}"
                command("admix", "--clusters", ref, "--K", K, "--iter", 2, "--out", out)
                self.assertTrue(np.isfinite(np.loadtxt(f"{out}.K{K}.s42.Q")).all())

    def test_empty_windows_unobserved_samples_and_missing_projection(self):
        Z, c, _, _, _ = missingFixture()
        ref = writeClusters(self.root, "input", Z, c)
        out = self.root / "fit"
        command("admix", "--clusters", ref, "--K", 5, "--iter", 3, "--out", out)
        Q = np.loadtxt(f"{out}.K5.s42.Q")
        np.testing.assert_allclose(Q[0], 0.2, atol=2e-10)
        command(
            "admix",
            "--clusters",
            ref,
            "--K",
            5,
            "--projection",
            f"{out}.K5.s42.P",
            "--iter",
            3,
            "--out",
            self.root / "project",
        )
        self.assertTrue(np.isfinite(np.loadtxt(self.root / "project.project.K5.s42.Q")).all())
        empty = writeClusters(self.root, "empty", np.full((2, 6), 255, np.uint8), np.zeros(3, int))
        res = command("admix", "--clusters", empty, "--K", 5, "--out", out, success=False)
        self.assertIn("No observed", res.stderr)

    def test_p_prior_changes_fit_and_reports_regularized_objective(self):
        _, _, ctx = fixture(5, missing=True)
        ref = writeClusters(self.root, "prior", ctx[0], ctx[2] // 5)
        fits = []
        for mass in (0, 2):
            out = self.root / f"prior{mass}"
            opts = ("--p-prior", mass) if mass else ()
            command(
                "admix",
                "--clusters",
                ref,
                "--K",
                5,
                "--random-init",
                "--batches",
                1,
                "--iter",
                2,
                *opts,
                "--out",
                out,
            )
            pfx = f"{out}.K5.s42"
            fits.append(np.loadtxt(f"{pfx}.P"))
            log = readLog(pfx)
            if mass:
                self.assertLess(float(log["Final objective"]), float(log["Final log-like"]))
                self.assertEqual(float(log["P prior"]), mass)
            else:
                self.assertNotIn("Final objective", log)
                self.assertNotIn("P prior", log)
        self.assertGreater(np.max(np.abs(fits[0] - fits[1])), 1e-6)

    def test_source_initializer_is_opt_in(self):
        _, _, ctx = fixture(5)
        ref = writeClusters(self.root, "source", ctx[0], ctx[2] // 5)
        for K, source in ((5, False), (5, True), (6, True)):
            out = self.root / f"source{K}_{source}"
            opts = ("--source-init",) if source else ()
            res = command(
                "admix",
                "--clusters",
                ref,
                "--K",
                K,
                "--power",
                2,
                "--als-iter",
                2,
                "--iter",
                1,
                *opts,
                "--out",
                out,
            )
            text = "Computing source estimates." if source else "Computing SVD/ALS estimates."
            self.assertIn(text, res.stdout)
            self.assertTrue(np.isfinite(np.loadtxt(f"{out}.K{K}.s42.Q")).all())
        for opts in (("--random-init",), ("--supervised", "sources"), ("--projection", "model.P")):
            res = command(
                "admix", "--clusters", ref, "--K", 5, "--source-init", *opts, success=False
            )
            self.assertIn("--source-init requires", res.stderr)

    def test_source_and_prior_across_chromosomes_with_missingness(self):
        rng = np.random.default_rng(617)
        K, C, W, N, mass = 6, 4, 48, 48, 0.5
        Z = rng.integers(0, C, (W, 2 * N), dtype=np.uint8)
        Z[:, :2] = 255
        Z[5] = 255
        Z[::4, 7] = 255
        refs = [
            writeClusters(self.root, f"chr{j}", z, np.arange(len(z) + 1) * C)
            for j, z in enumerate(np.split(Z, 2), 1)
        ]
        fits = []
        for nt in (1, 4):
            out = self.root / f"joint{nt}"
            command(
                "admix",
                "--clusters",
                *refs,
                "--K",
                K,
                "--source-init",
                "--p-prior",
                mass,
                "--threads",
                nt,
                "--power",
                2,
                "--als-iter",
                4,
                "--iter",
                4,
                "--check",
                1,
                "--batches",
                3,
                "--out",
                out,
            )
            pfx = f"{out}.K{K}.s42"
            Q = np.loadtxt(f"{pfx}.Q")
            P = np.concatenate([np.loadtxt(f"{pfx}.chr{j}.P") for j in (1, 2)])
            P = P.reshape(W, C, K)
            np.testing.assert_allclose(Q[0], 1 / K, atol=1e-10)
            np.testing.assert_allclose(Q.sum(axis=1), 1, atol=1e-9)
            np.testing.assert_allclose(P.sum(axis=1), 1, atol=1e-9)
            ll, penalty = 0.0, 0.0
            for z, p in zip(Z, P):
                h = np.flatnonzero(z != 255)
                if len(h):
                    ll += np.log((p[z[h]] * Q[h // 2]).sum(axis=1)).sum()
                    f = np.bincount(z[h], minlength=C) / len(h)
                    seen = f > 0
                    penalty += mass * np.sum(f[seen, None] * np.log(p[seen] / f[seen, None]))
            log = readLog(pfx)
            self.assertAlmostEqual(float(log["Final log-like"]), ll, places=5)
            self.assertAlmostEqual(float(log["Final objective"]), ll + penalty, places=5)
            fits.append((P, Q))
        for a, b in zip(*fits):
            np.testing.assert_array_equal(a, b)

    def test_prior_accepts_likelihood_loss_when_objective_improves(self):
        from hapla.main import main

        _, _, ctx = fixture(5)
        ref = writeClusters(self.root, "map", ctx[0], ctx[2] // 5)
        out = self.root / "fit"
        scale = 2 * (ctx[0].shape[1] // 2) * int(ctx[1].sum())
        argv = [
            "hapla",
            "admix",
            "--clusters",
            str(ref),
            "--K",
            "5",
            "--random-init",
            "--p-prior",
            "1",
            "--iter",
            "1",
            "--batches",
            "1",
            "--out",
            str(out),
        ]
        with (
            patch.object(sys, "argv", argv),
            patch.object(cy, "likelihood", side_effect=[-100.0, -90.0, -91.0]),
            patch.object(cy, "priorScore", side_effect=[-20.0 * scale, -10.0 * scale, -scale]),
            redirect_stdout(StringIO()),
        ):
            main()
        log = readLog(f"{out}.K5.s42")
        self.assertNotIn("Recoveries", log)
        self.assertEqual(float(log["Final log-like"]), -91.0 * scale)
        self.assertEqual(float(log["Final objective"]), -92.0 * scale)

    def test_projection_normalizes_accepted_reference_probabilities(self):
        ref = writeClusters(self.root, "single", np.zeros((1, 200), np.uint8), np.array([0, 1]))
        pfile = self.root / "rounded.P"
        np.savetxt(pfile, np.full((1, 5), 1.0005))
        out = self.root / "project"
        command(
            "admix",
            "--clusters",
            ref,
            "--K",
            5,
            "--projection",
            pfile,
            "--iter",
            1,
            "--batches",
            1,
            "--out",
            out,
        )
        log = readLog(f"{out}.project.K5.s42")
        self.assertAlmostEqual(float(log["Initial log-like"]), 0)
        self.assertAlmostEqual(float(log["Final log-like"]), 0)

    def test_em_matches_dense_full_batch_missing_and_projection(self):
        for K in (3, 5, 6):
            for missing in (False, True):
                P, Q, ctx = fixture(K, missing)
                for rows in (None, np.array([5, 0, 3, 2], np.uint32)):
                    expected, q, qo, _ = reference(P, Q, ctx, rows)
                    A, B = P.copy(), np.empty_like(Q)
                    functions.emStep(P, Q, A, B, ctx, rows, qo if missing else None)
                    np.testing.assert_allclose(A, expected, atol=3e-15)
                    np.testing.assert_allclose(B, q, atol=3e-15)
                    C = np.empty_like(Q)
                    functions.emStep(P, Q, None, C, ctx, rows, qo if missing else None)
                    np.testing.assert_allclose(C, q, atol=3e-15)
                    A, B = P.copy(), Q.copy()
                    functions.emStep(A, B, A, B, ctx, rows, qo if missing else None)
                    np.testing.assert_allclose(A, expected, atol=3e-15)
                    np.testing.assert_allclose(B, q, atol=3e-15)

    def test_likelihood_matches_dense_and_zero_jump_is_finite(self):
        for K in (5, 6):
            for missing in (False, True):
                P, Q, ctx = fixture(K, missing)
                _, _, _, expected = reference(P, Q, ctx)
                score = cy.likelihood(ctx[0], P, Q, ctx[2], np.empty(len(ctx[0])), ctx[6])
                self.assertAlmostEqual(score, expected, places=14)
                A, B = P.copy(), Q.copy()
                cy.jumpP(A, P, P, ctx[1], ctx[2], K)
                cy.jumpQ(B, Q, Q)
                expected = P.copy().reshape(-1, K)
                for beg, end in zip(ctx[2][:-1] // K, ctx[2][1:] // K):
                    expected[beg:end] = simplex(expected[beg:end])
                np.testing.assert_allclose(A, expected.ravel(), atol=5e-16)
                np.testing.assert_allclose(B, Q, atol=5e-16)

    def test_centered_initialization_matches_dense_svd_and_projection(self):
        rng = np.random.default_rng(14)
        k = np.array([2, 3, 4, 3, 5, 2, 3, 4], dtype=np.uint32)
        c = np.r_[0, np.cumsum(k)].astype(np.uint32)
        Z = np.array([rng.integers(0, C, 26) for C in k], dtype=np.uint8)
        D = np.concatenate([np.eye(C)[z.reshape(13, 2)].sum(axis=1).T for C, z in zip(k, Z)])
        p = (D.mean(axis=1) / 2).astype(np.float32)
        X = D - 2 * p[:, None]
        U, S, V = functions.centerSVD(Z, p, c, 5, 5, 8, 2, np.random.default_rng(1))
        _, exact, R = np.linalg.svd(X[: c[5]], full_matrices=False)
        np.testing.assert_allclose(S, exact[:4], rtol=2e-6)
        np.testing.assert_allclose(V @ V.T, R[:4].T @ R[:4], atol=2e-6)
        np.testing.assert_allclose(U * S, X[: c[5]] @ V, atol=2e-6)
        A = functions.centerSub(Z, S, V, p, c, 5, 8)
        np.testing.assert_allclose(A, X[c[5] :] @ (V / S), atol=2e-6)

    def test_short_runs_report_actual_final_likelihood_and_iteration_limit(self):
        P, Q, ctx = fixture(5)
        ref = writeClusters(self.root, "input", ctx[0], ctx[2] // 5)
        out = self.root / "fit"
        command(
            "admix",
            "--clusters",
            ref,
            "--K",
            5,
            "--random-init",
            "--iter",
            1,
            "--check",
            5,
            "--batches",
            1,
            "--out",
            out,
        )
        pfx = f"{out}.K5.s42"
        stats = readLog(pfx)
        self.assertEqual(stats["Iterations"], "1")
        self.assertEqual(stats["Converged"], "no")
        P, Q = np.loadtxt(f"{pfx}.P").ravel(), np.loadtxt(f"{pfx}.Q")
        score = reference(P, Q, ctx)[-1] * (2 * len(Q) * (len(P) // 5))
        self.assertAlmostEqual(float(stats["Final log-like"]), score, places=6)

    def test_progress_times_check_intervals_and_keeps_total_em_time(self):
        from hapla import admix
        from hapla.main import main

        _, _, ctx = fixture(5)
        ref = writeClusters(self.root, "input", ctx[0], ctx[2] // 5)
        out, clock = self.root / "timing", [0.0]
        step = functions.emQuasi

        def advance(*args, **kwargs):
            step(*args, **kwargs)
            clock[0] += 2

        argv = [
            "hapla",
            "admix",
            "--clusters",
            str(ref),
            "--K",
            "5",
            "--random-init",
            "--iter",
            "12",
            "--batches",
            "1",
            "--tole",
            "0",
            "--out",
            str(out),
        ]
        with (
            patch.object(sys, "argv", argv),
            patch.object(admix, "time", side_effect=lambda: clock[0]),
            patch.object(admix, "printTiming") as timing,
            patch.object(functions, "emQuasi", side_effect=advance),
            patch.object(cy, "likelihood", side_effect=[-100.0, -90.0, -80.0, -70.0, -60.0]),
            redirect_stdout(StringIO()),
        ):
            main()
        rows = [call.args for call in timing.call_args_list if call.args[0].startswith("(")]
        self.assertEqual([name.split()[0] for name, _ in rows], ["(5)", "(10)", "(12)"])
        self.assertEqual([sec for _, sec in rows], [10, 10, 4])
        stats = readLog(f"{out}.K5.s42")
        self.assertEqual(stats["Iterations"], "12")
        self.assertEqual(float(stats["EM seconds"]), 24)

    def test_threads_keep_missing_supervision_and_projection(self):
        _, _, ctx = fixture(6, missing=True)
        Z = ctx[0].copy()
        ref = writeClusters(self.root, "input", Z, ctx[2] // 6)
        y = np.zeros(19, dtype=int)
        y[1:7] = np.arange(1, 7)
        labels = self.root / "sources"
        np.savetxt(labels, y, fmt="%d")
        fits = []
        for nt in (1, 4):
            out = self.root / f"fit{nt}"
            command(
                "admix",
                "--clusters",
                ref,
                "--K",
                6,
                "--supervised",
                labels,
                "--threads",
                nt,
                "--iter",
                3,
                "--check",
                1,
                "--batches",
                2,
                "--out",
                out,
            )
            Q = np.loadtxt(f"{out}.K6.s42.Q")
            np.testing.assert_allclose(Q[1:7], np.eye(6) * (1 - 6e-5) + 1e-5, atol=2e-10)
            fits.append(Q)
        np.testing.assert_array_equal(*fits)
        keep = self.root / "keep"
        keep.write_text("s1\ns8\ns18\n")
        command(
            "admix",
            "--clusters",
            ref,
            "--K",
            6,
            "--projection",
            self.root / "fit1.K6.s42.P",
            "--keep",
            keep,
            "--iter",
            2,
            "--batches",
            30,
            "--out",
            self.root / "project",
        )
        q = np.loadtxt(self.root / "project.project.K6.s42.Q")
        self.assertEqual(q.shape, (3, 6))
        np.testing.assert_allclose(q.sum(axis=1), 1, atol=2e-9)

    def test_invalid_reference_preserves_existing_output(self):
        _, _, ctx = fixture(5)
        ref = writeClusters(self.root, "input", ctx[0], ctx[2] // 5)
        model = self.root / "invalid.P"
        np.savetxt(model, np.full((sum(ctx[1]), 5), np.nan))
        out = self.root / "fit"
        Path(f"{out}.project.K5.s42.Q").write_text("previous\n")
        res = command(
            "admix", "--clusters", ref, "--K", 5, "--projection", model, "--out", out, success=False
        )
        self.assertIn("finite", res.stderr)
        self.assertEqual(Path(f"{out}.project.K5.s42.Q").read_text(), "previous\n")

    def test_decreasing_full_updates_recover_or_restore_checkpoint(self):
        from hapla import admix

        _, _, ctx = fixture(5)
        ref = writeClusters(self.root, "input", ctx[0], ctx[2] // 5)
        for stalled, last in ((False, -89.0), (True, -96.0)):
            out = self.root / f"recover{stalled}"
            args = Namespace(
                clusters=str(ref),
                filelist=None,
                K=5,
                threads=1,
                seed=42,
                iter=1,
                tole=1e-9,
                batches=1,
                check=1,
                power=2,
                chunk=8,
                als_iter=10,
                als_tole=1e-4,
                p_prior=0.0,
                subsampling=4,
                random_init=True,
                source_init=False,
                loo=False,
                supervised=None,
                projection=None,
                keep=None,
                out=str(out),
                no_freqs=False,
                prefix="chr",
            )
            checkpoints = []
            step = functions.emStep

            def record(*args, **kwargs):
                step(*args, **kwargs)
                checkpoints.append((args[2].copy(), args[3].copy()))

            with (
                patch.object(cy, "likelihood", side_effect=[-100.0, -90.0, -95.0, last]),
                patch.object(functions, "emStep", side_effect=record),
                redirect_stdout(StringIO()),
            ):
                stats = admix.main(args)
            self.assertEqual(stats["recoveries"], 1)
            self.assertEqual(stats["stalled"], stalled)
            self.assertFalse(stats["converged"])
            self.assertEqual(len(checkpoints), 8)
            expected = checkpoints[3] if stalled else checkpoints[7]
            np.testing.assert_allclose(
                np.loadtxt(f"{out}.K5.s42.P").ravel(), expected[0], atol=5e-11
            )
            np.testing.assert_allclose(np.loadtxt(f"{out}.K5.s42.Q"), expected[1], atol=5e-11)
