"""Independent categorical-EM references and ancestry command regressions."""

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


def simplex(A):
    """Independent Lagrange-multiplier solve, using bisection rather than active sets."""
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
            mid = (lo + hi) / 2
            if np.maximum(1e-5, x / mid).sum() > 1:
                lo = mid
            else:
                hi = mid
        out[:, k] = np.maximum(1e-5, x / hi)
    return out


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


class AdmixtureTests(TemporaryTests):
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

        # Audit counterexample: a minor ancestry must retain its supported allele.
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
                subsampling=4,
                random_init=True,
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
