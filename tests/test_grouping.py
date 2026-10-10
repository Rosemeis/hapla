"""Dense references for implicit HSM grouping and membership output."""

__author__ = "Jonas Meisner"

from pathlib import Path
from unittest.mock import patch

import numpy as np
from helpers import TemporaryTests, command, readLog, writeClusters

from hapla import grouping, sharing


### Normalize a small dense profile fixture through the sparse cache
def profiles(C, path, root=False):
    i, j = np.nonzero(C)
    row = np.r_[0, np.cumsum(np.bincount(i, minlength=len(C)))].astype(np.int64)
    cov = C.sum(axis=1)
    data = sharing.profiles([(row, j.astype(np.uint32), C[i, j])], cov, path, root=root)
    X = C / cov[:, None]
    if root:
        X = np.sqrt(X)
    return data, X - X.mean(axis=0)


### Calculate centroid distances without the sparse implementation
def distances(X, z, K):
    means = np.array([X[z == k].mean(axis=0) for k in range(K)])
    return np.sum((X[:, None, :] - means[None, :, :]) ** 2, axis=2) * ((len(X) - 1) / np.sum(X * X))


### Check kernel grouping independently of randomized initialization
class GroupingTests(TemporaryTests):
    def test_frozen_centroid_scores_match_dense_kernel(self):
        rng = np.random.default_rng(981)
        N, K = 13, 3
        data, X = profiles(rng.uniform(0.01, 1, (N, N)), self.root / "cache")
        z = rng.permutation(np.arange(N) % K)
        score, norm, count = grouping.state(data, z, K)
        means = np.array([X[z == k].mean(axis=0) for k in range(K)])
        expected = np.sum(means * means, axis=1)
        np.testing.assert_allclose(norm, expected, atol=1e-14)
        np.testing.assert_allclose(score, expected - 2 * X @ means.T, atol=1e-14)
        np.testing.assert_array_equal(count, np.bincount(z, minlength=K))
        for bad in (z[:-1], np.r_[z, 0], z[:, None]):
            with self.assertRaisesRegex(ValueError, "sample count"):
                grouping.state(data, bad, K)
        with self.assertRaisesRegex(ValueError, "profile dimensions"):
            grouping.state(data, z, K, np.zeros(N - 1))

    def test_sparse_group_profiles_match_dense_centering(self):
        rng = np.random.default_rng(95)
        N, K = 19, 5
        C = rng.uniform(0.02, 1, (N, N))
        C[rng.random(C.shape) < 0.7] = 0
        C[np.arange(N), np.arange(N)] = 1
        z = rng.permutation(np.arange(N) % K).astype(np.int32)
        n = np.bincount(z, minlength=K)
        for root in (False, True):
            data, X = profiles(C, self.root / f"centers{root}", root)
            mean, _ = sharing.moments(data)
            out = np.empty((N, K))
            sharing.cy.centroids(*data[1], z, 1 / n, mean, out)
            expected = np.array([X[z == k].mean(axis=0) for k in range(K)]).T
            np.testing.assert_allclose(out, expected, atol=1e-14)

    def test_kernel_refinement_reuses_training_mean(self):
        rng = np.random.default_rng(147)
        N, K = 24, 4
        data, X = profiles(rng.uniform(0.02, 1, (N, N)), self.root / "cache")
        mean, ss = sharing.moments(data)
        start = (np.arange(N) % K).astype(np.int32)
        with patch("hapla.sharing.moments", side_effect=AssertionError("Mean recomputed")):
            z, _, _, _, done, _ = grouping.refine(data, start, K, X, ss, mean)
        self.assertTrue(done)
        self.assertEqual(len(np.unique(z)), K)
        with patch("hapla.sharing.moments", side_effect=AssertionError("Mean recomputed")):
            z, _, _ = grouping.fit(data, K, 2, 42, scores=X[:, :5], mom=(mean, ss))
        self.assertEqual(len(np.unique(z)), K)

    def test_dense_centroid_objective_and_margin(self):
        rng = np.random.default_rng(82)
        N, K = 31, 4
        C = rng.uniform(0.05, 1, (N, N))
        C[rng.random(C.shape) < 0.65] = 0
        C[np.arange(N), rng.integers(N, size=N)] += 1
        for root in (False, True):
            data, X = profiles(C, self.root / f"cache{root}", root)
            z, margin, info = grouping.fit(data, K, 2, 41, scores=X[:, :8])
            d = distances(X, z, K)
            own = d[np.arange(N), z]
            d[np.arange(N), z] = np.inf
            np.testing.assert_allclose(margin, np.min(d, axis=1) - own, atol=2e-12)
            self.assertAlmostEqual(info["objective"], own.sum(), places=11)
            self.assertGreaterEqual(margin.min(), -2e-12)
            np.testing.assert_array_equal(np.bincount(z, minlength=K), info["group_sizes"])
            self.assertEqual(len(np.unique(z)), K)

    def test_exact_kernel_refinement_never_increases_objective(self):
        rng = np.random.default_rng(613)
        N, K = 28, 4
        data, X = profiles(rng.uniform(0.01, 1, (N, N)), self.root / "cache")
        start = rng.permutation(np.arange(N) % K).astype(np.int32)
        _, ss = sharing.moments(data)
        z, _, loss, _, done, history = grouping.refine(data, start, K, X, ss)
        self.assertTrue(done)
        self.assertLessEqual(np.max(np.diff(history), initial=0), 1e-13)
        factor = ss / (N - 1)
        expected = distances(X, start, K)[np.arange(N), start].sum() * factor
        self.assertAlmostEqual(history[0], expected, places=13)
        expected = distances(X, z, K)[np.arange(N), z].sum() * factor
        self.assertAlmostEqual(loss, expected, places=13)
        self.assertEqual(history[-1], loss)

    def test_reproducible_grouping_recovers_separated_profiles(self):
        rng = np.random.default_rng(11)
        N, K = 18, 3
        target = np.repeat(np.arange(K), N // K)
        C = rng.uniform(0.01, 0.04, (N, N))
        C[np.arange(N), target] += 1
        data, X = profiles(C, self.root / "cache")
        first = grouping.fit(data, K, 2, 92)
        second = grouping.fit(data, K, 2, 92)
        for a, b in zip(first[:2], second[:2]):
            np.testing.assert_array_equal(a, b)
        z, margin, info = first
        np.testing.assert_array_equal(z[:, None] == z, target[:, None] == target)
        self.assertTrue(np.all(margin > 0))
        self.assertAlmostEqual(info["objective"], distances(X, z, K)[np.arange(N), z].sum())

    def test_duplicate_profiles_and_rank_deficient_initialization(self):
        N = 12
        C = np.zeros((N, N))
        C[: N // 2, 0] = 1
        C[N // 2 :, 1] = 1
        data, X = profiles(C, self.root / "cache")
        for K in (2, 4, N - 1):
            z, margin, info = grouping.fit(data, K, 2, 81)
            self.assertEqual(len(np.unique(z)), K)
            self.assertTrue(np.all(np.isfinite(margin)))
            d = distances(X, z, K)
            d[np.arange(N), z] = np.inf
            np.testing.assert_allclose(margin, d.min(axis=1), atol=1e-12)
            self.assertAlmostEqual(info["objective"], 0, places=11)
            np.testing.assert_allclose(distances(X, z, K)[np.arange(N), z], 0, atol=1e-12)

    def test_minimum_maximum_and_invalid_group_count(self):
        N = 7
        data, X = profiles(np.eye(N), self.root / "cache")
        z, margin, info = grouping.fit(data, 1, 2, 82)
        np.testing.assert_array_equal(z, 0)
        np.testing.assert_array_equal(margin, 0)
        self.assertAlmostEqual(info["objective"], N - 1)
        z, margin, info = grouping.fit(data, N - 1, 2, 82)
        self.assertEqual(len(np.unique(z)), N - 1)
        self.assertAlmostEqual(info["objective"], 1, places=11)
        d = distances(X, z, N - 1)
        own = d[np.arange(N), z]
        d[np.arange(N), z] = np.inf
        np.testing.assert_allclose(margin, np.min(d, axis=1) - own, atol=1e-12)
        for K in (0, -1, N, N + 1):
            with self.assertRaises(ValueError):
                grouping.fit(data, K, 2, 82)

    def test_kernel_iterations_use_only_thin_products(self):
        rng = np.random.default_rng(174)
        N, K = 53, 3
        data, X = profiles(rng.uniform(0.1, 1, (N, N)), self.root / "cache")
        shapes, original = [], sharing.product

        def product(data, Q, transpose=False):
            shapes.append(Q.shape)
            self.assertEqual(Q.shape[0], N)
            self.assertLess(Q.shape[1], N)
            return original(data, Q, transpose)

        with patch("hapla.sharing.product", product):
            z, _, _ = grouping.fit(data, K, 2, 44, scores=X[:, :5])
        self.assertTrue(shapes)
        self.assertEqual(len(np.unique(z)), K)


### Check membership files, combined analyses, and replacement
class GroupingPipeline(TemporaryTests):
    def fixture(self):
        rng = np.random.default_rng(193)
        Z = rng.integers(0, 3, (41, 24), dtype=np.uint8)
        Z[8:11, 0] = 255
        return writeClusters(self.root, "input", Z, np.arange(42, dtype=np.int64) * 3)

    def test_standalone_membership_preserves_sample_order(self):
        src, out = self.fixture(), self.root / "groups"
        result = command("struct", "--clusters", src, "--hsm-groups", 3, "--power", 2, "--out", out)
        path = Path(f"{out}.hsm.grp")
        self.assertEqual(path.read_text().splitlines()[0], "#FID\tIID\tGROUP\tMARGIN")
        rows = np.loadtxt(path, dtype=str)
        np.testing.assert_array_equal(rows[:, 0], "0")
        np.testing.assert_array_equal(rows[:, 1], np.loadtxt(f"{src}.ids", dtype=str))
        np.testing.assert_array_equal(np.unique(rows[:, 2].astype(int)), [1, 2, 3])
        self.assertTrue(np.all(np.isfinite(rows[:, 3].astype(float))))
        self.assertTrue(Path(f"{out}.hsm.cov").is_file())
        for suffix in (".hsm.vec", ".hsm.val", ".hsm.grm.bin", ".hsm.json"):
            self.assertFalse(Path(f"{out}{suffix}").exists())
        self.assertIn("Grouping", result.stdout)
        self.assertFalse(list(self.root.glob(".hapla-*")))

    def test_combined_svd_is_unchanged_and_group_output_is_replaced(self):
        src, out = self.fixture(), self.root / "result"
        args = (
            "struct",
            "--clusters",
            src,
            "--hsm-svd",
            3,
            "--power",
            2,
            "--duplicate-fid",
            "--out",
            out,
        )
        command(*args)
        previous = {s: Path(f"{out}{s}").read_bytes() for s in (".hsm.vec", ".hsm.val")}
        command(*args, "--hsm-groups", 3)
        for suffix, value in previous.items():
            self.assertEqual(Path(f"{out}{suffix}").read_bytes(), value)
        rows = np.loadtxt(f"{out}.hsm.grp", dtype=str)
        np.testing.assert_array_equal(rows[:, 0], rows[:, 1])
        command(*args)
        self.assertFalse(Path(f"{out}.hsm.grp").exists())
        self.assertEqual(readLog(out)["Component scaling"], "unit norm eigenvectors")

    def test_invalid_modes_and_counts_preserve_existing_memberships(self):
        src, out = self.fixture(), self.root / "result"
        path = Path(f"{out}.hsm.grp")
        path.write_text("previous\n")
        for opts in (
            ("--hsm-groups", 0),
            ("--hsm-groups", 12),
            ("--hsm-groups", 13),
            ("--hsm-groups", 3, "--pca", 2),
            ("--hsm-groups", 3, "--grm"),
        ):
            command("struct", "--clusters", src, "--out", out, *opts, success=False)
            self.assertEqual(path.read_text(), "previous\n")
        command("struct", "--clusters", src, "--hsm-groups", 1, "--power", 2, "--out", out)
        rows = np.loadtxt(path, dtype=str)
        np.testing.assert_array_equal(rows[:, 2].astype(int), 1)
        np.testing.assert_array_equal(rows[:, 3].astype(float), 0)
