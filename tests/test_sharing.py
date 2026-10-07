"""Exhaustive references for linked sharing, sparse products, and CLI boundaries."""

__author__ = "Jonas Meisner"

import numpy as np
from helpers import TemporaryTests, command, readLog, writeClusters

from hapla import sharing
from hapla import sharing_cy as cy
from hapla.formats import readWindows


### Find nonextendable best suffixes by exhaustive comparison
def exactMatches(Z, cut, info, rank, cap):
    W, H = Z.shape
    out = []
    for w in range(1, W + 1):
        for h in range(H):
            starts = {}
            for g in range(H):
                if h // 2 == g // 2:
                    continue
                b = w
                while b and info[b - 1] and Z[b - 1, h] != 255 and Z[b - 1, h] == Z[b - 1, g]:
                    b -= 1
                    if cut[b]:
                        break
                starts[g] = b
            b = min(starts.values())
            donors = [g for g, s in starts.items() if s == b]
            if b == w:
                continue
            if w < W and not cut[w] and info[w] and Z[w, h] != 255:
                if any(Z[w, h] == Z[w, g] for g in donors):
                    continue
            out.extend((h, g, b, w) for g in sorted(donors, key=lambda g: rank[g])[:cap])
    return np.asarray(out, dtype=np.uint32).reshape(-1, 4)


### Calculate dense sharing for small fixtures
def exactSharing(rec, H, x, left, right):
    P = np.zeros((H, len(x), H))
    for h, g, b, e in rec:
        P[h, b:e, g] += (x[b:e] - left[b]) * (right[e - 1] - x[b:e])
    den = P.sum(axis=2)
    np.divide(P, den[:, :, None], out=P, where=den[:, :, None] > 0)
    C = np.einsum("hwg,w->hg", P, right - left)
    C = C.reshape(H // 2, 2, H // 2, 2).sum(axis=(1, 3))
    cov = ((den > 0) @ (right - left)).reshape(-1, 2).sum(axis=1)
    return C, cov


### Check matching and products against independent references
class SharingTests(TemporaryTests):
    def test_categorical_pbwt_matches_exhaustive_reference(self):
        rng = np.random.default_rng(72)
        for rep in range(35):
            W, H = 19, 14 if rep < 30 else 80
            Z = rng.integers(0, 4, (W, H), dtype=np.uint8)
            Z[rng.random(Z.shape) < 0.09] = 255
            if rep % 3 == 0:
                Z[3:9] = np.tile(np.arange(H) % 2, (6, 1))
            info = np.array([len(set(row) - {255}) > 1 for row in Z], np.uint8)
            cut = (rng.random(W) < 0.1).astype(np.uint8)
            cut[0] = 1
            rank = rng.permutation(H).astype(np.int64)
            cap = (1, 3, 16, H)[rep % 4]
            path = self.root / str(rep)
            count, total, _ = cy.matches(Z, cut, info, rank, cap, bytes(path))
            actual = np.fromfile(path, np.uint32).reshape(-1, 4)
            expected = exactMatches(Z, cut, info, rank, cap)
            self.assertEqual(sorted(map(tuple, actual)), sorted(map(tuple, expected)))
            self.assertEqual(total, len(expected))
            np.testing.assert_array_equal(count, np.bincount(expected[:, 0], minlength=H))
            for h in range(H):
                np.testing.assert_array_equal(
                    actual[actual[:, 0] == h], expected[expected[:, 0] == h]
                )

    def test_dense_products_and_eigenspaces(self):
        rng = np.random.default_rng(42)
        Z = rng.integers(0, 3, (43, 18), dtype=np.uint8)
        Z[8:11, 0] = 255
        Z[20:25] = Z[19]
        info = np.ones(len(Z), np.uint8)
        cut = np.zeros(len(Z), np.uint8)
        cut[[0, 18]] = 1
        rank = rng.permutation(Z.shape[1]).astype(np.int64)
        x = np.cumsum(rng.uniform(0.1, 0.4, len(Z)))
        left, right = x - 0.025, x + 0.025
        path = self.root / "cache"
        path.mkdir()
        count, _, _ = cy.matches(Z, cut, info, rank, 4, bytes(path / "matches"))
        rec = np.fromfile(path / "matches", np.uint32).reshape(-1, 4)
        C, exposure = exactSharing(rec, Z.shape[1], x, left, right)
        data, cov = sharing.paint(path, count, x, left, right)
        np.testing.assert_allclose(cov, exposure, atol=1e-13)
        row, col, val = data
        actual = np.zeros_like(C)
        for i in range(len(C)):
            actual[i, col[row[i] : row[i + 1]]] = val[row[i] : row[i + 1]]
        np.testing.assert_allclose(actual, C, atol=1e-13)
        np.testing.assert_allclose(np.diag(C), 0, atol=0)
        C /= exposure[:, None]
        np.testing.assert_allclose(C.sum(axis=1), 1, atol=2e-15)
        Q = rng.normal(size=(len(C), 5))
        for root in (False, True):
            Y = np.sqrt(C) if root else C
            B = Y - Y.mean(axis=0)
            d = sharing.profiles([data], cov, self.root / f"profiles{root}", root=root)
            np.testing.assert_allclose(sharing.product(d, Q), B @ Q, atol=1e-13)
            np.testing.assert_allclose(sharing.product(d, Q, True), B.T @ Q, atol=1e-13)

            # Chromosome accumulation must normalize once using combined exposure.
            merged = sharing.profiles([data, data], 2 * cov, self.root / f"merged{root}", root=root)
            np.testing.assert_allclose(sharing.product(merged, Q), B @ Q, atol=1e-13)
            V, S = sharing.pca(d, 4, 2, 92)
            U, exact, _ = np.linalg.svd(B, full_matrices=False)
            np.testing.assert_allclose(S, exact[:4], atol=1e-13)
            np.testing.assert_allclose(V.T @ V, np.eye(4), atol=2e-12)
            np.testing.assert_allclose(V @ V.T, U[:, :4] @ U[:, :4].T, atol=2e-12)

    def test_full_kernel_combines_chromosomes_before_transformation(self):
        rng = np.random.default_rng(171)
        N = 17
        data, C = [], np.zeros((N, N))
        for n in range(2):
            A = rng.uniform(0.1, 1, (N, N))
            A[rng.random(A.shape) < 0.6] = 0
            A[n] = 0
            np.fill_diagonal(A, 0)
            i, j = np.nonzero(A)
            ptr = np.r_[0, np.cumsum(np.bincount(i, minlength=N))]
            col, val = j.astype(np.uint32), A[i, j]
            data.append((ptr, col, val))
            C += A
        cov = C.sum(axis=1)
        C /= cov[:, None]
        for root in (False, True):
            Y = np.sqrt(C) if root else C
            B = Y - Y.mean(axis=0)
            d = sharing.profiles(data, cov, self.root / f"profiles{root}", root=root)
            mean, ss = sharing.moments(d)
            np.testing.assert_allclose(mean, Y.mean(axis=0), atol=1e-15)
            np.testing.assert_allclose(ss, np.sum(B * B), rtol=1e-13)
            scale = (N - 1) / ss
            G = scale * (B @ B.T)
            for tile in (1, 4, N):
                path = self.root / "kernel.bin"
                sharing.grm(path, d, mean, ss, tile)
                actual = np.fromfile(path, "<f4")
                self.assertEqual(path.stat().st_size, 4 * N * (N + 1) // 2)
                np.testing.assert_allclose(actual, G[np.tril_indices(N)], rtol=6e-8, atol=1e-8)
            V, S = sharing.pca(d, N - 1, 2, 42)
            np.testing.assert_allclose(V.T @ V, np.eye(N - 1), atol=1e-12)
            np.testing.assert_allclose((V * (S * S * scale)) @ V.T, G, atol=1e-12)
            np.testing.assert_allclose(G @ V, V * (S * S * scale), atol=1e-12)
            np.testing.assert_allclose(G.sum(axis=0), 0, atol=1e-14)
            self.assertAlmostEqual(np.trace(G), N - 1)

    def test_uniform_profiles_keep_small_centered_variance(self):
        # Uniform off-diagonal profiles have squared centered norm 1/(N-1), or 1 with sqrt.
        N = 2001
        ptr = np.arange(N + 1, dtype=np.int64) * (N - 1)
        col = np.tile(np.arange(N - 1, dtype=np.uint32), (N, 1))
        col += col >= np.arange(N)[:, None]
        data = [(ptr, col.ravel(), np.ones(N * (N - 1)))]
        for root in (False, True):
            d = sharing.profiles(
                data,
                np.full(N, N - 1, dtype=float),
                self.root / f"uniform{root}",
                root=root,
                transpose=False,
            )
            mean, ss = sharing.moments(d)
            np.testing.assert_allclose(mean, (np.sqrt(N - 1) if root else 1) / N, rtol=1e-12)
            np.testing.assert_allclose(ss, 1 if root else 1 / (N - 1), rtol=1e-9)

    def test_ties_are_label_and_sample_order_invariant(self):
        rng = np.random.default_rng(6)
        Z = rng.integers(0, 2, (25, 22), dtype=np.uint8)
        Z[4:18] = np.tile(np.arange(22) % 3, (14, 1))
        rank = rng.permutation(Z.shape[1]).astype(np.int64)
        info, cut = np.ones(len(Z), np.uint8), np.zeros(len(Z), np.uint8)
        cut[0] = 1
        h = (2 * rng.permutation(11)[:, None] + [0, 1]).ravel()
        expected = None
        for rep, (z, r) in enumerate(((Z, rank), (8 - Z, rank), (Z[:, h].copy(), rank[h]))):
            path = self.root / str(rep)
            cy.matches(z, cut, info, r, 3, bytes(path))
            rec = np.fromfile(path, np.uint32).reshape(-1, 4)
            if rep == 2:
                rec[:, :2] = h[rec[:, :2]]
            actual = sorted(map(tuple, rec))
            if expected is None:
                expected = actual
            self.assertEqual(actual, expected)

    def test_different_phased_tuples_with_identical_dosages(self):
        Z = np.array([[0, 1, 0, 1, 0, 1, 0, 1], [0, 1, 0, 1, 1, 0, 1, 0]], np.uint8)
        cut, info = np.array([1, 0], np.uint8), np.ones(2, np.uint8)
        path = self.root / "tuples"
        cy.matches(Z, cut, info, np.arange(8, dtype=np.int64), 8, bytes(path))
        rec = np.fromfile(path, np.uint32).reshape(-1, 4)
        C, cov = exactSharing(rec, 8, np.array([0.5, 1.5]), np.arange(2), np.arange(1, 3))
        C /= cov[:, None]
        self.assertGreater(C[0, 1], C[0, 2])
        self.assertGreater(C[2, 3], C[2, 0])

    def test_geometry_constants_and_gaps(self):
        rows = [("1", b, b + 4, 4, 2, 2) for b in (1, 10, 20, 1000)]
        info = np.array([True, False, True, True])
        x, left, right, cut = sharing.geometry(rows, info, 50)
        np.testing.assert_allclose(x, np.array([3, 12, 22, 1002]) / 1e6)
        np.testing.assert_array_equal(cut, 1)
        np.testing.assert_allclose(right - left, 4e-6)
        with self.assertRaisesRegex(ValueError, "nonoverlapping"):
            sharing.geometry(rows[::-1], info, 50)

    def test_cli_threads_and_file_partitioning(self):
        rng = np.random.default_rng(81)
        Z = rng.integers(0, 3, (29, 20), dtype=np.uint8)
        Z[12, 0] = 255
        c = np.arange(len(Z) + 1, dtype=np.int64) * 3
        src = writeClusters(self.root, "all", Z, c)
        rows = readWindows(src)
        a = writeClusters(self.root, "a", Z[:13], c[:14])
        b = writeClusters(self.root, "b", Z[13:], c[13:] - c[13])
        for path, meta in ((a, rows[:13]), (b, rows[13:])):
            with open(f"{path}.win", "w") as dst:
                dst.writelines(" ".join(map(str, row)) + "\n" for row in meta)
        for root in (False, True):
            out = []
            for n, (paths, threads) in enumerate((([src], 1), ([a, b], 4))):
                pfx = self.root / f"result{root}{n}"
                result = command(
                    "struct",
                    "--clusters",
                    *paths,
                    "--hsm-svd",
                    3,
                    "--threads",
                    threads,
                    "--raw",
                    *(("--hsm-sqrt",) if root else ()),
                    "--out",
                    pfx,
                )
                self.assertIn("physical distance proxy", result.stdout)
                out.append(np.loadtxt(f"{pfx}.hsm.vec"))
                meta = readLog(pfx)
                self.assertEqual(meta["Profile transform"], "sqrt" if root else "linear")
                self.assertEqual(meta["Projection supported"], "no")
                self.assertGreater(float(meta["Gower scale"]), 0)
                self.assertFalse((self.root / f"result{root}{n}.hsm.json").exists())
                self.assertFalse(list(self.root.glob(".hapla-*")))
            np.testing.assert_array_equal(*out)
        result = command("struct", "--clusters", src, "--hsm-svd", 2, "--pca", 2, success=False)
        self.assertIn("separate analysis", result.stderr)
        for opt in (("--hsm-gap", 50), ("--hsm-sqrt",)):
            result = command("struct", "--clusters", src, "--pca", 2, *opt, success=False)
            self.assertIn("require --hsm", result.stderr)

    def test_cli_kernel_components_ids_and_optional_outputs(self):
        rng = np.random.default_rng(291)
        Z = rng.integers(0, 3, (29, 20), dtype=np.uint8)
        src = writeClusters(self.root, "input", Z, np.arange(30, dtype=np.int64) * 3)
        kernels, coverage = [], []
        for opt in ((), ("--hsm-sqrt",)):
            pfx = self.root / "kernel"
            args = ("struct", "--clusters", src, "--out", pfx, *opt)
            (self.root / "kernel.hsm.json").write_text("previous\n")
            command(*args, "--hsm", "--hsm-svd", 9, "--threads", 4, "--duplicate-fid", "--raw")
            self.assertFalse((self.root / "kernel.hsm.json").exists())
            vals = np.loadtxt(f"{pfx}.hsm.val")
            V = np.loadtxt(f"{pfx}.hsm.vec")
            packed = np.fromfile(f"{pfx}.hsm.grm.bin", "<f4")
            kernels.append(packed)
            coverage.append(np.loadtxt(f"{pfx}.hsm.cov", usecols=(1, 2)))
            meta = readLog(pfx)
            G = np.zeros((10, 10))
            G[np.tril_indices(10)] = packed
            G += np.tril(G, -1).T
            np.testing.assert_allclose(G, (V * vals) @ V.T, rtol=1e-6, atol=1e-7)
            np.testing.assert_allclose(V.T @ V, np.eye(9), atol=1e-8)
            np.testing.assert_allclose(G @ V, V * vals, rtol=1e-6, atol=1e-7)
            self.assertGreater(np.linalg.eigvalsh(G)[0], -1e-7)
            self.assertAlmostEqual(np.trace(G), 9, places=6)
            self.assertEqual(meta["Matrix exported"], "yes")
            self.assertEqual(meta["SNP counts available"], "no")
            self.assertEqual(meta["Component scaling"], "unit norm eigenvectors")
            self.assertEqual(meta["Profile transform"], "sqrt" if opt else "linear")
            ids = np.loadtxt(f"{pfx}.hsm.grm.id", dtype=str)
            np.testing.assert_array_equal(ids[:, 0], ids[:, 1])
            np.testing.assert_array_equal(ids[:, 1], np.loadtxt(f"{src}.ids", dtype=str))
            self.assertFalse((self.root / "kernel.hsm.grm.N.bin").exists())
            command(*args, "--hsm", "--power", 1)
            np.testing.assert_array_equal(packed, np.fromfile(f"{pfx}.hsm.grm.bin", "<f4"))
            single = readLog(pfx)
            self.assertEqual(
                2 * int(single["Cache bytes"].replace(",", "")),
                int(meta["Cache bytes"].replace(",", "")),
            )
            self.assertNotIn("Component scaling", single)
            self.assertFalse((self.root / "kernel.hsm.vec").exists())
            ids = np.loadtxt(f"{pfx}.hsm.grm.id", dtype=str)
            np.testing.assert_array_equal(ids[:, 0], "0")
            command(*args, "--hsm-svd", 9, "--raw")
            np.testing.assert_allclose(vals, np.loadtxt(f"{pfx}.hsm.val"), rtol=1e-9)
            self.assertFalse((self.root / "kernel.hsm.grm.bin").exists())
            self.assertFalse((self.root / "kernel.hsm.grm.id").exists())
        self.assertGreater(np.linalg.norm(kernels[0] - kernels[1]), 1e-4)
        np.testing.assert_array_equal(*coverage)
        for opt in (("--hsm",), ("--pca", 2)):
            result = command(*args, *opt, "--grm-no-center", success=False)
            self.assertIn("--grm-no-center requires --grm", result.stderr)
        for k in (0, 10):
            command(*args, "--hsm-svd", k, success=False)
        self.assertFalse(list(self.root.glob(".hapla-*")))

    def test_constant_data_has_no_sharing_signal(self):
        Z = np.zeros((3, 8), np.uint8)
        src = writeClusters(self.root, "constant", Z, np.arange(4, dtype=np.int64))
        result = command("struct", "--clusters", src, "--hsm-svd", 2, success=False)
        self.assertIn("No usable haplotype sharing", result.stderr)
        Z = np.tile([0, 1, 1, 0, 0, 1, 1, 0], (3, 1)).astype(np.uint8)
        src = writeClusters(self.root, "identical", Z, np.arange(4, dtype=np.int64) * 2)
        result = command("struct", "--clusters", src, "--hsm-svd", 2, success=False)
        self.assertIn("No usable haplotype sharing", result.stderr)
        Z[1, :2] = Z[1, :2][::-1]
        self.assertTrue(cy.variable(Z))

    def test_blocked_painting_with_empty_samples_and_zero_length_cells(self):
        rng = np.random.default_rng(21)
        Z = rng.integers(0, 3, (8, 278), dtype=np.uint8)
        Z[:, :250] = 255
        cut, info = np.zeros(8, np.uint8), np.ones(8, np.uint8)
        cut[0] = 1
        rank = rng.permutation(278).astype(np.int64)
        x = np.arange(8, dtype=float)
        left, right = x - 0.5, x + 0.5
        left[3] = right[3] = x[3]
        for bins in (1, 3):
            path = self.root / f"blocks{bins}"
            path.mkdir()
            count, _, _ = cy.matches(Z, cut, info, rank, 3, bytes(path / "matches"), bins)
            rec = np.concatenate(
                [np.fromfile(p, np.uint32).reshape(-1, 4) for p in sorted(path.glob("matches*"))]
            )
            C, exposure = exactSharing(rec, 278, x, left, right)
            data, cov = sharing.paint(path, count, x, left, right, bins)
            np.testing.assert_allclose(cov, exposure, atol=1e-13)
            rows, col, val = data
            actual = np.zeros_like(C)
            for i in range(len(C)):
                actual[i, col[rows[i] : rows[i + 1]]] = val[rows[i] : rows[i + 1]]
            np.testing.assert_allclose(actual, C, atol=1e-13)

    def test_chromosome_and_constant_boundaries(self):
        Z = np.tile([0, 1, 0, 1, 0, 1], (7, 1)).astype(np.uint8)
        Z[3] = 0
        info = np.ones(7, np.uint8)
        info[3] = 0
        cut = np.zeros(7, np.uint8)
        cut[[0, 6]] = 1
        path = self.root / "breaks"
        cy.matches(Z, cut, info, np.arange(6, dtype=np.int64), 6, bytes(path))
        rec = np.fromfile(path, np.uint32).reshape(-1, 4)
        self.assertEqual(set(map(tuple, rec[:, 2:])), {(0, 3), (4, 6), (6, 7)})
        a = writeClusters(self.root, "chr1", Z[:3], np.arange(4) * 2)
        b = writeClusters(self.root, "chr2", Z[4:], np.arange(4) * 2)
        with open(f"{b}.win", "w") as dst:
            dst.writelines("2 " + " ".join(map(str, r[1:])) + "\n" for r in readWindows(a))
        p = np.tile([0.5, 0.5], 6)
        data = [(Z[:3], np.arange(4) * 2, None), (Z[4:], np.arange(4) * 2, None)]
        parts = sharing.chromosomes([a, b], data, p)
        self.assertEqual([part[0] for part in parts], ["1", "2"])
