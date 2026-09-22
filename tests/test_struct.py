"""Independent dense references for matrix-free PCA, projection, and GRM."""

import unittest
from pathlib import Path

import numpy as np
from helpers import TemporaryTests, command, missingFixture, structureFixture, writeClusters

from hapla import struct
from hapla import struct_cy as cy
from hapla.identity import featureKeys, writeModel


class DirectProducts(unittest.TestCase):
    def test_missing_products_grm_pca_and_reference_mean_projection(self):
        Z, c, p, D, obs = missingFixture()
        est, counts = np.empty_like(p), np.empty(len(Z), np.int64)
        cy.frequencies(Z, c, est, counts)
        np.testing.assert_array_equal(est, p)
        np.testing.assert_array_equal(counts, obs)
        cy.frequencies(Z, c, None, counts)
        np.testing.assert_array_equal(counts, obs)
        data = [(Z, c, obs)]
        E = D - 2 * p[:, None]
        a = struct.scale(p)
        X = E * a[:, None]
        Q = np.random.default_rng(9).normal(size=(D.shape[1], 5))
        A = np.empty((len(p), 5))
        cy.leftProduct(Z, c, p, a, Q, Q.sum(axis=0), A, obs)
        np.testing.assert_allclose(A, X @ Q, atol=5e-13)
        for chunk in (1, 8, 4096):
            np.testing.assert_allclose(
                struct.product(data, p, a, 5, chunk, Q=Q), X.T @ X @ Q, atol=3e-12
            )
            G, den = struct.grm(data, p, chunk, False, 6)
            np.testing.assert_allclose(G, (E.T @ E / (2 * den))[np.tril_indices(len(Q))], atol=4e-7)
        V, S, _ = struct.pca(data, p, 4, 7, 2, 1)
        U, exact, R = np.linalg.svd(X, full_matrices=False)
        np.testing.assert_allclose(S, exact[:4], rtol=2e-12)
        np.testing.assert_allclose(V @ V.T, R[:4].T @ R[:4], atol=3e-12)
        np.testing.assert_allclose(V[0], 0, atol=1e-13)
        U = np.ascontiguousarray(X @ V / S)
        np.testing.assert_allclose(struct.project(data, p, U, S * S / len(p), 7), V, atol=3e-12)

        # A different query has missing haplotypes: impute from reference p.
        query = Z.copy()
        query[:, 3:7] = 255
        obs2 = (query != 255).sum(axis=1, dtype=np.int64)
        D2 = np.zeros_like(D)
        for w, row in enumerate(query):
            for h, z in enumerate(row):
                if z == 255:
                    D2[c[w] : c[w + 1], h // 2] += p[c[w] : c[w + 1]]
                else:
                    D2[c[w] + int(z), h // 2] += 1
        expected = ((D2 - 2 * p[:, None]) * a[:, None]).T @ (U / S)
        actual = struct.project([(query, c, obs2)], p, U, S * S / len(p), 7)
        np.testing.assert_allclose(actual, expected, atol=3e-12)

    def test_frequency_validation_and_full_cluster_range(self):
        rng = np.random.default_rng(80)
        k = np.array([1, 2, 255], dtype=np.int64)
        c = np.r_[0, np.cumsum(k)]
        Z = np.array([rng.integers(0, K, 600) for K in k], dtype=np.uint8)
        p = np.empty(c[-1])
        cy.frequencies(Z, c, p, np.empty(len(Z), np.int64))
        expected = np.concatenate([np.bincount(z, minlength=K) / 600 for z, K in zip(Z, k)])
        np.testing.assert_array_equal(p, expected)
        bad = Z.copy()
        bad[1, 0] = 2
        for out in (p, None):
            with self.assertRaisesRegex(ValueError, "outside"):
                cy.frequencies(bad, c, out, np.empty(len(Z), np.int64))

    def test_both_products_equal_dense_matrix(self):
        Z, c, p, _, X = structureFixture()
        rng = np.random.default_rng(610)
        Q = rng.normal(size=(X.shape[1], 5))
        a = struct.scale(p)
        A = np.empty((len(p), 5))
        cy.leftProduct(Z, c, p, a, Q, Q.sum(axis=0), A)
        np.testing.assert_allclose(A, X @ Q, atol=2e-13)
        for chunk in (1, 9, 4096):
            H = struct.product([(Z, c, None)], p, a, 5, chunk, Q=Q)
            np.testing.assert_allclose(H, X.T @ X @ Q, atol=2e-12)

    def test_grm_dense_reference_tiling_and_partition_invariance(self):
        Z, c, p, D, _ = structureFixture()
        X = D - 2 * p[:, None]
        expected = X.T @ X / (2 * np.sum(p * (1 - p)))
        centered = (
            expected
            - expected.mean(axis=0)[None, :]
            - expected.mean(axis=1)[:, None]
            + expected.mean()
        )
        centered *= (D.shape[1] - 1) / np.trace(centered)
        split = 5
        parts = [(Z[:split], c[: split + 1], None), (Z[split:], c[split:] - c[split], None)]
        for data in ([(Z, c, None)], parts):
            for center in (False, True):
                for chunk, tile in ((1, 3), (8, 6), (4096, None)):
                    G, den = struct.grm(data, p, chunk, center, tile)
                    np.testing.assert_allclose(
                        G,
                        (centered if center else expected)[np.tril_indices(D.shape[1])],
                        atol=3e-7,
                    )
                    self.assertAlmostEqual(den, np.sum(p * (1 - p)))

    def test_pca_matches_exact_svd_and_training_projection(self):
        Z, c, p, _, X = structureFixture()
        _, exact, Vt = np.linalg.svd(X, full_matrices=False)
        for chunk in (7, 4096):
            V, S, a = struct.pca([(Z, c, None)], p, 4, chunk, 2, 40)
            np.testing.assert_allclose(S, exact[:4], rtol=1e-12)
            np.testing.assert_allclose(V @ V.T, Vt[:4].T @ Vt[:4], atol=1e-12)
            np.testing.assert_allclose(V.T @ V, np.eye(4), atol=1e-12)
            U = X @ V / S
            P = struct.project([(Z, c, None)], p, U, S * S / len(p), chunk)
            np.testing.assert_allclose(P, V, atol=2e-12)

    def test_rank_deficiency_is_finite_and_excess_components_fail(self):
        Z = np.tile(np.repeat([0, 0, 1, 1], 2), (6, 1)).astype(np.uint8)
        c, p = np.arange(0, 13, 2, dtype=np.int64), np.full(12, 0.5)
        V, S, _ = struct.pca([(Z, c, None)], p, 1, 5, 2, 4)
        self.assertTrue(np.all(np.isfinite(V)))
        self.assertTrue(S[0] > 0)
        with self.assertRaisesRegex(ValueError, "only 1"):
            struct.pca([(Z, c, None)], p, 2, 5, 2, 4)
        with self.assertRaisesRegex(ValueError, "dimensions"):
            struct.pca([(Z, c, None)], p, 4, 5, 2, 4)

    def test_maximum_clusters_and_unobserved_alleles_match_dense_reference(self):
        rng = np.random.default_rng(118)
        Z = rng.integers(0, 255, (2, 600), dtype=np.uint8)
        Z[:, 0] = 254
        c = np.array([0, 255, 510], dtype=np.int64)
        D = np.concatenate([np.eye(255)[z.reshape(300, 2)].sum(axis=1).T for z in Z])
        p = D.mean(axis=1) / 2
        self.assertTrue(np.any(p == 0))
        a = struct.scale(p)
        E = D - 2 * p[:, None]
        X = E * a[:, None]
        Q = rng.normal(size=(300, 3))
        H = struct.product([(Z, c, None)], p, a, 3, 255, Q=Q)
        np.testing.assert_allclose(H, X.T @ X @ Q, atol=2e-11)
        G, den = struct.grm([(Z, c, None)], p, 255, center=False, tile=73)
        expected = E.T @ E / (2 * den)
        np.testing.assert_allclose(G, expected[np.tril_indices(300)], atol=4e-7)


class StructurePipeline(TemporaryTests):
    def test_shared_reader_with_and_without_frequencies_across_files(self):
        Z, c, p, _, obs = missingFixture()
        a = writeClusters(self.root, "a", Z[:3], c[:4])
        b = writeClusters(self.root, "b", Z[3:], c[3:] - c[3])
        for freq in (False, True):
            data, q = struct.readData([a, b], np.diff(c), [3, 3], Z.shape[1] // 2, freq=freq)
            np.testing.assert_array_equal(np.concatenate([z for z, _, _ in data]), Z)
            np.testing.assert_array_equal(np.concatenate([o for _, _, o in data]), obs)
            if freq:
                np.testing.assert_array_equal(q, p)
            else:
                self.assertIsNone(q)
        with self.assertRaisesRegex(ValueError, "window counts"):
            struct.readData([a, b], np.diff(c), [3], Z.shape[1] // 2)

    def test_missing_and_empty_windows_roundtrip_all_modes(self):
        Z, c, p, _, _ = missingFixture()
        ref = writeClusters(self.root, "input", Z, c)
        results = []
        for nt in (1, 4):
            out = self.root / f"missing{nt}"
            command(
                "struct",
                "--clusters",
                ref,
                "--pca",
                4,
                "--loadings",
                "--power",
                2,
                "--grm",
                "--raw",
                "--threads",
                nt,
                "--out",
                out,
            )
            command(
                "struct",
                "--clusters",
                ref,
                "--projection",
                out,
                "--raw",
                "--out",
                self.root / "project",
            )
            V = np.loadtxt(f"{out}.eigenvecs")
            np.testing.assert_allclose(
                np.loadtxt(self.root / "project.project.eigenvecs"), V, atol=3e-9
            )
            np.testing.assert_allclose(np.loadtxt(f"{out}.freqs"), p, atol=1e-10)
            results.append(V)
        np.testing.assert_allclose(*results, atol=2e-12)

    def test_combined_modes_multifile_threads_and_single_pc_export(self):
        Z, c, p, _, _ = structureFixture()
        ref = writeClusters(self.root, "input", Z, c)
        split = 5
        a = writeClusters(self.root, "chr1", Z[:split], c[: split + 1])
        b = writeClusters(self.root, "chr2", Z[split:], c[split:] - c[split])
        filelist = self.root / "files"
        filelist.write_text(f"{a}\n{b}\n")
        outputs = []
        for i, opt in enumerate(
            (("--clusters", ref), ("--filelist", filelist), ("--clusters", a, b))
        ):
            out = self.root / f"result{i}"
            command(
                "struct",
                *opt,
                "--pca",
                1,
                "--power",
                2,
                "--loadings",
                "--grm",
                "--threads",
                1 if i == 0 else 4,
                "--raw",
                "--out",
                out,
            )
            command("struct", *opt, "--projection", out, "--raw", "--out", self.root / f"query{i}")
            V = np.loadtxt(f"{out}.eigenvecs")
            np.testing.assert_allclose(
                np.loadtxt(self.root / f"query{i}.project.eigenvecs"), V, atol=2e-9
            )
            np.testing.assert_allclose(np.loadtxt(f"{out}.freqs"), p, atol=1e-10)
            outputs.append(np.fromfile(f"{out}.grm.bin", np.float32))
        for G in outputs[1:]:
            np.testing.assert_allclose(outputs[0], G, atol=3e-7)

    def test_invalid_labels_and_cluster_metadata_preserve_existing_outputs(self):
        Z, c, _, _, _ = structureFixture()
        Z[0, 0] = 1
        ref = writeClusters(self.root, "missing", Z, c)
        out = self.root / "result"
        Path(f"{out}.eigenvecs").write_text("previous\n")
        res = command("struct", "--clusters", ref, "--pca", 1, "--out", out, success=False)
        self.assertIn("outside", res.stderr)
        self.assertEqual(Path(f"{out}.eigenvecs").read_text(), "previous\n")
        Path(f"{ref}.win").write_text("#CHROM START END LENGTH SIZE K\n1 1 1 0 1 4294967295\n")
        res = command("struct", "--clusters", ref, "--grm", "--out", out, success=False)
        self.assertIn("0..255", res.stderr)

    def test_invalid_projection_rolls_back_completed_grm(self):
        Z, c, p, _, _ = structureFixture()
        ref = writeClusters(self.root, "input", Z, c)
        model = self.root / "model"
        np.savetxt(f"{model}.freqs", p)
        np.savetxt(f"{model}.loadings", np.ones((len(p), 1)))
        np.savetxt(f"{model}.eigenvals", [0])
        files = {s: Path(f"{model}{s}") for s in (".freqs", ".loadings", ".eigenvals", ".pca.json")}
        writeModel(files, featureKeys([ref], np.diff(c), [len(Z)]))
        out = self.root / "out"
        Path(f"{out}.grm.bin").write_bytes(b"previous")
        res = command(
            "struct", "--clusters", ref, "--grm", "--projection", model, "--out", out, success=False
        )
        self.assertIn("positive eigenvalues", res.stderr)
        self.assertEqual(Path(f"{out}.grm.bin").read_bytes(), b"previous")
        self.assertFalse(list(self.root.glob(".hapla-*")))
