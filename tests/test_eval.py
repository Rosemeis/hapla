"""Independent dense residual projections, missingness, and transactional CLI output."""

__author__ = "Jonas Meisner"

import io
import unittest
from pathlib import Path

import numpy as np
from hapla.eval_cy import writeMatrix
from helpers import TemporaryTests, command, writeClusters

from hapla.eval import correlation, covariance


### Build cluster labels and ancestry proportions with optional missingness
def fixture(missing=False, N=27):
    rng = np.random.default_rng(171)
    k = np.array([2, 7, 1, 255, 4, 3])
    Z = np.array([rng.integers(C, size=2 * N) for C in k], np.uint8)
    if missing:
        Z[rng.random(Z.shape) < 0.25] = 255
        Z[:, :2] = 255
        Z[1] = 255
        Z[2] = 255
        k[2] = 0
    c = np.r_[0, np.cumsum(k)].astype(np.int64)
    obs = (Z != 255).sum(axis=1) if missing else None
    Q = rng.dirichlet(np.ones(3), size=N)
    return Z, c, obs, Q


### Compute residual covariances through explicit dosage projections
def dense(Z, c, Q):
    N = len(Q)
    C, E = np.zeros((N, N)), np.zeros((N, N))
    for z, K in zip(Z, np.diff(c)):
        if not K:
            continue
        obs = (z.reshape(N, 2) != 255).sum(axis=1)
        D = obs[:, None] * Q
        H = D @ np.linalg.pinv(D, rcond=np.finfo(float).eps * max(D.shape))
        R = np.eye(N) - H
        X = np.array([(z.reshape(N, 2) == k).sum(axis=1) for k in range(K)], float).T
        fitted = H @ X
        v = np.zeros(N)
        for i in range(N):
            if obs[i]:
                for mean in fitted[i]:
                    if 0 < mean < obs[i]:
                        v[i] += mean * (1 - mean / obs[i]) / K
        C += (R @ X) @ (R @ X).T / K
        E += R @ np.diag(v) @ R.T
    return C, E


### Compare native covariance kernels with dense projections
class CovarianceTests(unittest.TestCase):
    def test_dense_complete_missing_and_rank_deficient_projections(self):
        for missing in (False, True):
            Z, c, obs, Q = fixture(missing)
            for kind in ("full", "singular", "nearly_singular"):
                q = Q.copy()
                if kind != "full":
                    q[:, 1] = q[:, 0] + (1e-8 * q[:, 2] if kind == "nearly_singular" else 0)
                    q /= q.sum(axis=1, keepdims=True)
                C, E = dense(Z, c, q)
                for chunk in (255, 4096):
                    actual = covariance([(Z, c, obs)], q, chunk)
                    for A, B in zip(actual, (C, E)):
                        np.testing.assert_allclose(A, B, atol=2e-12)
                        np.testing.assert_allclose(A, A.T, atol=2e-15)
                    for A, B in zip(actual, (C, E)):
                        np.testing.assert_allclose(
                            correlation(A), correlation(B.copy()), atol=3e-12
                        )

    def test_zero_information_and_no_residual_degrees_of_freedom(self):
        for missing in (False, True):
            Z, c, obs, _ = fixture(missing, N=3)
            Q = np.eye(3)
            C, E = covariance([(Z, c, obs)], Q)
            np.testing.assert_array_equal(C, 0)
            np.testing.assert_array_equal(E, 0)
        Z, c, obs = np.full((3, 8), 255, np.uint8), np.zeros(4, np.int64), np.zeros(3, np.int64)
        for A in covariance([(Z, c, obs)], np.full((4, 2), 0.5)):
            np.testing.assert_array_equal(correlation(A), 0)

    def test_shared_missingness_and_observed_runs_across_blocks(self):
        Z, c, _, Q = fixture(False)

        # Equal observed counts give the same fitted basis, regardless of which copy is missing.
        Z[[0, 3], 0] = 255
        Z[4, 1] = 255
        Z[5, 3:6] = 255
        obs = (Z != 255).sum(axis=1)
        for chunk in (255, 4096):
            got = covariance([(Z, c, obs)], Q, chunk)
            for A, B in zip(got, dense(Z, c, Q)):
                np.testing.assert_allclose(A, B, atol=2e-12)


### Check residual output, sample order, and failed writes
class EvaluationPipeline(TemporaryTests):
    def test_saved_q_ids_must_match_cluster_order(self):
        Z, c, _, Q = fixture(False)
        ref = writeClusters(self.root, "ref", Z, c)
        qfile = self.root / "swapped.Q"
        np.savetxt(qfile, Q)
        side = qfile.with_suffix(".ids")
        side.write_text("".join(f"s{i}\n" for i in reversed(range(len(Q)))))
        out = self.root / "order"
        res = command("eval", "--clusters", ref, "--qfile", qfile, "--out", out, success=False)
        self.assertIn("Q sample IDs differ", res.stderr)
        side.unlink()
        command("eval", "--clusters", ref, "--qfile", qfile, "--out", out)

    def test_parallel_text_blocks_match_numpy_rounding_and_subtraction(self):
        rng = np.random.default_rng(96)
        A = rng.uniform(-1, 1, (129, 513))
        x = (np.arange(-10000, 10000) + 0.5) / 10000
        A.flat[:60000] = np.r_[x, np.nextafter(x, -np.inf), np.nextafter(x, np.inf)]
        A.flat[-10:] = [0, -0.0, 2, -2, 3, -3, np.nan, np.inf, -np.inf, 1e-300]
        B = rng.uniform(-1, 1, A.shape)
        for sub in (None, B):
            ref = io.StringIO()
            np.savetxt(ref, A if sub is None else A - sub, fmt="%.4f")
            pth = self.root / "matrix"
            with pth.open("wb") as dst:
                writeMatrix(dst.fileno(), A, sub)
            self.assertEqual(pth.read_text(), ref.getvalue())
        with self.assertRaises(ValueError):
            writeMatrix(-1, A, B[:1])
        with self.assertRaises(OSError):
            writeMatrix(-1, A)

    def test_observed_subsets_match_dense_reference(self):
        Z, c, _, Q = fixture(False)
        idx = np.array([15, 6, 2, 19, 4])
        h = np.column_stack((2 * idx, 2 * idx + 1)).ravel()
        qfile, keep = self.root / "Q", self.root / "keep"
        np.savetxt(qfile, Q[idx])
        keep.write_text("".join(f"s{i}\n" for i in idx))
        C, E = dense(Z[:, h], c, Q[idx])
        for missing in (False, True):
            if missing:
                Z[:, :2] = 255
            ref = writeClusters(self.root, "ref", Z, c)
            out = self.root / "subset"
            command("eval", "--clusters", ref, "--qfile", qfile, "--keep", keep, "--out", out)
            np.testing.assert_allclose(
                np.loadtxt(f"{out}.bhat"), correlation(C.copy()), atol=5.1e-5
            )
            np.testing.assert_allclose(
                np.loadtxt(f"{out}.chat"), correlation(E.copy()), atol=5.1e-5
            )

    def test_threads_missing_windows_and_explicit_keep_order(self):
        Z, c, _, Q = fixture(True)
        ref = writeClusters(self.root, "ref", Z, c)
        qfile = self.root / "Q"
        np.savetxt(qfile, Q)
        expected = [correlation(A) for A in dense(Z, c, Q)]
        expected.append(expected[0] - expected[1])
        a = writeClusters(self.root, "a", Z[:2], c[:3])
        b = writeClusters(self.root, "b", Z[2:], c[2:] - c[2])
        files = self.root / "files"
        files.write_text(f"{a}\n{b}\n")
        outputs = []
        for i, opt in enumerate(
            (("--clusters", ref), ("--clusters", ref), ("--filelist", files), ("--clusters", a, b))
        ):
            out = self.root / f"eval{i}"
            command("eval", *opt, "--qfile", qfile, "--threads", 1 if i == 0 else 4, "--out", out)
            outputs.append([Path(f"{out}{s}").read_bytes() for s in (".bhat", ".chat", ".corres")])
            for s, A in zip((".bhat", ".chat", ".corres"), expected):
                np.testing.assert_allclose(np.loadtxt(f"{out}{s}"), A, atol=5.1e-5)
        self.assertEqual(outputs[0], outputs[1])
        self.assertEqual(outputs[2], outputs[3])
        idx = np.array([20, 7, 0, 2, 11, 9])
        keep = self.root / "keep"
        keep.write_text("".join(f"s{i}\n" for i in idx))
        np.savetxt(qfile, Q[idx])
        out = self.root / "subset"
        command("eval", "--clusters", ref, "--qfile", qfile, "--keep", keep, "--out", out)
        self.assertEqual(Path(f"{out}.ids").read_text(), keep.read_text())
        h = np.column_stack((idx * 2, idx * 2 + 1)).ravel()
        C, E = dense(Z[:, h], c, Q[idx])
        np.testing.assert_allclose(np.loadtxt(f"{out}.bhat"), correlation(C), atol=5.1e-5)
        np.testing.assert_allclose(np.loadtxt(f"{out}.chat"), correlation(E), atol=5.1e-5)

    def test_invalid_inputs_preserve_outputs_and_report_cli_errors(self):
        Z, c, _, Q = fixture(True)
        ref = writeClusters(self.root, "ref", Z, c)
        qfile = self.root / "Q"
        out = self.root / "eval"
        for s in (".bhat", ".chat", ".corres", ".log", ".ids"):
            Path(f"{out}{s}").write_text("previous")
        for bad in (Q * 2, Q * np.nan, Q[:1], Q[:, :1], -Q):
            np.savetxt(qfile, bad)
            result = command(
                "eval", "--clusters", ref, "--qfile", qfile, "--out", out, success=False
            )
            self.assertIn("error:", result.stderr)
            self.assertNotIn("Traceback", result.stderr)
            for s in (".bhat", ".chat", ".corres", ".log", ".ids"):
                self.assertEqual(Path(f"{out}{s}").read_text(), "previous")
        np.savetxt(qfile, Q)
        keep = self.root / "keep"
        keep.write_text("s0\nunknown\n")
        result = command(
            "eval", "--clusters", ref, "--qfile", qfile, "--keep", keep, "--out", out, success=False
        )
        self.assertIn("absent", result.stderr)
        result = command("eval", "--clusters", ref, "--qfile", qfile, "--out", ref, success=False)
        self.assertIn("conflicts", result.stderr)
