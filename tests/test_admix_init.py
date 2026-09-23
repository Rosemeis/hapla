"""Source-seeking ancestry initialization on small-reference mixtures."""

__author__ = "Jonas Meisner"

import unittest
from itertools import permutations

import numpy as np

from hapla import functions


### Match inferred components to the simulated sources
def sourceScores(Q, K, n):
    means = Q[: K * n].reshape(K, n, K).mean(axis=1)
    order = max(permutations(range(K)), key=lambda p: sum(means[k, p[k]] for k in range(K)))
    return np.array([means[k, order[k]] for k in range(K)])


### Check source initialization with small and weak reference groups
class SourceInitTests(unittest.TestCase):
    def test_nearly_singular_anchors_are_rejected(self):
        x = np.column_stack((np.linspace(-1, 1, 15), 1e-7 * np.sin(np.arange(15))))
        with self.assertRaisesRegex(ValueError, "poorly conditioned"):
            functions._sourceQ(x, np.ones(2), 3)

    def test_small_sources_with_admixed_majority_and_outlier(self):
        rng = np.random.default_rng(604)
        K, n = 5, 12
        A = rng.normal(size=(K, 8))
        A[:, :4] *= 2
        A[:, 4:] *= 0.5
        true = np.r_[np.repeat(np.eye(K), n, axis=0), rng.dirichlet(np.ones(K) * 0.75, 140)]
        X = true @ A + 0.13 * rng.normal(size=(len(true), 8))
        X[0, 7] += 4
        V, S, _ = np.linalg.svd(X - X.mean(axis=0), full_matrices=False)
        Q = functions._sourceQ(V.astype(np.float32), S.astype(np.float32), K)
        self.assertTrue(np.all(np.isfinite(Q)))
        np.testing.assert_allclose(Q.sum(axis=1), 1, atol=2e-7)
        self.assertGreater(sourceScores(Q, K, n).min(), 0.75)

    def test_extra_components_resolve_weak_source(self):
        rng = np.random.default_rng(6)
        K, n = 5, 15
        A = np.zeros((K, 12))
        A[:4, :4] = np.eye(4) * 2.5
        A[4, :4] = A[3, :4]
        A[4, 4] = 0.35
        true = np.r_[np.repeat(np.eye(K), n, axis=0), rng.dirichlet(np.ones(K) * 0.8, 150)]
        X = true @ A + 0.12 * rng.normal(size=(len(true), 12))
        X[0, 5] += 4
        V, S, _ = np.linalg.svd(X - X.mean(axis=0), full_matrices=False)
        narrow = functions._sourceQ(V[:, : K - 1], S[: K - 1], K)
        wide = functions._sourceQ(V[:, : K + 3], S[: K + 3], K)
        self.assertGreater(sourceScores(wide, K, n).min(), 0.65)
        self.assertGreater(sourceScores(wide, K, n).min(), sourceScores(narrow, K, n).min() + 0.3)


if __name__ == "__main__":
    unittest.main()
