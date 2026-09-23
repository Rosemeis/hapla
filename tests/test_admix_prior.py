"""Pooled-frequency prior for the categorical ancestry M-step."""

__author__ = "Jonas Meisner"

import unittest

import numpy as np

from hapla import admix_cy as cy


### Check pooled-frequency updates and the regularized objective
class AdmixPriorTests(unittest.TestCase):
    def test_prior_rejects_invalid_window_offsets(self):
        P = np.full(8, 0.5)
        pool = np.full(4, 0.5)
        k = np.array([2, 2], dtype=np.uint32)
        for offsets in ([1, 4, 8], [0, 0, 8], [0, 3, 8], [0, 10, 8]):
            with self.subTest(offsets=offsets), self.assertRaisesRegex(ValueError, "dimensions"):
                cy.priorScore(P, pool, k, np.array(offsets, dtype=np.uint32), 2, 1.0)

    def test_prior_mstep_missing_projection_and_objective(self):
        Z = np.array(
            [
                [0, 0, 1, 0, 1, 2, 0, 1],
                [255, 255, 255, 255, 255, 255, 255, 255],
                [0, 255, 1, 1, 2, 2, 0, 2],
            ],
            dtype=np.uint8,
        )
        k = np.array([3, 3, 3], dtype=np.uint32)
        obs = (Z != 255).sum(axis=1).astype(np.uint32)
        pool = np.concatenate(
            [
                np.bincount(z[z != 255], minlength=3) / n if n else np.zeros(3)
                for z, n in zip(Z, obs)
            ]
        )
        rng = np.random.default_rng(27)
        mass = 2.0
        for K in (3, 5, 6):
            c = np.arange(4, dtype=np.uint32) * 3 * K
            P = np.concatenate([rng.dirichlet(np.ones(3), size=K).T for _ in Z]).ravel()
            Q = rng.dirichlet(np.ones(K), size=4)
            T = np.zeros_like(Q)
            pt = np.empty((2, 3 * K))
            qt = np.empty((2, 4, K))
            A = P.copy()
            cy.em(Z, P, A, Q, T, k, c, pt, qt, obs=obs, pool=pool, mass=mass)

            counts = np.zeros((len(pool), K))
            qcounts = np.zeros_like(Q)
            for w, z in enumerate(Z):
                for h, label in enumerate(z):
                    if label == 255:
                        continue
                    i = h // 2
                    x = P.reshape(-1, K)[3 * w + label] * Q[i]
                    x /= x.sum()
                    counts[3 * w + label] += x
                    qcounts[i] += x
            expected = np.empty_like(counts)
            for w in range(len(Z)):
                x = counts[3 * w : 3 * w + 3] + mass * pool[3 * w : 3 * w + 3, None]
                expected[3 * w : 3 * w + 3] = (
                    x / x.sum(axis=0) if obs[w] else P.reshape(-1, K)[3 * w : 3 * w + 3]
                )
            np.testing.assert_allclose(A.reshape(-1, K), expected, atol=2e-15)
            np.testing.assert_allclose(T, qcounts / Q, atol=2e-15)
            np.testing.assert_allclose(A.reshape(-1, K).reshape(3, 3, K).sum(axis=1), 1)

            projected = np.empty_like(Q)
            cy.em(Z, P, None, Q, projected, k, c, pt, qt, obs=obs, mass=mass)
            np.testing.assert_allclose(projected, qcounts / Q, atol=2e-15)

            expected_score = mass * sum(
                x * np.log(A.reshape(-1, K)[j, t] / x)
                for j, x in enumerate(pool)
                if x
                for t in range(K)
            )
            self.assertAlmostEqual(cy.priorScore(A, pool, k, c, K, mass), expected_score)

            qo = (Z != 255).reshape(len(Z), len(Q), 2).sum(axis=(0, 2), dtype=np.uint32)
            scale = 2 * len(pool) * len(Q)
            old = cy.likelihood(Z, P, Q, c, np.empty(len(Z)), obs) * scale
            old += cy.priorScore(P, pool, k, c, K, mass)
            p_cur, q_cur = P.copy(), Q.copy()
            for _ in range(3):
                p_new, q_new = np.empty_like(P), np.empty_like(Q)
                cy.em(Z, p_cur, p_new, q_cur, T, k, c, pt, qt, obs=obs, pool=pool, mass=mass)
                cy.accelQMiss(q_cur, q_new, T, qo)
                now = cy.likelihood(Z, p_new, q_new, c, np.empty(len(Z)), obs) * scale
                now += cy.priorScore(p_new, pool, k, c, K, mass)
                self.assertGreaterEqual(now, old - 1e-12)
                p_cur, q_cur, old = p_new, q_new, now

            with self.assertRaises(ValueError):
                cy.em(Z, P, A, Q, T, k, c, pt, qt, obs=obs, mass=mass)
            with self.assertRaises(ValueError):
                cy.priorScore(A, pool, k, c, K, -1)
            bad = pool.copy()
            bad[0] = np.nan
            with self.assertRaises(ValueError):
                cy.priorScore(A, bad, k, c, K, mass)
            bad[0] = 0.0
            with self.assertRaises(ValueError):
                cy.priorScore(A, bad, k, c, K, mass)


if __name__ == "__main__":
    unittest.main()
