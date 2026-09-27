"""Independent full-distance reference and boundary contracts for packed clustering."""

__author__ = "Jonas Meisner"

import unittest

import numpy as np
from hapla.packed_cy import fitWindow, likelihoods, plinkWindow, predictHaplotypes


### Fit weighted medians using a full distance matrix
def reference(G, alpha=0.0625, min_freq=0.001, min_mac=5, K_max=255, reuse=True):
    observed = np.all(G != 255, axis=0)
    H = G[:, observed].T
    labels = np.full(G.shape[1], 255, dtype=np.uint8)
    if not len(H):
        return labels, np.empty((0, G.shape[0]), np.uint8), np.empty((0, G.shape[0]), np.uint64)
    total = 2 * H.sum(0)
    flip = ((total > len(H)) | ((total == len(H)) & (H[0] == 1))).astype(np.uint8)
    H = H ^ flip
    X, inverse, weights = np.unique(H[:, ::-1], axis=0, return_inverse=True, return_counts=True)
    X = X[:, ::-1]
    minimum = max(min_mac, int(np.ceil(len(H) * min_freq)))
    threshold = max(1, int(np.ceil(alpha * G.shape[0])))
    if threshold == 1 and len(X) <= K_max and weights.min() >= minimum:
        labels[observed] = inverse
        return labels, X ^ flip, (X ^ flip).astype(np.uint64) * weights[:, None]
    z = np.zeros(len(X), dtype=int)
    centers = [(2 * H.sum(0) > len(H)).astype(np.uint8)]
    active = np.ones(1, dtype=bool)

    def assignment():
        distance = np.stack([np.count_nonzero(X != r, axis=1) for r in centers], axis=1)
        distance[:, ~active] = G.shape[0] + 1
        new_z = len(centers) - 1 - np.argmin(distance[:, ::-1], axis=1)
        return new_z, distance[np.arange(len(X)), new_z]

    def refresh():
        sizes = np.bincount(z, weights=weights, minlength=len(centers)).astype(np.uint64)
        sums = np.array([(X[z == k] * weights[z == k, None]).sum(0) for k in range(len(centers))])
        new_active = sizes > 0
        changed = not np.array_equal(new_active, active)
        for k in np.flatnonzero(new_active):
            new = (2 * sums[k] > sizes[k]).astype(np.uint8)
            changed |= not np.array_equal(centers[k], new)
            centers[k] = new
        return sizes, sums, new_active, changed

    def candidate(d):
        eligible = np.flatnonzero(d >= threshold)
        if not len(eligible):
            return None

        def weighted(i):
            return (-int(weights[i] * d[i]), -int(d[i]), -int(weights[i]), i)

        short = set(
            sorted(eligible, key=weighted)[:2]
            + sorted(eligible, key=lambda i: (-int(d[i]), -int(weights[i]), i))[:2]
        )
        gains = {
            i: np.dot(weights, np.maximum(d - np.count_nonzero(X != X[i], axis=1), 0))
            for i in short
        }
        return min(short, key=lambda i: (-gains[i], *weighted(i)))

    def settle(prune):
        nonlocal z, sizes, sums, active
        for it in range(1000):
            rare = np.flatnonzero(active & (sizes < minimum)) if prune else []
            if len(rare):
                smallest = rare[::-1][np.argmin(sizes[rare[::-1]])]
                active[smallest] = False
            elif prune and it == 0:
                return
            z, _ = assignment()
            sizes, sums, active, changed = refresh()
            if not changed and (not prune or np.all(sizes[active] >= minimum)):
                return
        raise AssertionError("Reference refinement failed to converge")

    for _ in range(1000):
        z, distances = assignment()
        c = candidate(distances) if len(centers) < K_max else None
        if c is not None:
            z[c] = len(centers)
            centers.append(X[c].copy())
            active = np.append(active, True)
        sizes, sums, active, changed = refresh()
        if c is None and not changed:
            break
    else:
        raise AssertionError("Reference growth failed to converge")
    settle(True)
    free = np.flatnonzero(~active)
    if reuse and len(free):
        _, d = assignment()
        c = candidate(d)
        if c is not None:
            before = np.dot(weights, d)
            saved = (
                [r.copy() for r in centers],
                z.copy(),
                sizes.copy(),
                sums.copy(),
                active.copy(),
            )
            k = free[-1]
            centers[k], z[c], active[k] = X[c].copy(), k, True
            sizes, sums, active, _ = refresh()
            settle(False)
            settle(True)
            if np.dot(weights, assignment()[1]) >= before:
                centers, z, sizes, sums, active = saved
    kept = np.flatnonzero(active)
    mapping = np.zeros(len(centers), dtype=np.uint8)
    mapping[kept] = np.arange(len(kept))
    labels[observed] = mapping[z[inverse]]
    return (
        labels,
        np.asarray(centers)[kept] ^ flip,
        np.where(flip, sizes[kept, None] - sums[kept], sums[kept]),
    )


### Compare packed clustering with independent dense fits
class PackedClusteringTests(unittest.TestCase):
    def checkReference(self, G, **options):
        expected = reference(G, **options)
        result = fitWindow(G, **options)
        for value, key in zip(expected, ("labels", "medians", "counts")):
            np.testing.assert_array_equal(value, result[key])
        self.assertEqual(int(result["sizes"].sum()), np.count_nonzero(np.all(G != 255, axis=0)))
        return result

    def test_random_weighted_states_and_word_boundaries(self):
        rng = np.random.default_rng(98402)
        for B in (1, 2, 7, 8, 9, 15, 16, 17, 32, 63, 64, 65, 127, 128, 129, 255, 256, 257):
            for repeat in range(8):
                patterns = rng.integers(0, 2, (int(rng.integers(2, 25)), B), dtype=np.uint8)
                weights = rng.integers(1, 15, len(patterns))
                G = np.repeat(patterns, weights, axis=0).T.copy()
                if repeat % 2:
                    G[0, 0] = 255
                self.checkReference(G, min_mac=min(8, G.shape[1] - 1), K_max=16)

    def test_complete_fast_path_is_identical(self):
        G = np.random.default_rng(33).integers(0, 2, (65, 100), dtype=np.uint8)
        full = fitWindow(G, missing=True)
        fast = fitWindow(G, missing=False)
        for key in ("labels", "medians", "sizes", "counts"):
            np.testing.assert_array_equal(full[key], fast[key])
        self.assertEqual(full["stats"], fast["stats"])

    def checkFlip(self, G, flip, **options):
        A = fitWindow(G, **options)
        X = np.where(G == 255, 255, G ^ flip[:, None]).astype(np.uint8)
        B = fitWindow(X, **options)
        for key in ("labels", "sizes"):
            np.testing.assert_array_equal(A[key], B[key])
        self.assertEqual(A["stats"], B["stats"])
        np.testing.assert_array_equal(A["medians"] ^ flip, B["medians"])
        np.testing.assert_array_equal(
            np.where(flip, A["sizes"][:, None] - A["counts"], A["counts"]), B["counts"]
        )
        np.testing.assert_array_equal(
            likelihoods(A["medians"], A["counts"], A["sizes"]),
            likelihoods(B["medians"], B["counts"], B["sizes"]),
        )
        np.testing.assert_array_equal(predictHaplotypes(G, A["medians"]), A["labels"])
        np.testing.assert_array_equal(predictHaplotypes(X, B["medians"]), B["labels"])

    def test_allele_flips_with_weighted_ties_missingness_and_word_boundaries(self):
        rng = np.random.default_rng(716)
        for width in (1, 8, 63, 64, 65, 129):
            G = np.repeat(
                rng.integers(0, 2, (8, width), dtype=np.uint8), np.arange(1, 9), axis=0
            ).T.copy()
            G[::3] = np.tile([0, 1], 18)
            for missing in (False, True):
                if missing:
                    G[0, 0] = G[-1, 1] = 255
                for flip in (rng.integers(0, 2, width, dtype=np.uint8), np.ones(width, np.uint8)):
                    self.checkFlip(G, flip, min_mac=5, K_max=4, missing=missing)

    def test_all_small_window_recodings_and_single_cluster_ties(self):
        G = ((np.arange(16)[:, None] >> np.arange(4)) & 1).astype(np.uint8).T.copy()
        for cap in (1, 3, 16):
            for flip in G.T:
                self.checkFlip(G, flip, K_max=cap, min_mac=3)
        self.checkFlip(np.full((4, 6), 255, np.uint8), np.ones(4, np.uint8))

    def test_sample_order_does_not_affect_windows_without_balanced_sites(self):
        rng = np.random.default_rng(819)
        G = rng.integers(0, 2, (65, 41), dtype=np.uint8)
        order = rng.permutation(G.shape[1])
        A = fitWindow(G, K_max=7, min_mac=3)
        B = fitWindow(G[:, order].copy(), K_max=7, min_mac=3)
        np.testing.assert_array_equal(A["labels"][order], B["labels"])
        for key in ("medians", "counts", "sizes"):
            np.testing.assert_array_equal(A[key], B[key])
        self.checkFlip(G, rng.integers(0, 2, len(G), dtype=np.uint8), K_max=7, min_mac=3)

    def test_255_clusters_and_missing_sentinel(self):
        patterns = ((np.arange(255)[:, None] >> np.arange(8)) & 1).astype(np.uint8)
        G = np.concatenate(
            (np.repeat(patterns, 2, axis=0).T, np.full((8, 2), 255, np.uint8)), axis=1
        )
        result = fitWindow(G, min_mac=1)
        self.assertEqual(result["stats"]["K"], 255)
        np.testing.assert_array_equal(np.unique(result["labels"][:-2]), np.arange(255))
        np.testing.assert_array_equal(result["labels"][-2:], [255, 255])
        self.assertFalse(result["stats"]["capped"])
        self.checkFlip(G, np.ones(8, np.uint8), min_mac=1)

    def test_cluster_cap_never_labels_observed_haplotypes_missing(self):
        G = ((np.arange(256)[:, None] >> np.arange(8)) & 1).astype(np.uint8).T.copy()
        result = fitWindow(G, min_mac=1)
        self.assertTrue(result["stats"]["capped"])
        self.assertEqual(result["stats"]["K"], 255)
        self.assertTrue(np.all(result["labels"] < 255))
        self.checkFlip(G, np.array([0, 1] * 4, np.uint8), min_mac=1)

    def test_missing_windows_and_frequency_denominator(self):
        G = np.array([[0, 0, 1, 255], [0, 0, 1, 0]], np.uint8)
        result = self.checkReference(G, min_freq=0.3, min_mac=1)
        self.assertEqual(result["stats"]["K"], 2)
        self.assertEqual(result["labels"][3], 255)
        empty = fitWindow(np.full((8, 6), 255, np.uint8))
        self.assertEqual(empty["stats"]["K"], 0)
        self.assertTrue(np.all(empty["labels"] == 255))
        self.assertEqual(
            likelihoods(empty["medians"], empty["counts"], empty["sizes"]).shape, (0, 0)
        )

    def test_single_cluster_and_strict_minimum(self):
        result = self.checkReference(np.array([[0] * 99 + [1]] * 8, np.uint8), min_mac=10)
        self.assertEqual(result["stats"]["K"], 1)
        same = fitWindow(np.zeros((16, 4), np.uint8), K_max=1, min_mac=1)
        self.assertEqual(same["stats"]["K"], 1)

    def test_combined_support_uses_observed_haplotypes(self):
        for H, minor, expected in ((100, 4, 1), (100, 5, 2), (5001, 5, 1), (5001, 6, 2)):
            G = np.zeros((8, H + 1000), np.uint8)
            G[0, H - minor : H] = 1
            G[0, H:] = 255
            result = fitWindow(G)
            self.assertEqual(result["stats"]["K"], expected)
            self.assertTrue(np.all(result["sizes"] >= max(5, int(np.ceil(0.001 * H)))))
            self.assertTrue(np.all(result["labels"][H:] == 255))
        G = np.zeros((8, 5001), np.uint8)
        G[0, -5:] = 1
        self.assertEqual(fitWindow(G, min_freq=0)["stats"]["K"], 2)
        with self.assertRaisesRegex(ValueError, "exceeds the observed"):
            fitWindow(np.zeros((8, 4), np.uint8))

    def test_lambda_boundary_and_exact_pattern_shortcut(self):
        for B, alpha, K in ((16, 0.0625, 2), (17, 0.0625, 1), (129, 0, 2)):
            G = np.zeros((B, 10), np.uint8)
            G[0, 5:] = 1
            result = self.checkReference(G, alpha=alpha)
            self.assertEqual(result["stats"]["K"], K)
            self.assertEqual(result["stats"]["exact"], K == 2)
            if K == 2:
                self.assertEqual(result["stats"]["distance_pairs"], 0)
                self.assertEqual(result["stats"]["growth_passes"], 0)
                self.assertEqual(result["stats"]["distortion"], 0)
                self.checkFlip(G, np.ones(B, np.uint8), alpha=alpha)
            limited = self.checkReference(G, alpha=alpha, K_max=1)
            self.assertFalse(limited["stats"]["exact"])

    def test_neighborhood_gain_can_select_an_unsupported_seed(self):
        values = np.array([0, 15, 60, 56, 52])
        X = ((values[:, None] >> np.arange(8)) & 1).astype(np.uint8)
        G = np.repeat(X, [30, 4, 3, 3, 3], axis=0).T.copy()
        result = self.checkReference(G, K_max=2)
        # Pattern 15 has the highest count × distance, but 60 explains its neighbors.
        np.testing.assert_array_equal(result["medians"], X[[0, 2]])
        np.testing.assert_array_equal(result["sizes"], [30, 13])
        self.assertEqual(result["stats"]["distortion"], 22)

    def test_capacity_reuse_requires_a_completed_supported_improvement(self):
        for values, weights, accepted in (
            ([81, 128, 101, 103, 94, 187, 32, 196, 255, 144], [8, 3, 7, 6, 4, 8, 9, 3, 1, 6], 1),
            ([112, 71, 78, 127, 186, 132, 150, 225, 164, 228], [1, 2, 4, 3, 1, 3, 2, 2, 1, 4], 0),
        ):
            X = ((np.array(values)[:, None] >> np.arange(8)) & 1).astype(np.uint8)
            G = np.repeat(X, weights, axis=0).T.copy()
            before = reference(G, K_max=4, reuse=False)
            result = self.checkReference(G, K_max=4)
            self.assertEqual(result["stats"]["reuse_attempts"], 1)
            self.assertEqual(result["stats"]["reuse_accepts"], accepted)
            self.assertTrue(np.all(result["sizes"] >= 5))
            if accepted:
                loss = np.count_nonzero(G.T != before[1][before[0]])
                self.assertLess(result["stats"]["distortion"], loss)
            else:
                for value, key in zip(before, ("labels", "medians", "counts")):
                    np.testing.assert_array_equal(value, result[key])

    def test_limits_and_input_validation(self):
        G = np.random.default_rng(44).integers(0, 2, (8, 100), dtype=np.uint8)
        for options in (
            {"K_max": 256},
            {"alpha": float("nan")},
            {"min_mac": 101},
            {"min_mac": 2**65},
            {"min_freq": -0.1},
            {"n_iter": 0},
        ):
            with self.assertRaises(ValueError):
                fitWindow(G, **options)
        with self.assertRaisesRegex(RuntimeError, "Growth did not converge"):
            fitWindow(G, n_iter=1)
        with self.assertRaises(ValueError):
            fitWindow(np.full((2, 4), 2, np.uint8))
        with self.assertRaises(ValueError):
            fitWindow(np.full((2, 4), 255, np.uint8), missing=False)

    def test_scores_from_counts_and_missing_plink(self):
        G = np.random.default_rng(87).integers(0, 2, (8, 40), dtype=np.uint8)
        r = fitWindow(G, min_mac=2)
        p = np.clip(r["counts"] / r["sizes"][:, None], 1e-5, 1 - 1e-5)
        expected = r["medians"] @ np.log(p).T + (1 - r["medians"]) @ np.log1p(-p).T
        np.testing.assert_allclose(
            likelihoods(r["medians"], r["counts"], r["sizes"]), expected, rtol=1e-6, atol=1e-5
        )
        labels = np.array([0, 0, 0, 1, 1, 1, 255, 0], np.uint8)
        np.testing.assert_array_equal(plinkWindow(labels, 2), [[120], [75]])


if __name__ == "__main__":
    unittest.main()
