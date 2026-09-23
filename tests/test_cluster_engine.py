"""Independent full-distance reference and boundary contracts for packed clustering."""

__author__ = "Jonas Meisner"

import unittest

import numpy as np
from hapla.packed_cy import fit_window, likelihoods, plink_window, predict_haplotypes


### Fit weighted medians using a full distance matrix
def reference(G, alpha=0.1, min_freq=0.005, min_mac=None, K_max=255):
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
    minimum = min_mac or int(np.ceil(len(H) * min_freq))
    threshold = int(np.ceil(alpha * G.shape[0]))
    z = np.zeros(len(X), dtype=int)
    centers = [(2 * H.sum(0) > len(H)).astype(np.uint8)]
    active = np.ones(1, dtype=bool)

    def assignment():
        distance = np.stack([np.count_nonzero(X != r, axis=1) for r in centers], axis=1)
        distance[:, ~active] = G.shape[0] + 1
        new_z = len(centers) - 1 - np.argmin(distance[:, ::-1], axis=1)
        return new_z, distance[np.arange(len(X)), new_z]

    def recount():
        sizes = np.bincount(z, weights=weights, minlength=len(centers)).astype(np.uint64)
        sums = np.zeros((len(centers), G.shape[0]), dtype=np.uint64)
        for k in range(len(centers)):
            sums[k] = (X[z == k].astype(np.uint64) * weights[z == k, None]).sum(0)
        return sizes, sums

    def update(sizes, sums):
        new_active = sizes > 0
        changed = not np.array_equal(new_active, active)
        for k in np.flatnonzero(new_active):
            new = (2 * sums[k] > sizes[k]).astype(np.uint8)
            changed |= not np.array_equal(centers[k], new)
            centers[k] = new
        return new_active, changed

    for _ in range(1000):
        z, distances = assignment()
        candidates = np.flatnonzero(distances == distances.max())
        candidate = candidates[np.argmax(weights[candidates])]
        birth = distances[candidate] >= threshold and len(centers) < K_max
        if birth:
            z[candidate] = len(centers)
            centers.append(X[candidate].copy())
            active = np.append(active, True)
        sizes, sums = recount()
        active, changed = update(sizes, sums)
        if not birth and not changed:
            break
    else:
        raise AssertionError("Reference growth failed to converge")
    for _ in range(1000):
        rare = np.flatnonzero(active & (sizes < minimum))
        if not len(rare):
            # Complete any pending median movement after the last deletion.
            new_z, _ = assignment()
            if np.array_equal(new_z, z):
                break
        if len(rare):
            smallest = rare[::-1][np.argmin(sizes[rare[::-1]])]
            active[smallest] = False
        z, _ = assignment()
        sizes, sums = recount()
        active, changed = update(sizes, sums)
        if not changed and np.all(sizes[active] >= minimum):
            break
    else:
        raise AssertionError("Reference pruning failed to converge")
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
        result = fit_window(G, **options)
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
        full = fit_window(G, missing=True)
        fast = fit_window(G, missing=False)
        for key in ("labels", "medians", "sizes", "counts"):
            np.testing.assert_array_equal(full[key], fast[key])
        self.assertEqual(full["stats"], fast["stats"])

    def checkFlip(self, G, flip, **options):
        A = fit_window(G, **options)
        X = np.where(G == 255, 255, G ^ flip[:, None]).astype(np.uint8)
        B = fit_window(X, **options)
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
        np.testing.assert_array_equal(predict_haplotypes(G, A["medians"]), A["labels"])
        np.testing.assert_array_equal(predict_haplotypes(X, B["medians"]), B["labels"])

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
        A = fit_window(G, K_max=7, min_mac=3)
        B = fit_window(G[:, order].copy(), K_max=7, min_mac=3)
        np.testing.assert_array_equal(A["labels"][order], B["labels"])
        for key in ("medians", "counts", "sizes"):
            np.testing.assert_array_equal(A[key], B[key])
        self.checkFlip(G, rng.integers(0, 2, len(G), dtype=np.uint8), K_max=7, min_mac=3)

    def test_255_clusters_and_missing_sentinel(self):
        patterns = ((np.arange(255)[:, None] >> np.arange(8)) & 1).astype(np.uint8)
        G = np.concatenate(
            (np.repeat(patterns, 2, axis=0).T, np.full((8, 2), 255, np.uint8)), axis=1
        )
        result = fit_window(G, min_mac=1)
        self.assertEqual(result["stats"]["K"], 255)
        np.testing.assert_array_equal(np.unique(result["labels"][:-2]), np.arange(255))
        np.testing.assert_array_equal(result["labels"][-2:], [255, 255])
        self.assertFalse(result["stats"]["capped"])
        self.checkFlip(G, np.ones(8, np.uint8), min_mac=1)

    def test_cluster_cap_never_labels_observed_haplotypes_missing(self):
        G = ((np.arange(256)[:, None] >> np.arange(8)) & 1).astype(np.uint8).T.copy()
        result = fit_window(G, min_mac=1)
        self.assertTrue(result["stats"]["capped"])
        self.assertEqual(result["stats"]["K"], 255)
        self.assertTrue(np.all(result["labels"] < 255))
        self.checkFlip(G, np.array([0, 1] * 4, np.uint8), min_mac=1)

    def test_missing_windows_and_frequency_denominator(self):
        G = np.array([[0, 0, 1, 255], [0, 0, 1, 0]], np.uint8)
        result = self.checkReference(G, min_freq=0.3)
        self.assertEqual(result["stats"]["K"], 2)
        self.assertEqual(result["labels"][3], 255)
        empty = fit_window(np.full((8, 6), 255, np.uint8))
        self.assertEqual(empty["stats"]["K"], 0)
        self.assertTrue(np.all(empty["labels"] == 255))
        self.assertEqual(
            likelihoods(empty["medians"], empty["counts"], empty["sizes"]).shape, (0, 0)
        )

    def test_single_cluster_and_strict_minimum(self):
        result = self.checkReference(np.array([[0] * 99 + [1]] * 8, np.uint8), min_mac=10)
        self.assertEqual(result["stats"]["K"], 1)
        same = fit_window(np.zeros((16, 4), np.uint8), K_max=1)
        self.assertEqual(same["stats"]["K"], 1)

    def test_limits_and_input_validation(self):
        G = np.random.default_rng(44).integers(0, 2, (8, 100), dtype=np.uint8)
        for options in (
            {"K_max": 256},
            {"alpha": float("nan")},
            {"min_mac": 101},
            {"min_freq": 0},
            {"n_iter": 0},
        ):
            with self.assertRaises(ValueError):
                fit_window(G, **options)
        with self.assertRaisesRegex(RuntimeError, "Growth did not converge"):
            fit_window(G, n_iter=1)
        with self.assertRaises(ValueError):
            fit_window(np.full((2, 4), 2, np.uint8))
        with self.assertRaises(ValueError):
            fit_window(np.full((2, 4), 255, np.uint8), missing=False)

    def test_scores_from_counts_and_missing_plink(self):
        G = np.random.default_rng(87).integers(0, 2, (8, 40), dtype=np.uint8)
        r = fit_window(G, min_mac=2)
        p = np.clip(r["counts"] / r["sizes"][:, None], 1e-5, 1 - 1e-5)
        expected = r["medians"] @ np.log(p).T + (1 - r["medians"]) @ np.log1p(-p).T
        np.testing.assert_allclose(
            likelihoods(r["medians"], r["counts"], r["sizes"]), expected, rtol=1e-6, atol=1e-5
        )
        labels = np.array([0, 0, 0, 1, 1, 1, 255, 0], np.uint8)
        np.testing.assert_array_equal(plink_window(labels, 2), [[120], [75]])


if __name__ == "__main__":
    unittest.main()
