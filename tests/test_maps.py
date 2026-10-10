"""Genetic coordinates, variable-distance HMMs, and single-pulse dating."""

__author__ = "Jonas Meisner"

import gzip
import itertools
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np
from helpers import TemporaryTests, command, readLog, writeClusters
from test_fatash import transition
from test_sharing import exactMatches, exactSharing

from hapla import fatash_cy as cy
from hapla import sharing
from hapla.dating import maximize
from hapla.fatash import refine
from hapla.maps import coordinates, distances, readMap


### Enumerate paths and latent refreshes with a separate transition at every boundary
def exact(E, q, rates):
    W, K = E.shape
    states = np.array(list(itertools.product(range(K), repeat=W)))
    with np.errstate(divide="ignore"):
        scores = np.log(q[states[:, 0]]) + E[0, states[:, 0]]
        for w in range(1, W):
            T = transition(q, rates[w])
            scores += np.log(T[states[:, w], states[:, w - 1]]) + E[w, states[:, w]]
    ll = np.logaddexp.reduce(scores)
    prob = np.exp(scores - ll)
    G = np.array([[prob[states[:, w] == k].sum() for k in range(K)] for w in range(W)])
    C = G[0].copy()
    for w in range(1, W):
        T = transition(q, rates[w])
        for row, weight in zip(states, prob):
            if weight:
                C[row[w]] += weight * (-np.expm1(-rates[w])) * q[row[w]] / T[row[w], row[w - 1]]
    return G, C, ll, states, scores


### Check map parsing, coverage, and genetic sharing weights
class MapTests(TemporaryTests):
    def test_formats_units_bounds_and_plateaus(self):
        rows = [("chr1", 10, 30), ("chr1", 40, 60), ("chr2", 1, 101)]
        p = self.root / "map"
        p.write_text("#CHROM BP CM\n1 1 0\n1 20 1\n1 40 1\n1 60 3\n2 1 0\n2 101 4\n")
        g = readMap(p)
        np.testing.assert_allclose(coordinates(rows, g)[:, 1], [0.01, 0.02, 0.02])
        np.testing.assert_allclose(distances(rows, [(0, 2), (2, 3)], g), [0, 0.01, 0])
        with self.assertRaisesRegex(ValueError, "coverage"):
            coordinates([("1", 1, 61)], g)
        with self.assertRaisesRegex(ValueError, "absent"):
            coordinates([("3", 1, 2)], g)
        for header, body in (
            ("pos chr cM", "1 1 0\n60 1 3"),
            ("Chromosome Position(bp) Rate(cM/Mb) Map(cM)", "1 1 5 0\n1 60 5 3"),
        ):
            p = self.root / "map.gz"
            with gzip.open(p, "wt") as f:
                f.write(header + "\n" + body + "\n")
            np.testing.assert_array_equal(readMap(p)["1"], [[1, 0], [60, 3]])
        p = self.root / "invalid"
        for text in (
            "",
            "1 1 0",
            "1 1 0\n1 1 0",
            "1 2 0\n1 1 1",
            "1 1 1\n1 2 0",
            "1 nan 0",
            "1 1 nan",
            "1 1 -1",
            "1 .5 0",
            "1 1.5 0",
            "1 1 0\nwrong",
        ):
            p.write_text(text)
            with self.assertRaises(ValueError):
                readMap(p)

    def test_genetic_sharing_matches_dense_reference_and_uniform_scaling(self):
        rng = np.random.default_rng(719)
        W, H = 17, 12
        Z = rng.integers(0, 2, (W, H), dtype=np.uint8)
        rows = [("1", 10 + 10 * w, 16 + 10 * w, 6, 2, 2) for w in range(W)]
        ref = writeClusters(self.root, "labels", Z, np.arange(0, 2 * W + 1, 2))
        Path(f"{ref}.win").write_text(
            "#CHROM START END LENGTH SIZE K\n" + "".join(" ".join(map(str, r)) + "\n" for r in rows)
        )
        base = sharing.geometry(rows, np.ones(W, bool), 100)
        uniform = {"1": np.array([[1, 0.001], [1000, 1.0]])}
        geom = sharing.geometry(rows, np.ones(W, bool), 100, uniform)
        for a, b in zip(base[:3], geom[:3]):
            np.testing.assert_allclose(10 * a, b)
        np.testing.assert_array_equal(base[3], geom[3])
        gmap = {"1": np.array([[1, 0], [50, 0.3], [110, 0.3], [200, 2.0]])}
        x, left, right, cut = sharing.geometry(rows, np.ones(W, bool), 100, gmap)
        self.assertTrue(np.any(right == left))
        rec = exactMatches(Z, cut, np.ones(W, np.uint8), np.arange(H), H)
        C, cov = exactSharing(rec, H, x, left, right)
        c = np.arange(0, 2 * W + 1, 2, dtype=np.int64)
        ids = np.array([f"s{i}" for i in range(H // 2)])
        cache, actual, info = sharing.build(
            [ref], [(Z, c, None)], np.full(W, 2, np.uint8), ids, self.root, matches=H, gmap=gmap
        )
        np.testing.assert_allclose(actual, cov)
        C /= cov[:, None]
        B = C - C.mean(axis=0)
        np.testing.assert_allclose(sharing.product(cache, np.eye(H // 2)), B, atol=1e-13)
        self.assertEqual(info["distance_unit"], "Morgans")

        # The CLI must use the same map for both the kernel and its eigenvalues.
        path = self.root / "map"
        path.write_text("CHR BP CM\n1 1 0\n1 50 .3\n1 110 .3\n1 200 2\n")
        out = self.root / "genetic"
        command("struct", "--clusters", ref, "--hsm", "--hsm-svd", 2, "--map", path, "--out", out)
        target = (H // 2 - 1) * (B @ B.T) / np.sum(B * B)
        actual = np.fromfile(f"{out}.hsm.grm.bin", "<f4")
        np.testing.assert_allclose(actual, target[np.tril_indices(H // 2)], atol=1e-7)
        self.assertEqual(readLog(out)["Distance unit"], "Morgans")
        np.testing.assert_allclose(
            np.loadtxt(f"{out}.hsm.val"), np.linalg.eigvalsh(target)[-2:][::-1], rtol=1e-8
        )

    def test_zero_span_chromosomes_do_not_change_sharing(self):
        rng = np.random.default_rng(720)
        Z = rng.integers(0, 2, (17, 12), dtype=np.uint8)
        c = np.arange(0, 35, 2, dtype=np.int64)
        refs = [writeClusters(self.root, name, Z, c) for name in ("live", "flat")]
        pth = Path(f"{refs[1]}.win")
        pth.write_text(pth.read_text().replace("\n1 ", "\n2 "))
        ids = np.array([f"s{i}" for i in range(6)])
        gmap = {"1": np.array([[1, 0], [17, 1.6]]), "2": np.array([[1, 0], [17, 0]])}
        result = []
        for n in (1, 2):
            tmp = self.root / f"cache{n}"
            tmp.mkdir()
            cache, cov, info = sharing.build(
                refs[:n], [(Z, c, None)] * n, np.full(n * len(Z), 2, np.uint8), ids, tmp, gmap=gmap
            )
            self.assertEqual(info["chromosomes"], 1)
            result.append((sharing.product(cache, np.eye(6)), cov))
        for a, b in zip(*result):
            np.testing.assert_array_equal(a, b)


### Check posteriors, paths, refresh counts, and the direct forward scorer
class GeneticHMM(unittest.TestCase):
    def test_variable_steps_match_enumeration_and_forward_score(self):
        rng = np.random.default_rng(619)
        for K in (1, 2, 3):
            for dist in ([0.0, 0.003, 0, 0.04], [0.0, 0, 0, 0], [0.0, 1e-250, 0, 2.0]):
                q = rng.dirichlet(np.ones(K))
                P = rng.dirichlet([1, 1], (4, K)).transpose(0, 2, 1).reshape(-1, K)
                c = np.arange(0, 9, 2, dtype=np.int64)
                z = rng.integers(0, 2, (1, 4), dtype=np.uint8)
                z[0, 2] = 255
                use = np.ones(4, np.uint8)
                E = cy.emissions(z, cy.emissionTable(P.ravel(), c, K), c, use, K, 0, 4)
                target, count, ll, states, scores = exact(E[0], q, 30 * np.array(dist))
                G, C, L = cy.posterior(E, q[None], 30, resets=True, dist=dist)
                np.testing.assert_allclose(G[0], target, atol=2e-12)
                np.testing.assert_allclose(C[0], count, atol=2e-12)
                np.testing.assert_allclose(L, ll, atol=2e-12)
                D, prob, score = cy.posteriorDecode(E, q[None], 30, confidence=True, dist=dist)
                np.testing.assert_array_equal(D, G.argmax(axis=2))
                np.testing.assert_allclose(prob, G.max(axis=2), atol=2e-12)
                np.testing.assert_allclose(score, L)
                path = cy.viterbi(E, q[None], 30, dist=dist)[0]
                self.assertAlmostEqual(scores[np.all(states == path, axis=1)][0], scores.max())
                direct = cy.mapScore(z, P.ravel(), c, use, q[None], dist, 30)
                np.testing.assert_allclose(direct, L[:, 0], atol=2e-12)

    def test_zero_distance_underflow_falls_back_to_log_recursion(self):
        E = np.array([[0.0, -250], [0, -250], [0, -250], [-1000, 0], [-1000, 0]])
        q, dist = np.array([[0.5, 0.5]]), np.zeros(5)
        G, C, L = cy.posterior(E[None], q, 30, resets=True, dist=dist)
        target, counts, ll, _, _ = exact(E, q[0], dist)
        np.testing.assert_allclose(G[0], target, atol=1e-12)
        np.testing.assert_allclose(C[0], counts, atol=1e-12)
        np.testing.assert_allclose(L, ll, atol=1e-12)
        np.testing.assert_allclose(cy.posterior(E[None], q, 30, score=True, dist=dist)[2], L)
        with self.assertRaisesRegex(ValueError, "supported"):
            cy.posterior(np.array([[[0.0, -np.inf], [-np.inf, 0]]]), q, 30, dist=[0, 0])
        for d in ([0, -1], [0, np.nan], [0], [0, np.inf]):
            with self.assertRaises(ValueError):
                cy.posterior(E[None, :2], q, 30, dist=d)

    def test_uniform_distance_equivalence_and_map_refinement(self):
        rng = np.random.default_rng(519)
        H, W, K = 8, 19, 3
        Q = rng.dirichlet(np.ones(K), H // 2)
        Z = rng.integers(0, 2, (W, H), dtype=np.uint8)
        Z[3:6, 0] = 255
        c = np.arange(0, 2 * W + 1, 2, dtype=np.int64)
        P = rng.dirichlet([1, 1], (W, K)).transpose(0, 2, 1).ravel()
        use = np.ones(W, np.uint8)
        E = cy.emissions(Z.T.copy(), cy.emissionTable(P, c, K), c, use, K, 0, W)
        q, d = np.repeat(Q, 2, axis=0), np.full(W, 0.001)
        for actual, expected in zip(
            cy.posterior(E, q, 30, resets=True, dist=d), cy.posterior(E, q, 0.03, resets=True)
        ):
            np.testing.assert_allclose(actual, expected, atol=1e-12)
        np.testing.assert_array_equal(cy.viterbi(E, q, 30, dist=d), cy.viterbi(E, q, 0.03))
        d[3:6] = 0
        data = [(Z, c, use, [(0, W)], 4)]
        for bw in (False, True):
            args = Namespace(baum_welch=bw, p_prior=10, q_prior=10, iter=5, tole=0, loo=False)
            _, _, info = refine(data, [P], Q, [30.0], args, [d])
            self.assertTrue(np.all(np.diff([v["objective"] for v in info["history"]]) >= -1e-9))
        for a, b in ((0, W), (5, W)):
            E = cy.emissions(Z.T.copy(), cy.emissionTable(P, c, K), c, use, K, a, b)
            for t in (10, 30):
                direct = cy.mapScore(
                    Z[a:b].T.copy(),
                    P[c[a] * K : c[b] * K],
                    c[a : b + 1] - c[a],
                    use[a:b],
                    q,
                    d[a:b],
                    t,
                )
                np.testing.assert_allclose(
                    direct, cy.posterior(E, q, t, dist=d[a:b], score=True)[2][:, 0], atol=1e-12
                )

    def test_date_bounds_and_nonidentifiability(self):
        t, _, status = maximize(lambda x: -((x - np.log(27)) ** 2), 1, 500)
        self.assertAlmostEqual(t, 27, delta=0.01)
        self.assertEqual(status, "converged")
        for fun, stop in ((lambda x: x, "upper_bound"), (lambda x: -x, "lower_bound")):
            self.assertEqual(maximize(fun, 1, 500)[2], stop)
        self.assertEqual(maximize(lambda x: 1, 1, 500), (None, 1, "flat"))

    def test_direct_score_handles_empty_frequencies_and_zero_priors(self):
        q = np.array([[1.0, 0.0], [0.5, 0.5]])
        Z = np.full((2, 3), 255, np.uint8)
        score = cy.mapScore(
            Z, np.empty(0), np.zeros(4, np.int64), np.ones(3, np.uint8), q, [0, 0, 0.1], 30
        )
        np.testing.assert_allclose(score, 0, atol=1e-14)
        with self.assertRaisesRegex(ValueError, "requires genetic distances"):
            cy.mapScore(Z, np.empty(0), np.zeros(4, np.int64), np.ones(3, np.uint8), q, None, 30)
        with self.assertRaisesRegex(ValueError, "window dimensions"):
            cy.mapScore(
                Z[:, :0].copy(),
                np.empty(0),
                np.zeros(1, np.int64),
                np.empty(0, np.uint8),
                q,
                [],
                30,
            )

    def test_direct_score_log_fallback_and_batch_equivalence(self):
        E = np.array([[0.0, -250]] * 3 + [[-500.0, 0]] * 2)
        p = np.exp(E)
        P = np.stack((p, 1 - p), axis=1).ravel()
        Z = np.zeros((2, 5), np.uint8)
        c, use, d = np.arange(0, 11, 2, dtype=np.int64), np.ones(5, np.uint8), np.zeros(5)
        q = np.array([[0.5, 0.5], [1.0, 0.0]])
        actual = cy.mapScore(Z, P, c, use, q, d, 30)
        for i in range(2):
            self.assertAlmostEqual(actual[i], exact(E, q[i], d)[2], places=10)
            np.testing.assert_array_equal(
                actual[i : i + 1], cy.mapScore(Z[i : i + 1], P, c, use, q[i : i + 1], d, 30)
            )


### Exercise complete map-based analyses and simulated date recovery
class GeneticPipeline(TemporaryTests):
    def fixture(self):
        rng = np.random.default_rng(20260929)
        N, W, K, t = 24, 1800, 3, 30
        Q = rng.dirichlet([4] * K, N)
        q = np.repeat(Q, 2, axis=0)
        a = np.zeros(2 * N, int)
        Z = np.empty((W, 2 * N), np.uint8)
        for w in range(W):
            reset = (
                np.ones(2 * N, bool) if w % 600 == 0 else rng.random(2 * N) < -np.expm1(-t * 0.001)
            )
            fresh = (rng.random(2 * N)[:, None] > q.cumsum(axis=1)).sum(axis=1)
            a[reset] = fresh[reset]
            Z[w] = a
            noise = rng.random(2 * N) < 0.03
            Z[w, noise] = rng.integers(0, K, noise.sum())
        Z[rng.random(Z.shape) < 0.015] = 255
        c = np.arange(0, K * W + 1, K, dtype=np.int64)
        ref = writeClusters(self.root, "input", Z, c)
        rows = [(str(1 + w // 600), 1 + w % 600, 1 + w % 600, 0, 1, K) for w in range(W)]
        Path(f"{ref}.win").write_text(
            "#CHROM START END LENGTH SIZE K\n" + "".join(" ".join(map(str, r)) + "\n" for r in rows)
        )
        P = np.tile(0.97 * np.eye(K) + 0.03 / K, (W, 1))
        np.savetxt(f"{ref}.P", P)
        np.savetxt(f"{ref}.Q", Q)
        g = self.root / "map"
        g.write_text("#CHROM BP CM\n" + "".join(f"{j} 1 0\n{j} 600 59.9\n" for j in (1, 2, 3)))
        return ref, g, t

    def test_dating_self_reuse_threads_jackknife_and_map_rescaling(self):
        ref, g, t = self.fixture()
        opts = ["--clusters", ref, "--pfile", f"{ref}.P", "--qfile", f"{ref}.Q"]
        out, again = self.root / "dated", self.root / "again"
        command(
            "fatash",
            *opts,
            "--map",
            g,
            "--dating",
            "--date-jackknife",
            "--save-posteriors",
            "--out",
            out,
        )
        values = Path(f"{out}.date").read_text().splitlines()
        result = dict(zip(values[0].split(), values[1].split()))
        fitted = float(result["generations"])
        self.assertNotIn("Ensemble", readLog(out))
        self.assertEqual(readLog(out)["Decoding"], "posterior")
        self.assertAlmostEqual(fitted, t, delta=3)
        self.assertEqual(result["status"], "converged")
        self.assertTrue(float(result["lower"]) < fitted < float(result["upper"]))
        command(
            "fatash",
            *opts,
            "--map",
            g,
            "--time",
            fitted,
            "--fixed-model",
            "--save-posteriors",
            "--threads",
            4,
            "--buffer-mb",
            1,
            "--out",
            again,
        )
        np.testing.assert_array_equal(np.loadtxt(f"{out}.path"), np.loadtxt(f"{again}.path"))
        np.testing.assert_allclose(
            np.loadtxt(f"{out}.prob"), np.loadtxt(f"{again}.prob"), atol=1e-8
        )
        g.write_text(g.read_text().replace("59.9", "119.8"))
        command(
            "fatash",
            *opts,
            "--map",
            g,
            "--dating",
            "--date-min",
            0.5,
            "--date-max",
            250,
            "--threads",
            4,
            "--out",
            again,
        )
        self.assertAlmostEqual(float(readLog(again)["Generations"]) * 2, fitted, delta=0.02)
        np.testing.assert_allclose(np.loadtxt(f"{out}.Q"), np.loadtxt(f"{ref}.Q"), atol=1e-9)

    def test_map_option_validation(self):
        ref, g, _ = self.fixture()
        opts = ["--clusters", ref, "--pfile", f"{ref}.P", "--qfile", f"{ref}.Q"]
        for flags in (
            ["--map", g],
            ["--time", 20],
            ["--dating"],
            ["--map", g, "--time", 0],
            ["--map", g, "--time", 20, "--alpha", 0.1],
            ["--map", g, "--dating", "--loo"],
            ["--map", g, "--dating", "--baum-welch"],
            ["--map", g, "--time", 20, "--simple"],
            ["--map", g, "--time", 20, "--block", 2],
            ["--date-min", 2],
            ["--map", g, "--dating", "--date-min", 501],
        ):
            res = command("fatash", *opts, *flags, "--out", self.root / "bad", success=False)
            self.assertIn("error:", res.stderr)

    def test_target_selection_fixed_time_refinement_and_stale_dates(self):
        ref, g, _ = self.fixture()
        opts = ["--clusters", ref, "--pfile", f"{ref}.P", "--qfile", f"{ref}.Q", "--map", g]
        ids, out = self.root / "targets", self.root / "target-fit"
        ids.write_text("s0\ns3\ns2\ns7\n")
        command("fatash", *opts, "--dating", "--date-samples", ids, "--out", out)
        self.assertEqual(np.loadtxt(f"{out}.path").shape[0], 48)
        row = Path(f"{out}.date").read_text().splitlines()
        self.assertEqual(dict(zip(row[0].split(), row[1].split()))["samples"], "4")
        before = Path(f"{out}.path").read_bytes()
        for text in ("unknown\n", "s0\ns0\n", "s0 s3\n", ""):
            ids.write_text(text)
            res = command(
                "fatash", *opts, "--dating", "--date-samples", ids, "--out", out, success=False
            )
            self.assertIn("error:", res.stderr)
            self.assertEqual(before, Path(f"{out}.path").read_bytes())
        for flag in ([], ["--baum-welch"], ["--loo"]):
            command("fatash", *opts, "--time", 30, "--iter", 2, *flag, "--out", out)
            self.assertFalse(Path(f"{out}.date").exists())
            self.assertEqual(readLog(out)["Iterations"], "2")

        # A flat likelihood must preserve the previous result set.
        before = Path(f"{out}.path").read_bytes()
        g.write_text(g.read_text().replace("59.9", "0"))
        res = command("fatash", *opts, "--dating", "--out", out, success=False)
        self.assertIn("not identifiable", res.stderr)
        self.assertEqual(before, Path(f"{out}.path").read_bytes())
