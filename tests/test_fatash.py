"""Independent HMM enumeration, dense recursions, refinement, and streaming contracts."""

__author__ = "Jonas Meisner"

import itertools
import unittest
from argparse import Namespace
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import numpy as np
from helpers import TemporaryTests, command, readLog, writeClusters

from hapla import fatash_cy as cy
from hapla.fatash import refine
from hapla.formats import readMetadata


### Construct the dense haploid transition matrix
def transition(q, alpha, simple=False):
    e, s = np.exp(-alpha), -np.expm1(-alpha)
    T = e * np.eye(len(q)) + s * q[:, None]
    if simple:
        np.fill_diagonal(T, e)
        if len(q) == 1:
            T[:] = 1
        else:
            T /= T.sum(axis=0)
    return T


### Enumerate state paths and latent refresh events for short chains
def enumerateHMM(E, q, alpha, simple=False):
    W, K = E.shape
    states = np.array(list(itertools.product(range(K), repeat=W)))
    T = transition(q, alpha, simple)
    with np.errstate(divide="ignore"):
        scores = np.log(q[states[:, 0]]) + E[0, states[:, 0]]
        for w in range(1, W):
            scores += np.log(T[states[:, w], states[:, w - 1]]) + E[w, states[:, w]]
    ll = np.logaddexp.reduce(scores)
    weights = np.exp(scores - ll)
    G = np.array([[weights[states[:, w] == k].sum() for k in range(K)] for w in range(W)])
    C = G[0].copy()
    if not simple:
        for row, weight in zip(states, weights):
            if weight == 0:
                continue
            for w in range(1, W):
                C[row[w]] += weight * (-np.expm1(-alpha)) * q[row[w]] / T[row[w], row[w - 1]]
    return G, C, ll, states, scores


### Dense log-space reference independent of the structured transition algebra
def denseHMM(E, q, alpha):
    K, W = len(q), len(E)
    ls = np.log(-np.expm1(-alpha))
    with np.errstate(divide="ignore"):
        lq = np.log(q)
    T = np.broadcast_to(ls + lq[:, None], (K, K)).copy()
    np.fill_diagonal(T, np.logaddexp(-alpha, ls + lq))
    F, B = np.empty_like(E), np.zeros_like(E)
    F[0] = E[0] + lq
    for w in range(1, W):
        F[w] = E[w] + np.logaddexp.reduce(T + F[w - 1], axis=1)
    for w in range(W - 2, -1, -1):
        B[w] = np.logaddexp.reduce(T + (E[w + 1] + B[w + 1])[:, None], axis=0)
    ll = np.logaddexp.reduce(F[-1])
    return np.exp(F + B - ll), ll


### Compare HMM inference with enumerated and dense references
class HMMCorrectness(unittest.TestCase):
    def test_viterbi_dense_reference_across_kernel_boundary(self):
        rng = np.random.default_rng(918)
        for K in (1, 4, 5, 6, 8, 9, 20, 255):
            for simple in (False, True):
                W = 193
                q = rng.dirichlet(np.ones(K))
                E = np.log(rng.uniform(0.001, 1, (W, K)))

                # Cross periodic score recentering and large block log offsets.
                E[97] -= 1e6
                with np.errstate(divide="ignore"):
                    T = np.log(transition(q, 0.1, simple))
                f = E[0] + np.log(q)
                prev = np.empty((W, K), int)
                for w in range(1, W):
                    score = T + f
                    prev[w] = score.argmax(axis=1)
                    f = E[w] + score.max(axis=1)
                    f -= f.max()
                path = np.empty(W, np.uint8)
                path[-1] = f.argmax()
                for w in range(W - 2, -1, -1):
                    path[w] = prev[w + 1, path[w + 1]]
                np.testing.assert_array_equal(cy.viterbi(E[None], q[None], 0.1, simple)[0], path)

    def test_full_ancestry_range_and_score_only_recursion(self):
        rng = np.random.default_rng(69)
        Q = rng.dirichlet(np.ones(255), 2)
        E = np.log(rng.uniform(0.001, 1, (2, 7, 255)))
        G, _, L = cy.posterior(E, Q, 0.1)
        empty, counts, score = cy.posterior(E, Q, 0.1, score=True)
        self.assertEqual(empty.size + counts.size, 0)
        np.testing.assert_array_equal(score, L)
        for i in range(2):
            exact, ll = denseHMM(E[i], Q[i], 0.1)
            np.testing.assert_allclose(G[i], exact, atol=2e-14)
            self.assertAlmostEqual(L[i, 0], ll, places=12)
        self.assertTrue(np.all(cy.viterbi(E, Q, 0.1) < 255))

    def test_enumerated_posteriors_paths_likelihood_and_reset_counts(self):
        rng = np.random.default_rng(143)
        for K, W, alpha, simple in itertools.product(
            (1, 2, 3), (1, 2, 4), (1e-15, 0.03, 1.5), (False, True)
        ):
            q = rng.dirichlet(np.ones(K))
            E = np.log(rng.uniform(0.005, 1, (W, K)))
            exact, count, ll, states, scores = enumerateHMM(E, q, alpha, simple)
            G, C, L = cy.posterior(E[None], q[None], alpha, simple, resets=not simple)
            np.testing.assert_allclose(G[0], exact, atol=2e-13)
            np.testing.assert_allclose(L, ll, atol=2e-13)
            if not simple:
                np.testing.assert_allclose(C[0], count, atol=2e-13)
            D = cy.viterbi(E[None], q[None], alpha, simple)
            selected = np.flatnonzero(np.all(states == D[0], axis=1))[0]
            self.assertAlmostEqual(scores[selected], scores.max(), places=12)

    def test_ensemble_is_mean_of_complete_posteriors_and_resets(self):
        rng = np.random.default_rng(7)
        E = np.log(rng.uniform(0.01, 1, (4, 17, 3)))
        Q = rng.dirichlet(np.ones(3), 4)
        alpha = np.array([1e-5, 0.02, 5.0])
        G, C, L = cy.posterior(E, Q, alpha, resets=True)
        parts = [cy.posterior(E, Q, a, resets=True) for a in alpha]
        np.testing.assert_allclose(G, np.mean([p[0] for p in parts], axis=0), atol=2e-15)
        np.testing.assert_allclose(C, np.mean([p[1] for p in parts], axis=0), atol=2e-15)
        np.testing.assert_allclose(G.sum(axis=2), 1, atol=2e-15)
        np.testing.assert_allclose(L, np.column_stack([p[2][:, 0] for p in parts]), atol=1e-13)
        D = cy.viterbi(E, Q, np.full(257, 0.03))
        np.testing.assert_array_equal(D, cy.viterbi(E, Q, 0.03))

    def test_direct_decode_matches_mean_posterior(self):
        rng = np.random.default_rng(31)
        E = np.log(rng.uniform(0.001, 1, (4, 23, 5)))
        Q = rng.dirichlet(np.ones(5), 4)
        alpha = np.array([1e-9, 1e-5, 0.1])
        for simple in (False, True):
            G, _, L = cy.posterior(E, Q, alpha, simple)
            D, P, score = cy.posteriorDecode(E, Q, alpha, simple, True)
            np.testing.assert_array_equal(D, G.argmax(axis=2))
            np.testing.assert_array_equal(P, G.max(axis=2))
            np.testing.assert_array_equal(score, L)
            path, empty, score = cy.posteriorDecode(E, Q, alpha, simple)
            np.testing.assert_array_equal(path, D)
            self.assertEqual(empty.size, 0)
            np.testing.assert_array_equal(score, L)

    def test_extreme_log_ranges_zero_priors_and_long_neutral_chains(self):
        cases = [
            (np.array([[-1000.0, 0], [0, -1000.0], [-1000.0, 0]]), [0.8, 0.2], 1e-250),
            (np.array([[-10000.0, 0], [-12000.0, 0]]), [1.0, 0], 0.01),
            (np.array([[-600.0, 0], [0, -600.0], [-600.0, 0]]), [1.0, 1e-250], 0.01),
            (np.array([[0.0, -np.inf], [-np.inf, 0.0]]), [0.3, 0.7], 0.1),
        ]
        for E, q, alpha in cases:
            Q = np.array([q])
            G, C, L = cy.posterior(E[None], Q, alpha, resets=True)
            _, count, _, _, _ = enumerateHMM(E, Q[0], alpha)
            np.testing.assert_allclose(C[0], count, atol=2e-10)
            exact, ll = denseHMM(E, Q[0], alpha)
            np.testing.assert_allclose(G[0], exact, atol=2e-10)
            np.testing.assert_allclose(L, ll, atol=2e-10)
            D, P, score = cy.posteriorDecode(E[None], Q, alpha, confidence=True)
            np.testing.assert_array_equal(D, G.argmax(axis=2))
            np.testing.assert_allclose(P, G.max(axis=2), atol=2e-10)
            np.testing.assert_array_equal(score, L)
        Q = np.array([[0.7, 0.2, 0.1], [1, 0, 0]])
        E = np.zeros((2, 10000, 3))
        G, _, L = cy.posterior(E, Q, [1e-20, 0.1, 1000.0])
        np.testing.assert_allclose(G, np.broadcast_to(Q[:, None], E.shape), atol=3e-14)
        np.testing.assert_allclose(L, 0, atol=2e-10)
        with self.assertRaisesRegex(ValueError, "supported"):
            cy.posterior(np.full((1, 2, 2), -np.inf), np.array([[0.5, 0.5]]), 0.1)
        with self.assertRaisesRegex(ValueError, "supported"):
            cy.viterbi(np.full((1, 2, 2), -np.inf), np.array([[0.5, 0.5]]), 0.1)

    def test_scaled_likelihood_accumulation_on_long_forced_paths(self):
        W = 1201
        alpha = np.array([1e-125, 0.1])
        for K in (5, 6):
            q = np.arange(1, K + 1, dtype=float)
            q /= q.sum()
            path = np.arange(W) % K
            shift = -np.arange(W) / 20
            E = np.full((1, W, K), -np.inf)
            E[0, np.arange(W), path] = shift
            G, C, L = cy.posterior(E, q[None], alpha, resets=True)
            exact = np.eye(K)[path]
            ll = shift.sum() + np.log(q[path]).sum()
            ll += (W - 1) * np.log(-np.expm1(-alpha))
            np.testing.assert_allclose(G[0], exact, atol=2e-12)
            np.testing.assert_allclose(C[0], exact.sum(axis=0), atol=2e-10)
            np.testing.assert_allclose(L[0], ll, rtol=2e-14, atol=2e-9)
            D, P, score = cy.posteriorDecode(E, q[None], alpha, confidence=True)
            np.testing.assert_array_equal(D[0], path)
            np.testing.assert_allclose(P, 1, atol=2e-12)
            np.testing.assert_array_equal(score, L)
            _, _, score = cy.posterior(E, q[None], alpha, score=True)
            np.testing.assert_array_equal(score, L)

    def test_soft_emissions_use_each_candidate_and_stable_likelihoods(self):
        P, c = np.array([0.9, 0.2, 0.1, 0.8]), np.array([0, 2], np.int64)
        likes = np.log(np.array([0.8, 0.2, 0.1, 0.9], np.float32))
        expected = np.array([[0.74, 0.32], [0.18, 0.74]])
        table = cy.emissionTable(P, c, 2, likes)
        np.testing.assert_allclose(np.exp(table.reshape(2, 2)), expected, rtol=2e-8)
        table = cy.emissionTable(P, c, 2, np.full(4, -1000, np.float32))
        np.testing.assert_allclose(np.exp(table), 0.5, atol=2e-15)
        with self.assertRaisesRegex(ValueError, "finite support"):
            cy.emissionTable(P, c, 2, np.full(4, -np.inf, np.float32))

        # A tiny candidate likelihood must survive even when another ancestry has zero frequency.
        rare = cy.emissionTable(
            np.array([1.0, 0.0, 0.0, 1.0]), c, 2, np.array([0.0, -1000.0, -1000.0, 0.0], np.float32)
        )
        np.testing.assert_allclose(rare, [0.0, -1000.0, -1000.0, 0.0], atol=1e-12)
        Z = np.array([[0], [255]], np.uint8)
        E = cy.emissions(Z, table, c, np.ones(1, np.uint8), 2, 0, 1)
        np.testing.assert_allclose(E[0, 0], np.log(0.5))
        np.testing.assert_array_equal(E[1], 0)

    def test_hard_emissions_blocks_missingness_and_mask(self):
        rng = np.random.default_rng(49)
        k = np.array([2, 0, 3, 2, 1], np.int64)
        c = np.r_[0, np.cumsum(k)]
        P = np.concatenate([rng.dirichlet(np.ones(C), 3).T for C in k if C]).ravel()
        Z = np.array([[0, 255, 2, 0, 0], [1, 255, 255, 1, 0]], np.uint8)
        use = np.array([1, 1, 1, 0, 1], np.uint8)
        table = cy.emissionTable(P, c, 3)
        for block in (1, 2):
            expected = np.zeros((2, (5 + block - 1) // block, 3))
            for i in range(2):
                for w in range(5):
                    if use[w] and Z[i, w] != 255:
                        expected[i, w // block] += np.log(P.reshape(-1, 3)[c[w] + Z[i, w]])
            np.testing.assert_allclose(cy.emissions(Z, table, c, use, 3, 0, 5, block), expected)

    def test_joint_em_step_matches_enumeration_with_missingness_and_priors(self):
        rng = np.random.default_rng(105)
        K, N = 3, 3
        Q = rng.dirichlet(np.ones(K), N)
        data, P, cp, cq = [], [], [], np.zeros_like(Q)
        ll = 0.0
        alpha = np.array([0.01, 0.7])
        n = 0
        for W, regions in ((3, [(0, 3)]), (4, [(0, 2), (2, 4)])):
            sizes = [2, 2, 2] if W == 3 else [2, 0, 255, 2]
            c = np.r_[0, np.cumsum(sizes, dtype=np.int64)]
            p = np.concatenate([rng.dirichlet(np.ones(C), K).T for C in sizes if C])
            Z = np.array(
                [rng.integers(0, C, 2 * N) if C else np.full(2 * N, 255) for C in sizes], np.uint8
            )
            if W == 4:
                Z[2, -1] = 254
            Z[:, :2] = 255
            Z[0, 2] = 255
            use = np.ones(W, np.uint8)
            use[1] = 0
            n += int(np.sum((Z != 255) & use[:, None]))
            count = np.zeros_like(p)
            for h in range(2 * N):
                for beg, end in regions:
                    E = np.array(
                        [
                            np.log(p[c[w] + Z[w, h]]) if use[w] and Z[w, h] != 255 else np.zeros(K)
                            for w in range(beg, end)
                        ]
                    )
                    for a in alpha:
                        G, C, L, _, _ = enumerateHMM(E, Q[h // 2], a)
                        ll += L / len(alpha)
                        cq[h // 2] += C / len(alpha)
                        for w in range(beg, end):
                            if use[w] and Z[w, h] != 255:
                                count[c[w] + Z[w, h]] += G[w - beg] / len(alpha)
            data.append((Z, c, use, regions, 2))
            P.append(p.ravel())
            cp.append(count)
        for bw, tp, tq in ((True, 17, 29), (False, 0, 0), (False, 3, 7)):
            args = Namespace(baum_welch=bw, p_prior=tp, q_prior=tq, iter=1, tole=0)
            pp, qq, info = refine(data, P, Q, alpha, args)
            tp, tq = (0, 0) if bw else (tp, tq)
            expected = (cq + tq * Q) / (cq.sum(axis=1, keepdims=True) + tq)
            expected[0] = Q[0]
            np.testing.assert_allclose(qq, expected, atol=2e-14)
            self.assertEqual(info["observed_assignments"], n)
            self.assertEqual(info["unobserved_samples"], 1)
            self.assertEqual(info["stop"], "iteration_limit")
            self.assertAlmostEqual(info["history"][0]["log_likelihood"], ll, places=12)
            penalty = tq * np.sum(Q * np.log(qq / Q))
            for p, base, count, (_, c, use, _, _) in zip(pp, P, cp, data):
                expected = base.reshape(-1, K).copy()
                for w in range(len(use)):
                    C = count[c[w] : c[w + 1]]
                    if use[w]:
                        expected[c[w] : c[w + 1]] = (C + tp * expected[c[w] : c[w + 1]]) / (
                            C.sum(axis=0) + tp
                        )
                np.testing.assert_allclose(p.reshape(-1, K), expected, atol=2e-14)
                if tp:
                    penalty += tp * np.sum(base * np.log(p / base))
            last = info["history"][-1]
            self.assertAlmostEqual(last["objective"], last["log_likelihood"] + penalty, places=12)
            self.assertGreater(last["objective"], info["history"][0]["objective"])

    def test_convergence_empty_data_and_rejected_proposal(self):
        c = np.array([0, 2], np.int64)
        Z = np.zeros((1, 2), np.uint8)
        use = np.ones(1, np.uint8)
        data = [(Z, c, use, [(0, 1)], 2)]
        P = [np.array([0.8, 0.2, 0.2, 0.8])]
        Q = np.array([[0.9, 0.1]])
        args = Namespace(baum_welch=False, p_prior=10, q_prior=10, iter=50, tole=1e-6)
        pp, qq, info = refine(data, P, Q, [0.1], args)
        self.assertEqual(info["stop"], "converged")
        self.assertLess(info["iterations"], 50)
        self.assertTrue(np.all(np.diff([r["objective"] for r in info["history"]]) >= -1e-12))

        # An invalid M-step must never replace the last accepted model.
        with patch.object(cy, "refineP", return_value=np.array([0.01, 0.01, 0.99, 0.99])):
            pp, qq, info = refine(data, P, Q, [0.1], args)
        np.testing.assert_array_equal(pp[0], P[0])
        np.testing.assert_array_equal(qq, Q)
        self.assertEqual(info["stop"], "objective_decreased")
        self.assertEqual(info["iterations"], 0)
        for z, mask in ((np.full_like(Z, 255), use), (Z, np.zeros_like(use))):
            pp, qq, info = refine([(z, c, mask, [(0, 1)], 2)], P, Q, [0.1], args)
            np.testing.assert_array_equal(pp[0], P[0])
            np.testing.assert_array_equal(qq, Q)
            self.assertEqual(info["stop"], "no_observations")
            self.assertEqual(info["iterations"], 0)

    def test_progress_times_completed_updates_and_final_partial_block(self):
        rng = np.random.default_rng(0)
        W, N, K = 7, 4, 3
        c = np.arange(0, 2 * W + 1, 2, dtype=np.int64)
        Z = rng.integers(0, 2, (W, 2 * N), dtype=np.uint8)
        P = [np.concatenate([rng.dirichlet([0.4, 0.4], K).T for _ in range(W)]).ravel()]
        Q = rng.dirichlet([0.4] * K, N)
        data = [(Z, c, np.ones(W, np.uint8), [(0, W)], 2 * N)]
        posterior = cy.posterior
        clock = [0.0]

        def advance(*args, **kwargs):
            clock[0] += 2
            return posterior(*args, **kwargs)

        for bw in (False, True):
            args = Namespace(baum_welch=bw, p_prior=10, q_prior=10, iter=12, tole=0)
            text = StringIO()
            clock[0] = 0
            with (
                patch("hapla.fatash.perf_counter", side_effect=lambda: clock[0]),
                patch("hapla.fatash.printTiming") as timing,
                patch.object(cy, "posterior", side_effect=advance),
                redirect_stdout(text),
            ):
                _, _, info = refine(data, P, Q, [0.2], args)
            self.assertEqual(
                [c.args[0].split()[0] for c in timing.call_args_list], ["(5)", "(10)", "(12)"]
            )
            self.assertEqual([c.args[1] for c in timing.call_args_list], [10, 10, 4])
            self.assertIn("Initial ", text.getvalue())
            self.assertEqual(info["iterations"], 12)
            self.assertEqual(info["seconds"], 26)

    def test_refinement_counts_and_frequency_update_ignore_missing(self):
        Z = np.array([[0, 255, 1], [1, 255, 0]], np.uint8)
        c = np.array([0, 2, 2, 4], np.int64)
        use = np.array([1, 1, 0], np.uint8)
        G = np.array([[[0.9, 0.1], [0.8, 0.2], [0.7, 0.3]], [[0.2, 0.8], [0.1, 0.9], [0.3, 0.7]]])
        C = np.zeros(8)
        cy.accumulate(Z, G, use, c, 0, 3, C)
        np.testing.assert_allclose(C, [0.9, 0.1, 0.2, 0.8, 0, 0, 0, 0])
        base = np.full(8, 0.5)
        actual = cy.refineP(base, C, c, 2, 0)
        np.testing.assert_allclose(actual[:4], [0.9 / 1.1, 0.1 / 0.9, 0.2 / 1.1, 0.8 / 0.9])
        np.testing.assert_array_equal(actual[4:], base[4:])
        self.assertAlmostEqual(
            cy.penalty(np.array([1.0, 0.0]), np.array([0.4, 0.6]), 3), 3 * np.log(0.4)
        )
        self.assertEqual(cy.penalty(np.array([1.0, 0.0]), np.array([0.0, 1.0]), 0), 0)
        self.assertEqual(cy.penalty(np.array([1.0, 0.0]), np.array([0.0, 1.0]), 3), -np.inf)

    def test_map_accepts_likelihood_decrease_when_penalized_objective_improves(self):
        rng = np.random.default_rng(0)
        W, N, K = 7, 4, 3
        c = np.arange(0, 2 * W + 1, 2, dtype=np.int64)
        Z = rng.integers(0, 2, (W, 2 * N), dtype=np.uint8)
        P = [np.concatenate([rng.dirichlet([0.4, 0.4], K).T for _ in range(W)]).ravel()]
        Q = rng.dirichlet([0.4] * K, N)
        args = Namespace(baum_welch=False, p_prior=1, q_prior=10, iter=50, tole=1e-8)
        _, _, info = refine([(Z, c, np.ones(W, np.uint8), [(0, W)], 4)], P, Q, [0.1], args)
        self.assertEqual(info["stop"], "converged")
        self.assertTrue(np.any(np.diff([v["log_likelihood"] for v in info["history"]]) < -1e-5))
        self.assertTrue(np.all(np.diff([v["objective"] for v in info["history"]]) >= -1e-12))

    def test_phase_correction_swaps_confidence_with_labels(self):
        D = np.array([[0, 0, 1, 1, 0], [1, 1, 0, 0, 1]], np.uint8)
        P = np.arange(10, dtype=float).reshape(2, 5)
        self.assertEqual(cy.phaseCorrect(D, P, 0, True), 2)
        np.testing.assert_array_equal(D, [[0] * 5, [1] * 5])
        np.testing.assert_array_equal(P, [[0, 1, 7, 8, 4], [5, 6, 2, 3, 9]])


### Check local ancestry fitting, streaming, and saved output
class FatashPipeline(TemporaryTests):
    def test_saved_q_ids_must_match_cluster_order(self):
        ref, _, _, _, _ = self.fixture()
        qfile = self.root / "swapped.Q"
        qfile.write_bytes(Path(f"{ref}.Q").read_bytes())
        side = qfile.with_suffix(".ids")
        side.write_text("".join(f"s{i}\n" for i in reversed(range(7))))
        opts = ("--clusters", ref, "--pfile", f"{ref}.P", "--qfile", qfile, "--fixed-model")
        out = self.root / "order"
        res = command("fatash", *opts, "--out", out, success=False)
        self.assertIn("Q sample IDs differ", res.stderr)
        side.unlink()
        command("fatash", *opts, "--out", out)

    def test_native_text_output_appends_across_buffer_boundaries(self):
        D = np.tile(np.arange(255, dtype=np.uint8), (1200, 1))
        P = np.linspace(0, 1, D.size).reshape(D.shape)
        for name, data in (("path", D), ("prob", P)):
            pth = self.root / name
            with pth.open("w") as dst:
                for rows in (data[:400], data[400:]):
                    cy.writeRows(dst.fileno(), **({"D": rows} if name == "path" else {"P": rows}))
            np.testing.assert_allclose(np.loadtxt(pth), data, atol=5e-9)
        edge = np.array(
            [[0.001, 0.00987654321, 0.01, 0.0987654321, 0.1, 0.987654321, 1, 0, -0.5, 1.5]]
        )
        pth = self.root / "edge"
        with pth.open("w") as dst:
            cy.writeRows(dst.fileno(), P=edge)
        self.assertEqual(pth.read_text(), " ".join(f"{x:.8g}" for x in edge[0]) + "\n")
        with (self.root / "path").open("r") as src:
            with self.assertRaises(OSError):
                cy.writeRows(src.fileno(), D=D[:1])

    def fixture(self, name="input", N=7, W=13, missing=True):
        rng = np.random.default_rng(249)
        c = np.arange(0, W * 2 + 1, 2, dtype=np.int64)
        Z = rng.integers(0, 2, (W, 2 * N), dtype=np.uint8)
        if missing:
            Z[3] = 255
            Z[:, :2] = 255
        ref = writeClusters(self.root, name, Z, c)
        P = np.concatenate([rng.dirichlet([0.5, 0.5], 3).T for _ in range(W)])
        Q = rng.dirichlet([1, 1, 1], N)
        np.savetxt(f"{ref}.P", P)
        np.savetxt(f"{ref}.Q", Q)
        return ref, Z, c, P, Q

    def test_mean_ensemble_bounded_batches_threads_and_single_filelist(self):
        ref, Z, c, P, Q = self.fixture(W=1031)
        files, pfiles = self.root / "files", self.root / "pfiles"
        files.write_text(f"{ref}\n")
        pfiles.write_text(f"\n {ref}.P \r\n\n")
        result = []
        for i, threads in enumerate((1, 4)):
            out = self.root / f"out{i}"
            command(
                "fatash",
                "--filelist",
                files,
                "--pfilelist",
                pfiles,
                "--qfile",
                f"{ref}.Q",
                "--fixed-model",
                "--threads",
                threads,
                "--buffer-mb",
                1,
                "--save-posteriors",
                "--out",
                out,
            )
            result.append((np.loadtxt(f"{out}.path"), np.loadtxt(f"{out}.prob")))
        for a, b in zip(*result):
            np.testing.assert_array_equal(a, b)
        table = cy.emissionTable(P.ravel(), c, 3)
        E = cy.emissions(Z.T.copy(), table, c, np.ones(len(Z), np.uint8), 3, 0, len(Z))
        G, _, _ = cy.posterior(E, np.repeat(Q, 2, axis=0), np.logspace(-9, -4, 5))
        np.testing.assert_array_equal(result[0][0], G.argmax(axis=2))
        np.testing.assert_allclose(result[0][1], G.max(axis=2), atol=5e-9)

    def test_multichromosome_resets_blocks_and_no_phase_correction_across_boundary(self):
        ref, Z, c, P, Q = self.fixture(W=8, missing=False)
        rows = Path(f"{ref}.win").read_text().splitlines()
        rows[5:] = ["2" + r[1:] for r in rows[5:]]
        Path(f"{ref}.win").write_text("\n".join(rows) + "\n")
        out = self.root / "joined"
        command(
            "fatash",
            "--clusters",
            ref,
            "--pfile",
            f"{ref}.P",
            "--qfile",
            f"{ref}.Q",
            "--fixed-model",
            "--alpha",
            0.3,
            "--block",
            3,
            "--phase-correct",
            "--save-posteriors",
            "--out",
            out,
        )
        expected, confidence = [], []
        table = cy.emissionTable(P.ravel(), c, 3)
        for beg, end in ((0, 4), (4, 8)):
            E = cy.emissions(Z.T.copy(), table, c, np.ones(8, np.uint8), 3, beg, end, 3)
            G, _, _ = cy.posterior(E, np.repeat(Q, 2, axis=0), 0.3)
            d = np.repeat(G.argmax(axis=2).astype(np.uint8), 3, axis=1)[:, :4].copy()
            p = np.repeat(G.max(axis=2), 3, axis=1)[:, :4].copy()
            cy.phaseCorrect(d, p, 0, True)
            expected.append(d)
            confidence.append(p)
        np.testing.assert_array_equal(np.loadtxt(f"{out}.path"), np.concatenate(expected, axis=1))
        np.testing.assert_allclose(
            np.loadtxt(f"{out}.prob"), np.concatenate(confidence, axis=1), atol=5e-9
        )

        # A block larger than the chromosome must allocate only real output windows.
        command(
            "fatash",
            "--clusters",
            ref,
            "--pfile",
            f"{ref}.P",
            "--qfile",
            f"{ref}.Q",
            "--fixed-model",
            "--alpha",
            0.3,
            "--block",
            1000000000,
            "--buffer-mb",
            1,
            "--save-posteriors",
            "--out",
            out,
        )
        expected, confidence = [], []
        for beg, end in ((0, 4), (4, 8)):
            E = cy.emissions(Z.T.copy(), table, c, np.ones(8, np.uint8), 3, beg, end, 1000000000)
            G, _, _ = cy.posterior(E, np.repeat(Q, 2, axis=0), 0.3)
            expected.append(np.repeat(G.argmax(axis=2), 4, axis=1))
            confidence.append(np.repeat(G.max(axis=2), 4, axis=1))
        np.testing.assert_array_equal(np.loadtxt(f"{out}.path"), np.concatenate(expected, axis=1))
        np.testing.assert_allclose(
            np.loadtxt(f"{out}.prob"), np.concatenate(confidence, axis=1), atol=5e-9
        )

    def test_fitting_modes_and_saved_model_reuse(self):
        ref, _, _, _, Q = self.fixture()
        for flag, mode in (
            ([], "regularized"),
            (["--baum-welch"], "baum-welch"),
            (["--fixed-model"], "fixed"),
        ):
            out = self.root / mode
            res = command(
                "fatash",
                "--clusters",
                ref,
                "--pfile",
                f"{ref}.P",
                "--qfile",
                f"{ref}.Q",
                "--alpha",
                0.1,
                "--tole",
                0,
                "--save-posteriors",
                "--out",
                out,
                *flag,
            )
            log = Path(f"{out}.log").read_text()
            stats = readLog(out)
            self.assertNotIn("tolerance", res.stdout)
            self.assertNotIn("(0)", res.stdout)
            self.assertEqual(stats["Mode"], mode)
            if mode != "fixed":
                self.assertEqual(stats["Unobserved samples"], "1")
                self.assertEqual(stats["Iterations"], "10")
                self.assertEqual(
                    [r.split()[0] for r in res.stdout.splitlines() if r.startswith("(")],
                    ["(5)", "(10)"],
                )
                rows = log.split("History:\n", 1)[1].split("\n\n", 1)[0].splitlines()
                col = rows[0].split().index("Objective" if mode == "regularized" else "Log-like")
                self.assertTrue(
                    np.all(np.diff([float(r.split()[col]) for r in rows[1:] if r.strip()]) >= -1e-9)
                )
            q = np.loadtxt(f"{out}.Q")
            np.testing.assert_allclose(q[0], Q[0], atol=5e-10)
            repeat = self.root / f"{mode}-reuse"
            command(
                "fatash",
                "--clusters",
                ref,
                "--pfile",
                f"{out}.P",
                "--qfile",
                f"{out}.Q",
                "--alpha",
                0.1,
                "--fixed-model",
                "--save-posteriors",
                "--out",
                repeat,
            )
            np.testing.assert_array_equal(np.loadtxt(f"{out}.path"), np.loadtxt(f"{repeat}.path"))
            np.testing.assert_allclose(
                np.loadtxt(f"{out}.prob"), np.loadtxt(f"{repeat}.prob"), atol=1e-8
            )
            self.assertEqual(Path(f"{out}.ids").read_text(), Path(f"{ref}.ids").read_text())

    def test_joint_chromosomes_and_thread_batch_invariance(self):
        ref, Z, c, P, _ = self.fixture(N=300, W=67)
        rows = Path(f"{ref}.win").read_text().splitlines()
        rows[33:] = ["2" + r[1:] for r in rows[33:]]
        Path(f"{ref}.win").write_text("\n".join(rows) + "\n")
        files, pfiles = self.root / "files", self.root / "pfiles"
        a = writeClusters(self.root, "a", Z[:32], c[:33])
        b = writeClusters(self.root, "b", Z[32:], c[32:] - c[32])
        np.savetxt(f"{a}.P", P[: c[32]])
        np.savetxt(f"{b}.P", P[c[32] :])
        files.write_text(f"{a}\n{b}\n")
        pfiles.write_text(f"{a}.P\n{b}.P\n")
        inputs = (
            ["--clusters", ref, "--pfile", f"{ref}.P"],
            ["--filelist", files, "--pfilelist", pfiles],
            ["--clusters", a, b, "--pfile", f"{a}.P", f"{b}.P"],
            ["--clusters", a, b, "--pfilelist", pfiles],
            ["--filelist", files, "--pfile", f"{a}.P", f"{b}.P"],
        )
        for bw in ([], ["--baum-welch"]):
            results = []
            for i, inp in enumerate(inputs):
                multi, threads, buf = i > 0, 4 if i else 1, 1 if i else 256
                out = self.root / f"joint{i}{bool(bw)}"
                command(
                    "fatash",
                    *inp,
                    "--qfile",
                    f"{ref}.Q",
                    "--alpha",
                    0.2,
                    "--iter",
                    3,
                    "--tole",
                    0,
                    "--threads",
                    threads,
                    "--buffer-mb",
                    buf,
                    "--save-posteriors",
                    "--out",
                    out,
                    *bw,
                )
                stems = [f"{out}.chr1", f"{out}.chr2"] if multi else [str(out)]
                results.append(
                    (
                        np.loadtxt(f"{out}.Q"),
                        np.concatenate([np.loadtxt(f"{s}.P") for s in stems]),
                        np.concatenate([np.loadtxt(f"{s}.path") for s in stems], axis=1),
                        np.concatenate([np.loadtxt(f"{s}.prob") for s in stems], axis=1),
                    )
                )
                if multi:
                    self.assertEqual(
                        Path(f"{out}.pfilelist").read_text(), "".join(f"{s}.P\n" for s in stems)
                    )
            for result in results[1:]:
                for x, y in zip(results[0], result):
                    np.testing.assert_allclose(x, y, atol=1e-8)

    def test_invalid_fit_options(self):
        ref, *_ = self.fixture()
        for flags in (
            ["--iter", 0],
            ["--tole", -1],
            ["--p-prior", "nan"],
            ["--q-prior", -1],
            ["--baum-welch", "--fixed-model"],
            ["--simple"],
            ["--medians"],
            ["--block", 2],
        ):
            result = command(
                "fatash",
                "--clusters",
                ref,
                "--pfile",
                f"{ref}.P",
                "--qfile",
                f"{ref}.Q",
                "--out",
                self.root / "bad",
                *flags,
                success=False,
            )
            self.assertIn("error:", result.stderr)

    def test_one_sample_one_window_one_ancestry_and_empty_windows(self):
        ref = writeClusters(
            self.root, "single", np.array([[255, 255], [0, 255]], np.uint8), np.array([0, 0, 1])
        )
        Path(f"{ref}.Q").write_text("1\n")
        Path(f"{ref}.P").write_text("1\n")
        for mode in ([], ["--viterbi"], ["--fixed-model", "--simple"]):
            command(
                "fatash",
                "--clusters",
                ref,
                "--pfile",
                f"{ref}.P",
                "--qfile",
                f"{ref}.Q",
                "--out",
                self.root / "one",
                *mode,
            )
            np.testing.assert_array_equal(np.loadtxt(self.root / "one.path"), 0)
        empty = writeClusters(self.root, "empty", np.full((1, 2), 255, np.uint8), np.array([0, 0]))
        Path(f"{empty}.P").touch()
        command(
            "fatash",
            "--clusters",
            empty,
            "--pfile",
            f"{empty}.P",
            "--qfile",
            f"{ref}.Q",
            "--out",
            self.root / "empty-result",
        )

    def test_invalid_inputs_preserve_outputs_and_stale_confidence_is_removed(self):
        ref, _, _, P, _ = self.fixture()
        out = self.root / "protected"
        command(
            "fatash",
            "--clusters",
            ref,
            "--pfile",
            f"{ref}.P",
            "--qfile",
            f"{ref}.Q",
            "--fixed-model",
            "--save-posteriors",
            "--out",
            out,
        )
        before = Path(f"{out}.path").read_bytes()
        P[0, 0] = np.nan
        np.savetxt(f"{ref}.P", P)
        r = command(
            "fatash",
            "--clusters",
            ref,
            "--pfile",
            f"{ref}.P",
            "--qfile",
            f"{ref}.Q",
            "--out",
            out,
            success=False,
        )
        self.assertNotEqual(r.returncode, 0)
        self.assertEqual(Path(f"{out}.path").read_bytes(), before)
        self.assertTrue(Path(f"{out}.prob").exists())
        ref, *_ = self.fixture()
        command(
            "fatash", "--clusters", ref, "--pfile", f"{ref}.P", "--qfile", f"{ref}.Q", "--out", out
        )
        self.assertFalse(Path(f"{out}.prob").exists())
        r = command(
            "fatash",
            "--clusters",
            ref,
            "--pfile",
            f"{ref}.P",
            "--qfile",
            f"{ref}.Q",
            "--out",
            ref,
            success=False,
        )
        self.assertIn("conflicts with input", r.stderr)

    def test_sample_identifiers_with_hash_are_preserved(self):
        ref, *_ = self.fixture(N=3)
        Path(f"{ref}.ids").write_text("A#1\nA#2\nB\n")
        _, ids, _, _ = readMetadata(ref)
        np.testing.assert_array_equal(ids, ["A#1", "A#2", "B"])
        Path(f"{ref}.ids").write_text("A\nA\nB\n")
        with self.assertRaisesRegex(ValueError, "unique"):
            readMetadata(ref)
