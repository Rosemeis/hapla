"""Dense compression references, adaptive boundaries, and frozen prediction."""

__author__ = "Jonas Meisner"

import math
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from helpers import TemporaryTests, command, toBcf, writeVcf

from hapla import packed_cy
from hapla.cluster import adaptiveFit


### Count dictionary, assignment, and residual bits independently
def denseCost(G, res):
    B, H = G.shape
    z, R, n = res["labels"], res["medians"], res["sizes"]
    K, observed = len(R), int(n.sum())
    lost = int(np.count_nonzero((G != 255) & (z[None] == 255)))
    bits = lost + 2 * B.bit_length() + 2 * (K + 1).bit_length() - 2 + K * B
    if K:
        bits += math.log2(math.comb(observed - 1, K - 1))
        bits += math.log2(math.factorial(observed))
        for k in range(K):
            bits -= math.log2(math.factorial(int(n[k])))
            cells = G[:, z == k]
            errors = int(np.count_nonzero(cells != R[k, :, None]))
            bits += 1
            if errors:
                bits += math.log2(cells.size) + math.log2(math.comb(cells.size, errors))
    return bits, lost


### Enumerate all partitions allowed by the bounded binary hierarchy
def partitions(a, b, minimum):
    yield [(a, b)]
    if b - a >= 2 * minimum:
        mid = a + (b - a) // 2
        for left in partitions(a, mid, minimum):
            for right in partitions(mid, b, minimum):
                yield left + right


### Run the selector with exact variant positions
def adaptive(G, minimum=8, **options):
    pos = np.arange(1, len(G) + 1, dtype=np.int64) * 10
    opt = dict(alpha=0, min_freq=0, min_mac=1, K_max=255, n_iter=1000)
    opt.update(options)
    return adaptiveFit(
        (0, "1", int(pos[0]), int(pos[-1]), len(G)), G, bool(np.any(G == 255)), opt, minimum, pos
    )


### Convert phased allele arrays to VCF rows
def rows(G, chroms=None, positions=None):
    return [
        (
            "1" if chroms is None else str(chroms[j]),
            j + 1 if positions is None else int(positions[j]),
            tuple(f"{a}|{b}".replace("255", ".") for a, b in row.reshape(-1, 2)),
        )
        for j, row in enumerate(G)
    ]


### Read simple window rows without inferring mixed column types
def windows(pfx):
    return [line.split() for line in Path(f"{pfx}.win").read_text().splitlines()[1:]]


### Check native code length against literal dense reconstruction
class AdaptiveCostTests(unittest.TestCase):
    def test_packed_children_match_direct_fits(self):
        rng = np.random.default_rng(702)
        G = rng.integers(0, 2, (139, 40), dtype=np.uint8)
        for missing in (False, True):
            if missing:
                G[rng.random(G.shape) < 0.002] = 255
            packed = packed_cy.packWindow(G, missing)
            before = packed[0].copy()
            for beg, end in ((0, 139), (0, 64), (1, 65), (31, 96), (63, 128), (65, 139)):
                x = G[beg:end]
                if missing:
                    self.assertEqual(
                        packed_cy.observedWindow(packed[1], beg, end),
                        np.count_nonzero(np.all(x != 255, axis=0)),
                    )
                for cap in (3, 255):
                    opt = dict(min_mac=1, K_max=cap, missing=bool(np.any(x == 255)))
                    a = packed_cy.fitWindow(x, **opt)
                    b = packed_cy.fitWindow(x, packed=packed, offset=beg, **opt)
                    for key in ("labels", "medians", "counts", "sizes"):
                        np.testing.assert_array_equal(a[key], b[key])
                    self.assertEqual(a["stats"], b["stats"])
            np.testing.assert_array_equal(packed[0], before)
            for offset in (-1, 140):
                with self.assertRaises(ValueError):
                    packed_cy.fitWindow(G[:8], packed=packed, offset=offset, min_mac=1)
        for G, missing in (
            (np.zeros((0, 2), np.uint8), False),
            (np.full((8, 2), 2, np.uint8), True),
            (np.full((8, 2), 255, np.uint8), False),
        ):
            with self.assertRaises(ValueError):
                packed_cy.packWindow(G, missing)

    def test_dense_score_with_residuals_missingness_and_allele_flips(self):
        rng = np.random.default_rng(701)
        for B in (1, 8, 17, 64, 65):
            G = rng.integers(0, 2, (B, 80), dtype=np.uint8)
            G[0, :3] = 255
            for cap in (1, 4, 255):
                res = packed_cy.fitWindow(G, min_mac=1, K_max=cap)
                actual = packed_cy.windowCost(
                    G, res["labels"], res["medians"], res["counts"], res["sizes"]
                )
                expected = denseCost(G, res)
                self.assertAlmostEqual(actual[0], expected[0], delta=1e-8)
                self.assertEqual(actual[1], expected[1])
                flip = rng.integers(0, 2, B, dtype=np.uint8)
                X = np.where(G == 255, 255, G ^ flip[:, None]).astype(np.uint8)
                other = packed_cy.fitWindow(X, min_mac=1, K_max=cap)
                recoded = packed_cy.windowCost(
                    X, other["labels"], other["medians"], other["counts"], other["sizes"]
                )
                self.assertEqual(actual, recoded)
        G = np.full((8, 6), 255, np.uint8)
        res = packed_cy.fitWindow(G)
        self.assertEqual(
            packed_cy.windowCost(G, res["labels"], res["medians"], res["counts"], res["sizes"]),
            denseCost(G, res),
        )

    def test_invalid_dimensions_labels_medians_and_counts(self):
        G = np.zeros((8, 6), np.uint8)
        res = packed_cy.fitWindow(G, min_mac=1)
        valid = [G, res["labels"], res["medians"], res["counts"], res["sizes"]]
        for i, bad in (
            (0, np.empty((0, 6), np.uint8)),
            (1, np.zeros(5, np.uint8)),
            (1, np.ones(6, np.uint8)),
            (2, np.zeros((1, 7), np.uint8)),
            (2, np.full((1, 8), 2, np.uint8)),
            (3, np.zeros((1, 7), np.uint64)),
            (3, np.full((1, 8), 7, np.uint64)),
            (4, np.array([5], np.uint64)),
        ):
            args = valid.copy()
            args[i] = bad
            with self.assertRaises(ValueError):
                packed_cy.windowCost(*args)


### Compare fitted boundaries with independent costs for all candidate partitions
class AdaptiveFitTests(unittest.TestCase):
    def test_equal_cost_prefers_the_parent(self):
        G = np.tile(np.array([0, 0, 0, 1, 1, 1], np.uint8), (16, 1))
        with patch("hapla.packed_cy.windowCost", side_effect=lambda x, *args: (len(x), 0)):
            bits, lost, result, _ = adaptive(G)
        self.assertEqual((bits, lost), (16, 0))
        self.assertEqual([r["window"][4] for r in result], [16])
        G[0, 0] = 255

        def cost(x, z, *args):
            return len(x), int(np.count_nonzero((x != 255) & (z[None] == 255)))

        with patch("hapla.packed_cy.windowCost", side_effect=cost):
            bits, lost, result, _ = adaptive(G)
        self.assertEqual((bits, lost), (16, 7))
        self.assertEqual([r["window"][4] for r in result], [8, 8])

    def test_global_hierarchy_optimum_matches_dense_enumeration(self):
        rng = np.random.default_rng(391)
        for _ in range(8):
            G = rng.integers(0, 2, (16, 48), dtype=np.uint8)
            G[0, 0] = 255
            expected = []
            for partition in partitions(0, len(G), 4):
                costs = [
                    denseCost(G[a:b], packed_cy.fitWindow(G[a:b], min_mac=1, K_max=4))
                    for a, b in partition
                ]
                expected.append((sum(v[0] for v in costs), sum(v[1] for v in costs)))
            bits, lost, result, fitted = adaptive(G, minimum=4, K_max=4)
            target = min(expected)
            self.assertEqual(lost, target[1])
            self.assertAlmostEqual(bits, target[0], delta=1e-8)
            self.assertLessEqual(fitted, 7)
            self.assertEqual(sum(r["window"][4] for r in result), len(G))

    def test_ld_join_and_independent_block_split(self):
        H = 256
        first = (np.arange(H) % 2).astype(np.uint8)
        second = ((np.arange(H) // 2) % 2).astype(np.uint8)
        same = np.tile(first, (32, 1))
        independent = np.vstack((np.tile(first, (16, 1)), np.tile(second, (16, 1))))
        self.assertEqual([r["window"][4] for r in adaptive(same)[2]], [32])
        self.assertEqual([r["window"][4] for r in adaptive(independent)[2]], [16, 16])
        self.assertEqual(
            [r["window"][:4] for r in adaptive(independent)[2]],
            [(0, "1", 10, 160), (16, "1", 170, 320)],
        )

    def test_sparse_missingness_is_charged_without_forcing_smallest_windows(self):
        G = np.zeros((16, 32), np.uint8)
        G[0, 0] = 255
        parent = denseCost(G, packed_cy.fitWindow(G, min_mac=1))
        bits, lost, result, _ = adaptive(G)
        self.assertAlmostEqual(parent[0], bits, delta=1e-8)
        self.assertEqual(parent[1], lost)
        self.assertEqual(lost, 15)
        self.assertEqual([r["window"][4] for r in result], [16])
        G[0, :8] = 255
        _, lost, result, _ = adaptive(G)
        self.assertEqual(lost, 56)
        self.assertEqual([r["window"][4] for r in result], [8, 8])

    def test_missing_support_can_recover_in_children_and_irreducible_fails(self):
        G = np.zeros((16, 6), np.uint8)
        G[0, :2] = G[-1, 2:4] = 255
        _, _, result, _ = adaptive(G, min_mac=3)
        self.assertEqual([r["window"][4] for r in result], [8, 8])
        self.assertTrue(all(int(r["sizes"].sum()) == 4 for r in result))
        with self.assertRaises(ValueError):
            adaptive(G[:8].copy(), min_mac=5)

    def test_infeasible_split_retains_valid_all_missing_parent(self):
        G = np.zeros((16, 6), np.uint8)
        G[:8, 2:] = G[8:, :2] = 255
        _, lost, result, _ = adaptive(G, min_mac=3)
        self.assertEqual(lost, 48)
        self.assertEqual([r["window"][4] for r in result], [16])
        self.assertEqual(result[0]["stats"]["K"], 0)
        G[8:, :4] = 0
        G[8:, 4:] = 255
        with self.assertRaises(ValueError):
            adaptive(G, min_mac=3)


### Check bounded streaming, deterministic fitting, and reusable references
class AdaptivePipelineTests(TemporaryTests):
    def test_prediction_threads_buffers_and_allele_coding(self):
        rng = np.random.default_rng(406)
        G = rng.integers(0, 2, (137, 24), dtype=np.uint8)
        G[0, 0] = G[90, 2] = 255
        G[64:72] = 255
        chroms = np.r_[np.ones(73, int), np.full(64, 2)]
        pos = np.r_[np.arange(1, 74), np.arange(1, 65)]
        samples = tuple(f"s{i}" for i in range(G.shape[1] // 2))
        vcf, bcf = self.root / "input.vcf", self.root / "input.bcf"
        writeVcf(vcf, rows(G, chroms, pos), samples=samples)
        toBcf(vcf, bcf)
        outputs = []
        for j, (threads, buffer, batch) in enumerate(((1, 1, 1), (4, 2, 7))):
            pfx, pred = self.root / f"fit{j}", self.root / f"prediction{j}"
            common = ("--bcf", bcf, "--threads", threads, "--buffer-mb", buffer)
            result = command(
                "cluster",
                *common,
                "--adaptive",
                "--min-mac",
                1,
                "--batch-windows",
                batch,
                "--medians",
                "--plink",
                "--out",
                pfx,
            )
            self.assertIn("Size: adaptive (8-64)", result.stdout)
            command("predict", *common, "--ref", pfx, "--plink", "--out", pred)
            for suffix in (".bca", ".win", ".ids", ".bed", ".bim", ".fam"):
                self.assertEqual(
                    Path(f"{pfx}{suffix}").read_bytes(),
                    Path(f"{pred}{suffix}").read_bytes(),
                    suffix,
                )
            outputs.append(pfx)
        for suffix in (".bca", ".win", ".ids", ".bcm", ".blk", ".wix", ".ref"):
            self.assertEqual(
                Path(f"{outputs[0]}{suffix}").read_bytes(),
                Path(f"{outputs[1]}{suffix}").read_bytes(),
                suffix,
            )
        flip = rng.integers(0, 2, len(G), dtype=np.uint8)
        X = np.where(G == 255, 255, G ^ flip[:, None]).astype(np.uint8)
        writeVcf(vcf, rows(X, chroms, pos), samples=samples)
        recoded = self.root / "recoded"
        command("cluster", "--vcf", vcf, "--adaptive", "--min-mac", 1, "--out", recoded)
        for suffix in (".bca", ".win"):
            self.assertEqual(
                Path(f"{outputs[0]}{suffix}").read_bytes(),
                Path(f"{recoded}{suffix}").read_bytes(),
                suffix,
            )

    def test_chromosome_physical_and_genetic_caps_include_all_tails(self):
        G = np.zeros((12, 6), np.uint8)
        pos = [1, 2, 3, 20, 21, 22, 1, 2, 3, 20, 21, 22]
        chroms = [1] * 6 + [2] * 6
        vcf, gmap = self.root / "input.vcf", self.root / "map"
        writeVcf(vcf, rows(G, chroms, pos))
        gmap.write_text("CHR BP CM\n1 1 0\n1 100 9.9\n2 1 0\n2 100 9.9\n")
        for j, extra in enumerate(
            (("--max-length", 2), ("--map", gmap, "--max-cm", 0.15), ("--map", gmap))
        ):
            pfx = self.root / f"bounded{j}"
            command(
                "cluster",
                "--vcf",
                vcf,
                "--adaptive",
                "--min-size",
                2,
                "--max-size",
                4,
                "--min-mac",
                1,
                "--out",
                pfx,
                *extra,
            )
            result = windows(pfx)
            self.assertEqual(sum(int(r[4]) for r in result), len(G))
            self.assertEqual([r[0] for r in result], sorted(r[0] for r in result))
            self.assertTrue(all(1 <= int(r[4]) <= 4 for r in result))
            limit = 2 if j == 0 else 1
            self.assertTrue(all(int(r[2]) - int(r[1]) <= limit for r in result))
        for text, expected in (
            ("1 1 0\n1 10 1\n2 1 0\n2 100 1\n", "coverage"),
            ("1 1 0\n1 100 1\n", "absent"),
        ):
            gmap.write_text(text)
            result = command(
                "cluster",
                "--vcf",
                vcf,
                "--adaptive",
                "--map",
                gmap,
                "--min-mac",
                1,
                "--out",
                self.root / "invalid",
                success=False,
            )
            self.assertIn(expected, result.stderr)

    def test_adaptive_options_fail_before_publishing_outputs(self):
        vcf, pfx = self.root / "input.vcf", self.root / "protected"
        writeVcf(vcf, rows(np.zeros((8, 6), np.uint8)))
        Path(f"{pfx}.bca").write_bytes(b"existing analysis")
        for options in (
            ("--size", 8, "--min-size", 4),
            ("--size", 8, "--max-size", 16),
            ("--size", 8, "--max-length", 100),
            ("--size", 8, "--map", vcf),
            ("--adaptive", "--size", 8),
            ("--adaptive", "--step", 8),
            ("--adaptive", "--tail", "drop"),
            ("--adaptive", "--min-size", 0),
            ("--adaptive", "--min-size", 9, "--max-size", 8),
            ("--adaptive", "--max-length", 0),
            ("--adaptive", "--max-cm", 0.1),
            ("--adaptive", "--map", vcf, "--max-cm", "nan"),
        ):
            result = command("cluster", "--vcf", vcf, "--out", pfx, *options, success=False)
            self.assertEqual(result.returncode, 2, options)
            self.assertEqual(Path(f"{pfx}.bca").read_bytes(), b"existing analysis")
            self.assertFalse(list(self.root.glob(".hapla-*")))


if __name__ == "__main__":
    unittest.main()
