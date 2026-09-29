"""Phased genotype validation, reference alignment, and cluster prediction."""

__author__ = "Jonas Meisner"

import unittest
from pathlib import Path

import numpy as np
from helpers import TemporaryTests, command, toBcf, writeVcf

from hapla import packed_cy
from hapla.formats import MAGIC


### Read predicted cluster labels from a binary assignment file
def labels(prefix, samples=3):
    return np.frombuffer(Path(f"{prefix}.bca").read_bytes()[len(MAGIC) :], np.uint8).reshape(
        -1, 2 * samples
    )


### Compare packed predictions with scalar references
class PackedPredictionTests(unittest.TestCase):
    def test_short_window_lookup_matches_reference_with_duplicates_and_missingness(self):
        rng = np.random.default_rng(904)
        B, H, K = 16, 1024, 17
        R = rng.integers(0, 2, (K, B), dtype=np.uint8)
        R[-1] = R[0]
        for unique in (8, H):
            G = rng.integers(0, 2, (B, unique), dtype=np.uint8)
            if unique != H:
                G = np.ascontiguousarray(G[:, rng.integers(0, unique, H)])
            G[0, 5] = G[7, 700] = 255
            D = np.stack([np.count_nonzero(G != r[:, None], axis=0) for r in R])
            expected = (K - 1 - np.argmin(D[::-1], axis=0)).astype(np.uint8)
            expected[np.any(G == 255, axis=0)] = 255
            np.testing.assert_array_equal(packed_cy.predictHaplotypes(G, R), expected)

    def test_exact_distances_missingness_and_word_boundaries(self):
        rng = np.random.default_rng(620)
        for B in (1, 63, 64, 65, 129):
            for K in (0, 1, 7, 255):
                G = rng.integers(0, 2, (B, 100), dtype=np.uint8)
                R = rng.integers(0, 2, (K, B), dtype=np.uint8)
                G[0, 4] = 255
                expected = np.full(100, 255, np.uint8)
                if K:
                    D = np.stack([np.sum(G != r[:, None], axis=0) for r in R])
                    expected = (K - 1 - np.argmin(D[::-1], axis=0)).astype(np.uint8)
                    expected[4] = 255
                np.testing.assert_array_equal(packed_cy.predictHaplotypes(G, R), expected)
        with self.assertRaisesRegex(ValueError, "binary"):
            packed_cy.predictHaplotypes(np.zeros((1, 2), np.uint8), np.full((1, 1), 2, np.uint8))


### Check phased input, reference alignment, and output
class NativePredictionTests(TemporaryTests):
    def reference(self, rows=None, **extra):
        rows = rows or [("1", i * 10, ("0|0", "0|1", "1|1")) for i in range(1, 20)]
        vcf = self.root / "reference.vcf"
        prefix = self.root / "reference"
        writeVcf(vcf, rows)
        extra.setdefault("min_mac", 1)
        options = [
            value
            for key, arg in extra.items()
            for value in (f"--{key.replace('_', '-')}", str(arg))
        ]
        command("cluster", "--vcf", vcf, "--size", 8, "--medians", "--out", prefix, *options)
        return vcf, prefix, rows

    def test_training_data_reproduces_saved_assignments(self):
        rng = np.random.default_rng(841)
        G = np.repeat(rng.integers(0, 2, (274, 32), dtype=np.uint8), 16, axis=1)
        G[1, :2] = G[72, 2] = 255
        G[16:24] = 255
        rows = [
            (
                str(j // 137 + 1),
                j % 137 + 1,
                tuple(f"{a}|{b}".replace("255", ".") for a, b in row.reshape(-1, 2)),
            )
            for j, row in enumerate(G)
        ]
        vcf, bcf = self.root / "training.vcf", self.root / "training.bcf"
        writeVcf(vcf, rows, samples=tuple(f"s{i}" for i in range(G.shape[1] // 2)))
        toBcf(vcf, bcf)
        windows = self.root / "windows.txt"
        windows.write_text("0\n8\n16\n24\n65\n137\n202\n274\n")
        for i, opt in enumerate(
            (
                ("--size", 8),
                ("--size", 65, "--step", 32, "--tail", "drop"),
                ("--length", 64),
                ("--windows", windows),
            )
        ):
            ref = self.root / f"reference{i}"
            source, threads = (vcf, 1) if i == 0 else (bcf, 4)
            common = ("--bcf", source, "--threads", threads, "--buffer-mb", 1, "--plink")
            command("cluster", *common, *opt, "--medians", "--out", ref)
            out = self.root / f"predicted{i}"
            command("predict", *common, "--ref", ref, "--out", out)
            for suffix in (".bca", ".win", ".ids", ".bed", ".bim", ".fam"):
                self.assertEqual(
                    Path(f"{ref}{suffix}").read_bytes(),
                    Path(f"{out}{suffix}").read_bytes(),
                    (opt, suffix),
                )

    def test_unphased_calls_fail_without_publishing(self):
        _, ref, rows = self.reference(tail="drop")
        out = self.root / "protected"
        Path(f"{out}.bca").write_bytes(b"existing analysis")
        vcf = self.root / "unphased.vcf"
        for i, gt in enumerate(("0/1", "1/0", "./0", "0/.", "./1", "1/.")):
            bcf = self.root / f"unphased{i}.bcf"
            rows[-1] = ("1", 190, ("0|0", "0|1", gt))
            writeVcf(vcf, rows)
            toBcf(vcf, bcf)
            for source in (vcf, bcf):
                result = command(
                    "predict", "--bcf", source, "--ref", ref, "--out", out, success=False
                )
                self.assertIn("Unphased heterozygous or partially missing GT", result.stderr)
                self.assertIn("record 19, position 190", result.stderr)
                self.assertEqual(Path(f"{out}.bca").read_bytes(), b"existing analysis")
                self.assertFalse(Path(f"{out}.ref.json").exists())
                self.assertFalse(list(self.root.glob(".hapla-*")))

    def test_unindexed_parallel_phased_prediction_and_missingness(self):
        _, reference, rows = self.reference()
        rows[0] = ("1", 10, ("0/0", "0|1", "1/1"))
        rows[5] = ("1", 60, (".|0", "0|1", "1|."))
        vcf, bcf = self.root / "query.vcf", self.root / "query.bcf"
        writeVcf(vcf, rows)
        toBcf(vcf, bcf)
        expected = None
        for i, source in enumerate((vcf, bcf)):
            out = self.root / f"out{i}"
            command(
                "predict",
                "--bcf",
                source,
                "--ref",
                reference,
                "--threads",
                1 if i == 0 else 4,
                "--buffer-mb",
                1,
                "--batch-windows",
                1,
                "--plink",
                "--out",
                out,
            )
            current = [
                Path(f"{out}{s}").read_bytes()
                for s in (".bca", ".win", ".ids", ".bed", ".bim", ".fam")
            ]
            if expected is not None:
                self.assertEqual(current, expected)
            expected = current
            np.testing.assert_array_equal(
                labels(out)[0] == 255, [True, False, False, False, False, True]
            )
        self.assertFalse(Path(f"{bcf}.csi").exists())

    def test_reference_sites_and_corrupt_medians_fail_without_publishing(self):
        vcf, reference, _ = self.reference()
        original = vcf.read_text()
        out = self.root / "protected"
        Path(f"{out}.bca").write_bytes(b"existing analysis")
        query = self.root / "bad.vcf"
        variants = [
            original.replace("1\t40\t", "1\t41\t"),
            original.replace("1\t40\t.\tA\tG", "1\t40\t.\tG\tA"),
            original.rsplit("\n", 2)[0] + "\n",
            original + "1\t200\t.\tA\tG\t.\tPASS\t.\tGT\t0|0\t0|1\t1|1\n",
        ]
        for text in variants:
            query.write_text(text)
            result = command(
                "predict", "--vcf", query, "--ref", reference, "--out", out, success=False
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("reference", result.stderr)
            self.assertEqual(Path(f"{out}.bca").read_bytes(), b"existing analysis")
        path = Path(f"{reference}.bcm")
        raw = path.read_bytes()
        path.write_bytes(MAGIC + b"\x02" + raw[len(MAGIC) + 1 :])
        result = command("predict", "--vcf", vcf, "--ref", reference, "--out", out, success=False)
        self.assertIn("identity metadata", result.stderr)
        for old_magic in (bytes((7, 9, 13)), bytes((7, 9, 15))):
            path.write_bytes(old_magic + raw[len(MAGIC) :])
            result = command(
                "predict", "--vcf", vcf, "--ref", reference, "--out", out, success=False
            )
            self.assertIn("identity metadata", result.stderr)
        self.assertFalse(list(self.root.glob(".hapla-*")))

    def test_multiple_chromosomes_and_dropped_tails(self):
        rows = [
            (chrom, i * 10, ("0|0", "0|1", "1|1")) for chrom in ("1", "2") for i in range(1, 20)
        ]
        vcf, reference, _ = self.reference(rows, tail="drop")
        out = self.root / "predicted"
        command("predict", "--vcf", vcf, "--ref", reference, "--out", out)
        self.assertEqual(Path(f"{reference}.bca").read_bytes(), Path(f"{out}.bca").read_bytes())
        self.assertEqual(labels(out).shape[0], 4)

    def test_different_query_samples_and_haplotypes(self):
        from hapla.predict import readReference

        _, reference, rows = self.reference()
        rng = np.random.default_rng(138)
        G = rng.integers(0, 2, (len(rows), 4), dtype=np.uint8)
        query = self.root / "new_samples.vcf"
        query_rows = [
            (chrom, pos, tuple(f"{a}|{b}" for a, b in genotype.reshape(2, 2)))
            for (chrom, pos, _), genotype in zip(rows, G)
        ]
        writeVcf(query, query_rows, samples=("new_a", "new_b"))
        out = self.root / "new_samples"
        command("predict", "--vcf", query, "--ref", reference, "--out", out)
        specs, medians = readReference(reference)
        expected = []
        for meta, K, off in specs:
            idx, _, _, _, B = meta
            R = medians[off : off + B * K].reshape(K, B)
            block = G[idx : idx + B]
            distances = np.stack([np.sum(block != median[:, None], axis=0) for median in R])
            expected.append(K - 1 - np.argmin(distances[::-1], axis=0))
        np.testing.assert_array_equal(labels(out, samples=2), expected)
        self.assertEqual(Path(f"{out}.ids").read_text(), "new_a\nnew_b\n")

    def test_all_missing_reference_window_predicts_missing(self):
        rows = [("1", i, (".|.", ".|.", ".|.")) for i in range(1, 9)]
        vcf, reference, _ = self.reference(rows)
        out = self.root / "empty"
        command("predict", "--vcf", vcf, "--ref", reference, "--out", out, "--plink")
        self.assertTrue(np.all(labels(out) == 255))
        self.assertEqual(Path(f"{out}.bed").read_bytes(), bytes((108, 27, 1)))


if __name__ == "__main__":
    unittest.main()
