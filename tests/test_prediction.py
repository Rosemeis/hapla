"""Native prediction: phase detection, exact alignment, bounded I/O, and the 1.0 break."""

import unittest
from pathlib import Path

import numpy as np
from hapla.vcf_cy import Reader
from helpers import TemporaryTests, command, readLog, toBcf, writeVcf

from hapla import packed_cy
from hapla.formats import MAGIC


def labels(prefix, samples=3):
    return np.frombuffer(Path(f"{prefix}.bca").read_bytes()[len(MAGIC) :], np.uint8).reshape(
        -1, 2 * samples
    )


def unphased_reference(G, R):
    """Direct scalar version of the retained unphased pair heuristic."""
    dosage = G[:, 0::2] + G[:, 1::2]
    result = np.full(G.shape[1], 255, np.uint8)
    for sample in range(dosage.shape[1]):
        if np.any(G[:, 2 * sample : 2 * sample + 2] == 255) or not len(R):
            continue
        d = dosage[:, sample]
        best = np.float32(len(d) + 1)
        for a, ra in enumerate(R):
            h1 = np.where(d == 1, ra, d == 2)
            h2 = np.where(d == 1, 1 - ra, d == 2)
            d1 = np.count_nonzero(h1 != ra)
            for b, rb in enumerate(R):
                score = 0.66 * d1 + 0.33 * np.count_nonzero(h2 != rb)
                if score < best:
                    best = np.float32(score)
                    result[2 * sample : 2 * sample + 2] = a, b
    return result


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
            np.testing.assert_array_equal(packed_cy.predict_haplotypes(G, R), expected)

    def test_unphased_packed_pairs_and_ties_match_scalar_reference(self):
        from hapla.predict import predictBatch

        rng = np.random.default_rng(154)
        for B, K, N in [(b, k, 3) for b in (1, 8, 63, 64, 65, 129) for k in (1, 5, 17)] + [
            (65, 255, 1)
        ]:
            G = rng.integers(0, 2, (B, 2 * N), dtype=np.uint8)
            R = rng.integers(0, 2, (K, B), dtype=np.uint8)
            R[-1] = R[0]
            meta = (0, "1", 1, B, B)
            actual = predictBatch([(meta, G, np.ones(N, bool), R)])[0][2]
            np.testing.assert_array_equal(actual, unphased_reference(G, R))

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
                np.testing.assert_array_equal(packed_cy.predict_haplotypes(G, R), expected)
        with self.assertRaisesRegex(ValueError, "binary"):
            packed_cy.predict_haplotypes(np.zeros((1, 2), np.uint8), np.full((1, 1), 2, np.uint8))

    def test_mixed_phase_batch_against_independent_pair_reference(self):
        from hapla.predict import predictBatch

        rng = np.random.default_rng(400)
        for B in (8, 65):
            G = rng.integers(0, 2, (B, 12), dtype=np.uint8)
            R = rng.integers(0, 2, (5, B), dtype=np.uint8)
            G[3, 2] = 255
            G[4, 6] = 255
            phase = np.array([0, 0, 1, 1, 0, 1], bool)
            meta = (0, "1", 1, B, B)
            actual = predictBatch([(meta, G, phase, R)])[0][2]
            expected = packed_cy.predict_haplotypes(G, R)
            expected.reshape(-1, 2)[phase] = unphased_reference(G, R).reshape(-1, 2)[phase]
            np.testing.assert_array_equal(actual, expected)
            self.assertEqual(actual[2], 255)
            self.assertNotEqual(actual[3], 255)
            np.testing.assert_array_equal(actual[6:8], [255, 255])


class NativePredictionTests(TemporaryTests):
    def reference(self, rows=None, **extra):
        rows = rows or [("1", i * 10, ("0|0", "0|1", "1|1")) for i in range(1, 20)]
        vcf = self.root / "reference.vcf"
        prefix = self.root / "reference"
        writeVcf(vcf, rows)
        options = [
            value
            for key, arg in extra.items()
            for value in (f"--{key.replace('_', '-')}", str(arg))
        ]
        command("cluster", "--vcf", vcf, "--size", 8, "--medians", "--out", prefix, *options)
        return vcf, prefix, rows

    def test_native_phase_flags_inspect_every_call(self):
        path = self.root / "phase.vcf"
        writeVcf(
            path,
            [
                ("1", 1, ("0/0", "0|1", "1/1")),
                ("1", 2, ("0/1", "0|1", "./1")),
                ("1", 3, ("1|0", ".|1", "./.")),
            ],
        )
        bcf = self.root / "phase.bcf"
        toBcf(path, bcf)
        for file in (path, bcf):
            with Reader(file, phased=False, save=True) as reader:
                G = np.empty((3, 6), np.uint8)
                pos, rid, absent = (
                    np.empty(3, np.int64),
                    np.empty(3, np.int32),
                    np.empty(3, np.uint8),
                )
                phase = np.empty((3, 3), np.uint8)
                self.assertEqual(reader.read_into(G, pos, rid, absent, phase), 3)
                np.testing.assert_array_equal(phase, [[0, 0, 0], [1, 0, 1], [0, 0, 0]])
                self.assertEqual(
                    reader.sites, b"".join(f"1\t{i}\tA\tG\n".encode() for i in (1, 2, 3))
                )
        with Reader(path) as reader:
            with self.assertRaisesRegex(ValueError, "Unphased"):
                reader.read_into(G, pos, rid, absent)

    def test_unindexed_parallel_mixed_phasing_and_missingness(self):
        _, reference, rows = self.reference()
        rows[0] = ("1", 10, ("0/0", "0|1", "1/1"))
        rows[3] = ("1", 40, ("0|0", "0/1", "1|1"))
        rows[5] = ("1", 60, (".|0", "0|1", "./1"))
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
                labels(out)[0] == 255, [True, False, False, False, True, True]
            )
            self.assertEqual(readLog(out)["Unphased sample windows"], "2")
        result = command(
            "predict",
            "--vcf",
            vcf,
            "--ref",
            reference,
            "--phase-mode",
            "phased",
            "--out",
            self.root / "strict",
            success=False,
        )
        self.assertIn("Unphased", result.stderr)
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

    def test_plink_matches_unphased_vcf_and_checks_alleles(self):
        _, reference, rows = self.reference()
        vcf = self.root / "unphased.vcf"
        rows = [(c, p, ("0/0", "0/1", "1/1")) for c, p, _ in rows]
        rows[3] = ("1", 40, ("0/0", "./.", "1/1"))
        writeVcf(vcf, rows)
        bfile = self.root / "input"
        Path(f"{bfile}.fam").write_text("".join(f"0 {s} 0 0 0 -9\n" for s in ("A", "B", "C")))
        Path(f"{bfile}.bim").write_text(
            "".join(f"1 v{i} 0 {p} G A\n" for i, (_, p, _) in enumerate(rows))
        )
        encoded = bytes(3 | ((1 if i == 3 else 2) << 2) for i in range(len(rows)))
        Path(f"{bfile}.bed").write_bytes(bytes((108, 27, 1)) + encoded)
        outputs = []
        for option, source in (("--vcf", vcf), ("--bfile", bfile)):
            out = self.root / option[2:]
            command(
                "predict",
                option,
                source,
                "--ref",
                reference,
                "--phase-mode",
                "unphased",
                "--out",
                out,
            )
            outputs.append(labels(out))
        np.testing.assert_array_equal(*outputs)
        path = Path(f"{bfile}.bim")
        path.write_text(path.read_text().replace(" G A", " A G"))
        result = command(
            "predict",
            "--bfile",
            bfile,
            "--ref",
            reference,
            "--out",
            self.root / "fail",
            success=False,
        )
        self.assertIn("REF/ALT", result.stderr)

    def test_all_missing_reference_window_predicts_missing(self):
        rows = [("1", i, (".|.", ".|.", ".|.")) for i in range(1, 9)]
        vcf, reference, _ = self.reference(rows)
        out = self.root / "empty"
        command("predict", "--vcf", vcf, "--ref", reference, "--out", out, "--plink")
        self.assertTrue(np.all(labels(out) == 255))
        self.assertEqual(Path(f"{out}.bed").read_bytes(), bytes((108, 27, 1)))


if __name__ == "__main__":
    unittest.main()
