"""Cluster and SNP ancestry-copy output contracts."""

__author__ = "Jonas Meisner"

import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from helpers import TemporaryTests, command, toBcf, writeClusters, writeVcf

from hapla import deconv, vcf_cy
from hapla.formats import mapLabels
from hapla.identity import readIdentity


### Check ancestry expansion, native BCF alignment, and output publication
class DeconvolutionTests(TemporaryTests):
    def setUp(self):
        super().setUp()
        self.N, self.W, self.K = 3, 4, 2
        self.Z = np.array(
            [[0, 1, 1, 0, 0, 1], [1, 0, 0, 1, 1, 0], [0, 0, 1, 1, 0, 1], [1, 1, 0, 0, 1, 0]],
            np.uint8,
        )
        self.pfx = writeClusters(self.root, "input.chr1", self.Z, np.array([0, 2, 4, 6, 8]))
        self.path = self.root / "path"
        np.savetxt(
            self.path,
            np.array(
                [[0, 0, 1, 1], [0, 0, 1, 1], [0, 1, 0, 1], [0, 1, 0, 1], [1, 1, 1, 1], [1, 1, 1, 1]]
            ),
            fmt="%d",
        )
        self.paths = self.root / "paths"
        self.paths.write_text(f"{self.path}\n")
        self.vcf = self.root / "input.vcf"
        writeVcf(
            self.vcf,
            [("1", i + 1, ("0|1", "1|0", "0|0")) for i in range(self.W)],
            samples=("s0", "s1", "s2"),
        )
        self.bcfs = self.root / "bcfs"
        self.bcfs.write_text(f"{self.vcf}\n")

    def call(self, *extra, success=True):
        return command(
            "deconv",
            "--clusters",
            self.pfx,
            "--path-filelist",
            self.paths,
            "--K",
            self.K,
            "--out",
            self.root / "out",
            *extra,
            success=success,
        )

    def test_cluster_output_retains_originals_and_masks_nonmatching_haplotypes(self):
        self.call("--include-original", "--min-fraction", "0")
        ids = Path(f"{self.root / 'out.chr1'}.ids").read_text().splitlines()
        self.assertEqual(ids, ["s0", "s1", "s2", "s0_0", "s0_1", "s1_0", "s1_1", "s2_0", "s2_1"])
        got = mapLabels(self.root / "out.chr1", self.W, len(ids))
        np.testing.assert_array_equal(got[:, : 2 * self.N], self.Z)
        self.assertEqual(got[0, 6], self.Z[0, 0])
        self.assertEqual(got[2, 6], 255)
        self.assertEqual(got[2, 7], 255)
        command(
            "admix",
            "--clusters",
            self.root / "out.chr1",
            "--K",
            "2",
            "--power",
            "1",
            "--out",
            self.root / "fit",
        )

    def test_support_and_short_tract_filters_apply_before_bcf_output(self):
        writeVcf(
            self.vcf,
            [("1", i + 1, ("0|1:0.1,0.9", "1|0:0.1,0.9", "0|0:0.1,0.9")) for i in range(self.W)],
            fields="GT:PP",
            samples=("s0", "s1", "s2"),
        )
        support = self.root / "support"
        prob = np.ones((2 * self.N, self.W))
        prob[0, 2] = 0.9
        np.savetxt(support, prob)
        supports = self.root / "supports"
        supports.write_text(f"{support}\n")
        self.call(
            "--format",
            "bcf",
            "--bcf-filelist",
            self.bcfs,
            "--support-filelist",
            supports,
            "--min-call-support",
            "0.95",
            "--min-tract-windows",
            "2",
            "--min-fraction",
            "0",
        )
        output = Path(f"{self.root / 'out.chr1'}.bcf")
        toBcf(output, self.root / "out.vcf", mode=b"w")
        rows = (self.root / "out.vcf").read_text().splitlines()
        self.assertFalse(any("ID=PP" in row for row in rows if row.startswith("##")))
        header = next(row for row in rows if row.startswith("#CHROM")).split("\t")
        s0 = header.index("s0_0")
        records = [row.split("\t") for row in rows if not row.startswith("#")]
        self.assertEqual(records[0][s0], "0|1")
        self.assertEqual(records[2][s0], ".|.")

    def test_bcf_removes_all_sample_fields_and_keeps_original_calls(self):
        writeVcf(
            self.vcf,
            [("1", i + 1, ("0|1:8", "1|0:9", "0|0:10")) for i in range(self.W)],
            fields="GT:DP",
            samples=("s0", "s1", "s2"),
        )
        self.call("--format", "both", "--bcf-filelist", self.bcfs, "--include-original")
        out = self.root / "decoded.vcf"
        toBcf(self.root / "out.chr1.bcf", out, mode=b"w")
        rows = [
            line.split("\t") for line in out.read_text().splitlines() if not line.startswith("#")
        ]
        self.assertTrue(all(row[8] == "GT" for row in rows))
        self.assertTrue(all(row[9:12] == ["0|1", "1|0", "0|0"] for row in rows))
        self.assertNotIn("ID=DP", out.read_text())

    def test_bcf_rejects_reordered_samples_and_shifted_coordinates(self):
        for names, shift, expected in (
            (("s1", "s0", "s2"), 0, "sample order"),
            (("s0", "s1", "s2"), 1, "coordinates"),
        ):
            with self.subTest(names=names, shift=shift):
                writeVcf(
                    self.vcf,
                    [("1", i + 1 + shift, ("0|1", "1|0", "0|0")) for i in range(self.W)],
                    samples=names,
                )
                res = self.call("--format", "bcf", "--bcf-filelist", self.bcfs, success=False)
                self.assertIn(expected, res.stderr)
                self.assertFalse((self.root / "out.chr1.bcf").exists())

    def test_failed_bcf_preserves_previous_cluster_outputs(self):
        self.call()
        paths = [
            self.root / f"out{s}"
            for s in (".filelist", ".log", ".chr1.bca", ".chr1.ids", ".chr1.win", ".chr1.ref")
        ]
        saved = [p.read_bytes() for p in paths]
        writeVcf(
            self.vcf,
            [("1", i + 1, ("0/1", "1|0", "0|0")) for i in range(self.W)],
            samples=("s0", "s1", "s2"),
        )
        res = self.call("--format", "both", "--bcf-filelist", self.bcfs, success=False)
        self.assertIn("phased", res.stderr)
        self.assertEqual([p.read_bytes() for p in paths], saved)
        self.assertFalse((self.root / "out.bcfs").exists())

    def test_outputs_cannot_overwrite_cluster_inputs(self):
        saved = Path(f"{self.pfx}.bca").read_bytes()
        res = self.call("--out", self.root / "input", success=False)
        self.assertIn("conflicts with input", res.stderr)
        self.assertEqual(Path(f"{self.pfx}.bca").read_bytes(), saved)

    def test_cluster_identity_and_window_bytes_are_preserved(self):
        self.call("--out", self.root / "nested" / "out")
        dst = self.root / "nested" / "out.chr1"
        self.assertEqual(Path(f"{dst}.win").read_bytes(), Path(f"{self.pfx}.win").read_bytes())
        src = readIdentity(self.pfx, (".bca", ".ids", ".win"))
        out = readIdentity(dst, (".bca", ".ids", ".win"))
        self.assertEqual(src["reference"], out["reference"])

    def test_filters_reject_invalid_labels_support_and_extra_rows(self):
        args = SimpleNamespace(
            K=2, min_call_support=0.5, homozygous_only=False, min_tract_windows=None
        )
        for value in (65536, -2, 2):
            np.savetxt(self.path, np.full((6, 4), value), fmt="%d")
            with self.assertRaises(ValueError):
                deconv.filterPath(self.path, None, 3, 4, args)
        np.savetxt(self.path, np.zeros((7, 4)), fmt="%d")
        with self.assertRaisesRegex(ValueError, "dimensions"):
            deconv.filterPath(self.path, None, 3, 4, args)
        np.savetxt(self.path, np.zeros((6, 4)), fmt="%d")
        support = self.root / "support"
        np.savetxt(support, np.full((6, 4), 1.1))
        with self.assertRaisesRegex(ValueError, "support"):
            deconv.filterPath(self.path, support, 3, 4, args)
        args.min_tract_windows = 3
        out = deconv.filterPath(self.path, None, 3, 4, args, breaks=[2])
        np.testing.assert_array_equal(out, -1)

    def test_native_expansion_and_counts_match_dense_reference(self):
        rng = np.random.default_rng(512)
        N, W, K = 7, 13, 3
        labels = rng.integers(4, size=(W, 2 * N), dtype=np.uint8)
        path = rng.integers(-1, K, size=(2 * N, W), dtype=np.int16)
        widths = rng.integers(1, 15, size=W, dtype=np.int64)
        counts = np.zeros((N, K), np.int64)
        vcf_cy.deconvWeights(path, widths, counts)
        expected = np.column_stack(
            [((path[::2] == k).astype(np.int64) + (path[1::2] == k)) @ widths for k in range(K)]
        )
        np.testing.assert_array_equal(counts, expected)
        selected = np.array([(i, k) for i in range(N) for k in range(K)], np.int32)
        for original in (False, True):
            expected = np.full((W, 2 * (len(selected) + N * original)), 255, np.uint8)
            off = 2 * N if original else 0
            if original:
                expected[:, :off] = labels
            for j, (i, k) in enumerate(selected):
                for h in (0, 1):
                    keep = path[2 * i + h] == k
                    expected[keep, off + 2 * j + h] = labels[keep, 2 * i + h]
            out = np.empty_like(expected)
            for beg in range(0, W, 5):
                end = min(W, beg + 5)
                vcf_cy.expandDeconv(labels[beg:end], path, selected, out[beg:end], beg, original)
            np.testing.assert_array_equal(out, expected)
        self.assertEqual(len(set(deconv.suffixes(["a.2", "b.2", "c.2_2"]))), 3)


if __name__ == "__main__":
    unittest.main()
