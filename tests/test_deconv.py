"""Cluster and SNP ancestry-copy output contracts."""

import unittest
from pathlib import Path

import numpy as np

from helpers import TemporaryTests, command, writeClusters, writeVcf

from hapla.formats import mapLabels


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
            np.array([[0, 0, 1, 1], [0, 0, 1, 1], [0, 1, 0, 1], [0, 1, 0, 1], [1, 1, 1, 1], [1, 1, 1, 1]]),
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

    def call(self, *extra):
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

    def test_support_and_short_tract_filters_apply_before_vcf_output(self):
        support = self.root / "support"
        prob = np.ones((2 * self.N, self.W))
        prob[0, 2] = 0.9
        np.savetxt(support, prob)
        supports = self.root / "supports"
        supports.write_text(f"{support}\n")
        self.call(
            "--format",
            "vcf",
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
        rows = Path(f"{self.root / 'out.chr1'}.vcf").read_text().splitlines()
        header = next(row for row in rows if row.startswith("#CHROM")).split("\t")
        s0 = header.index("s0_0")
        records = [row.split("\t") for row in rows if not row.startswith("#")]
        self.assertEqual(records[0][s0], "0|1")
        self.assertEqual(records[2][s0], ".|.")


if __name__ == "__main__":
    unittest.main()
