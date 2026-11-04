"""Shared metadata, output ownership, and native reader lifecycle."""

__author__ = "Jonas Meisner"

from contextlib import ExitStack
from io import StringIO
from pathlib import Path

import numpy as np
from hapla.vcf_cy import Reader
from helpers import TemporaryTests, command, writeVcf

from hapla.formats import readMetadata, readPaths, readQ, writeMatrix
from hapla.runtime import commitOutputs, stageOutputs


### Check ordered metadata and shared input validation
class SharedInputTests(TemporaryTests):
    def fixture(self, name, ids, counts):
        pfx = self.root / name
        Path(f"{pfx}.ids").write_text("\n".join(ids) + "\n")
        Path(f"{pfx}.bca").touch()
        Path(f"{pfx}.win").write_text(
            "#CHROM START END LENGTH SIZE K\n"
            + "".join(f"1 {i + 1} {i + 1} 0 1 {k}\n" for i, k in enumerate(counts))
        )
        return pfx

    def test_numeric_blocks_match_numpy_formatting_exactly(self):
        rng = np.random.default_rng(716)
        X = rng.standard_normal((4099, 7))
        X[0] = [0, -0.0, np.inf, -np.inf, np.nan, 1e-300, 1e300]
        cases = [
            (X, "%.10g"),
            (X[::3, ::2], "%.10g"),
            (X[:, 1], "%.10g"),
            (X[1:].astype(np.float32), "%.10g"),
            (np.empty((0, 3)), "%.10g"),
            (np.empty((4, 0)), "%.10g"),
            (np.arange(-12000, 12000).reshape(-1, 4), "%d"),
        ]
        for values, fmt in cases:
            with self.subTest(shape=values.shape, dtype=values.dtype, fmt=fmt):
                expected, actual = StringIO(), StringIO()
                np.savetxt(expected, values, fmt=fmt)
                writeMatrix(actual, values, fmt)
                self.assertFalse(actual.closed)
                self.assertEqual(actual.getvalue(), expected.getvalue())
                path = self.root / "matrix"
                writeMatrix(path, values, fmt)
                self.assertEqual(path.read_text(), expected.getvalue())

    def test_shared_q_reader_preserves_command_bounds_and_normalization(self):
        path = self.root / "model.Q"
        Q = np.array([[0.2, 0.800000001], [0.4, 0.6]])
        np.savetxt(path, Q)
        np.testing.assert_array_equal(readQ(path, 2), Q / Q.sum(axis=1, keepdims=True))
        for bad in (Q * 2, Q * np.nan, -Q, Q[:1]):
            np.savetxt(path, bad)
            with self.assertRaises(ValueError):
                readQ(path, 2)
        np.savetxt(path, np.ones((2, 1)))
        np.testing.assert_array_equal(readQ(path, 2), 1)
        with self.assertRaises(ValueError):
            readQ(path, 2, minimum=2)
        np.savetxt(path, np.full((2, 256), 1 / 256))
        with self.assertRaises(ValueError):
            readQ(path, 2)
        self.assertEqual(readQ(path, 2, minimum=2, maximum=None).shape, (2, 256))

    def test_metadata_preserves_sample_and_chromosome_order(self):
        a = self.fixture("a", ["s2", "s1"], [2])
        b = self.fixture("b with spaces", ["s2", "s1"], [3, 4])
        pth = self.root / "files"
        pth.write_text(f"\n {b} \r\n\n{a}\n")
        for values, files in ((None, pth), ([b, a], None)):
            paths, ids, counts, windows = readMetadata(values, files)
            self.assertEqual(list(map(str, paths)), [str(b), str(a)])
            np.testing.assert_array_equal(ids, ["s2", "s1"])
            np.testing.assert_array_equal(counts, [3, 4, 2])
            np.testing.assert_array_equal(windows, [2, 1])
            self.assertEqual(counts.dtype, np.uint32)
            self.assertEqual(windows.dtype, np.uint32)
        _, _, counts, windows = readMetadata(a)
        np.testing.assert_array_equal(counts, [2])
        np.testing.assert_array_equal(windows, [1])

    def test_metadata_rejects_mismatched_samples_and_missing_likelihoods(self):
        a = self.fixture("a", ["x", "y"], [2])
        b = self.fixture("b", ["x"], [2])
        pth = self.root / "files"
        pth.write_text(f"{a}\n{b}\n")
        for ids in (["x"], ["y", "x"]):
            Path(f"{b}.ids").write_text("\n".join(ids) + "\n")
            with self.assertRaisesRegex(ValueError, "Samples do not match"):
                readMetadata(None, pth)
        with self.assertRaisesRegex(ValueError, "blk file"):
            readMetadata(a, likes=True)
        pth.write_text(" \n\t\n")
        with self.assertRaisesRegex(ValueError, "empty"):
            readMetadata(None, pth)
        for values in ([], [""], [" "]):
            with self.assertRaisesRegex(ValueError, "empty"):
                readMetadata(values)

    def test_output_prefix_requires_a_filename(self):
        for pfx in ("", ".", "..", str(self.root) + "/", str(self.root) + "/."):
            with ExitStack() as stack, self.assertRaisesRegex(ValueError, "Output prefix"):
                stageOutputs(stack, pfx, (".Q",), stale=())

    def test_multifile_admix_matches_concatenated_assignments(self):
        vcf = self.root / "input.vcf"
        calls = ("0|0", "0|1", "1|1")
        writeVcf(vcf, [("1", i + 1, calls) for i in range(32)])
        ref = self.root / "ref"
        command("cluster", "--vcf", vcf, "--size", 8, "--min-mac", 1, "--out", ref)

        # Repeating the same chromosome doubles the data, with identical sample order
        from hapla.formats import MAGIC

        combined = self.root / "combined"
        raw = Path(f"{ref}.bca").read_bytes()[len(MAGIC) :]
        Path(f"{combined}.bca").write_bytes(MAGIC + raw + raw)
        Path(f"{combined}.win").write_text(Path(f"{ref}.win").read_text() * 2)
        Path(f"{combined}.ids").write_bytes(Path(f"{ref}.ids").read_bytes())
        pth = self.root / "files"
        pth.write_text(f"{ref}\n{ref}\n")
        results = []
        for i, opt in enumerate(
            (("--filelist", pth), ("--clusters", combined), ("--clusters", ref, ref))
        ):
            out = f"admix{i}"
            command(
                "admix",
                *opt,
                "--K",
                2,
                "--random-init",
                "--iter",
                5,
                "--batches",
                1,
                "--check",
                1,
                "--seed",
                1,
                "--out",
                out,
                cwd=self.root,
            )
            results.append(np.loadtxt(self.root / f"{out}.K2.s1.Q"))
        for Q in results[1:]:
            np.testing.assert_array_equal(results[0], Q)
        P = np.concatenate([np.loadtxt(self.root / f"admix0.K2.s1.chr{i}.P") for i in (1, 2)])
        np.testing.assert_array_equal(P, np.loadtxt(self.root / "admix1.K2.s1.P"))
        np.testing.assert_array_equal(
            P, np.concatenate([np.loadtxt(self.root / f"admix2.K2.s1.chr{i}.P") for i in (1, 2)])
        )
        paths = readPaths(self.root / "admix0.K2.s1.plist")
        self.assertTrue(all(Path(p).is_absolute() and Path(p).is_file() for p in paths))
        command(
            "admix",
            "--filelist",
            pth,
            "--K",
            2,
            "--random-init",
            "--iter",
            1,
            "--batches",
            1,
            "--check",
            1,
            "--seed",
            1,
            "--no-freqs",
            "--out",
            "admix0",
            cwd=self.root,
        )
        for sfx in (".P", ".plist", ".chr1.P", ".chr2.P"):
            self.assertFalse((self.root / f"admix0.K2.s1{sfx}").exists())


### Check reader cleanup and output ownership
class OwnershipTests(TemporaryTests):
    def test_native_reader_close_is_idempotent_and_state_is_readonly(self):
        pth = self.root / "input.vcf"
        writeVcf(pth, [("1", 1, ("0|0", "0|1", "1|1"))])
        src = Reader(pth)
        self.assertFalse(src.finished)
        self.assertTrue(src.htslib_version)
        for key in ("finished", "htslib_version"):
            with self.assertRaises(AttributeError):
                setattr(src, key, None)
        src.close()
        src.close()
        with self.assertRaisesRegex(ValueError, "closed"):
            src.readInto(
                np.empty((1, 6), np.uint8),
                np.empty(1, np.int64),
                np.empty(1, np.int32),
                np.empty(1, np.uint8),
            )

    def test_incomplete_publication_preserves_optional_outputs(self):
        pfx = self.root / "output"
        for sfx in (".bca", ".bcm"):
            Path(f"{pfx}{sfx}").write_bytes(b"previous")
        with ExitStack() as stack:
            out = stageOutputs(stack, pfx, (".bca", ".win"))
            out[".bca"].write_bytes(b"replacement")
            with self.assertRaisesRegex(RuntimeError, "Incomplete"):
                commitOutputs(pfx, out)
        for sfx in (".bca", ".bcm"):
            self.assertEqual(Path(f"{pfx}{sfx}").read_bytes(), b"previous")
        self.assertFalse(list(self.root.glob(".hapla-*")))
