"""Native reader, window boundaries, versioned files, and streaming integration."""

import gzip
import importlib.util
import os
import struct
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import numpy as np
from hapla.vcf_cy import Reader
from helpers import TemporaryTests, command, toBcf, writeVcf

from hapla.formats import FORMAT_VERSION, MAGIC, readHeader
from hapla.runtime import commitOutputs, stageOutputs, threadPlan
from hapla.struct import readData, scale
from hapla.windows import createBuffer, fillBuffer, fixedWindows, physicalWindows, predefinedWindows


class ReaderTests(TemporaryTests):
    def test_phased_fast_path_matches_scalar_with_missing_and_odd_sample_count(self):
        samples = tuple(f"sample{i}" for i in range(33))
        calls = ("0|0", "0|1", "1|0", "1|1", ".|0", "0|.", ".|1", "1|.", ".|.")
        rows = [
            ("1", j + 1, tuple(calls[(i + j) % len(calls)] for i in range(33))) for j in range(9)
        ]
        vcf, bcf = self.root / "vector.vcf", self.root / "vector.bcf"
        writeVcf(vcf, rows, samples=samples)
        toBcf(vcf, bcf)
        expected = np.array(
            [
                [
                    255 if allele == "." else int(allele)
                    for call in row[2]
                    for allele in call.split("|")
                ]
                for row in rows
            ],
            dtype=np.uint8,
        )
        for path in (vcf, bcf):
            for phased in (True, False):
                G = np.empty_like(expected)
                pos, rid, absent = (
                    np.empty(9, np.int64),
                    np.empty(9, np.int32),
                    np.empty(9, np.uint8),
                )
                phase = None if phased else np.empty((9, 33), np.uint8)
                with Reader(path, phased=phased) as reader:
                    self.assertEqual(reader.read_into(G, pos, rid, absent, phase), 9)
                np.testing.assert_array_equal(G, expected)
                np.testing.assert_array_equal(absent, np.any(expected == 255, axis=1))
                if phase is not None:
                    self.assertFalse(np.any(phase))

    def read(self, path, capacity=2):
        rows, positions = [], []
        with Reader(path) as reader:
            G = np.empty((capacity, 2 * len(reader.samples)), np.uint8)
            pos = np.empty(capacity, np.int64)
            rid = np.empty(capacity, np.int32)
            missing = np.empty(capacity, np.uint8)
            while True:
                n = reader.read_into(G, pos, rid, missing)
                if not n:
                    break
                self.assertTrue(np.array_equal(missing[:n], np.any(G[:n] == 255, axis=1)))
                rows.append(G[:n].copy())
                positions.extend(pos[:n])
            self.assertTrue(reader.finished)
            self.assertEqual(reader.variants, len(positions))
        return np.concatenate(rows), positions

    def test_vcf_bcf_extra_fields_missing_and_reused_buffers(self):
        path = self.root / "gt.vcf"
        rows = [
            ("1", 1, ("2:0|0", "7:0|1", "9:1|1")),
            ("1", 2, ("2:.|1", "7:0|.", "9:./.")),
            ("2", 1, ("2:0/0", "7:1/1", "9:0|1")),
        ]
        writeVcf(path, rows, fields="DP:GT")
        bcf = self.root / "gt.bcf"
        toBcf(path, bcf)
        expected = [[0, 0, 0, 1, 1, 1], [255, 1, 0, 255, 255, 255], [0, 0, 1, 1, 0, 1]]
        for file in (path, bcf):
            actual, positions = self.read(file)
            np.testing.assert_array_equal(actual, expected)
            self.assertEqual(positions, [1, 2, 1])

    def test_wide_bcf_gt_encodings(self):
        vcf = self.root / "source.vcf"
        bcf = self.root / "source.bcf"
        writeVcf(vcf, [("1", 1, ("0|0", "0|1", "1|1"))])
        toBcf(vcf, bcf)
        raw = gzip.decompress(bcf.read_bytes())
        header_end = 9 + struct.unpack_from("<I", raw, 5)[0]
        shared, indiv = struct.unpack_from("<II", raw, header_end)
        begin = header_end + 8 + shared
        block = raw[begin : begin + indiv]
        self.assertEqual(block[0], 0x11)  # Scalar FORMAT ID in this controlled fixture.
        self.assertEqual(block[2], 0x21)  # Two INT8 values per sample.
        for encoding, format_string in ((2, "<h"), (3, "<i")):
            values = b"".join(struct.pack(format_string, value) for value in block[3:])
            altered = block[:2] + bytes((0x20 | encoding,)) + values
            fixture = self.root / f"wide{encoding}.bcf"
            fixture.write_bytes(
                raw[:header_end]
                + struct.pack("<II", shared, len(altered))
                + raw[header_end + 8 : begin]
                + altered
            )
            actual, _ = self.read(fixture)
            np.testing.assert_array_equal(actual, [[0, 0, 0, 1, 1, 1]])

    def test_invalid_phase_ploidy_alleles_sorting_and_missing_gt(self):
        cases = [
            (("0/1", "0|0", "1|1"), "Unphased"),
            (("0", "0|0", "1|1"), "diploid"),
            (("0|1|0", "0|0", "1|1"), "diploid"),
            (("0|2", "0|0", "1|1"), "allele index"),
            (("./1", "0|0", "1|1"), "Unphased"),
        ]
        for calls, expected in cases:
            path = self.root / "bad.vcf"
            writeVcf(path, [("1", 1, calls)])
            with self.assertRaisesRegex(ValueError, expected):
                self.read(path)
        path = self.root / "order.vcf"
        writeVcf(path, [("1", pos, ("0|0", "0|1", "1|1")) for pos in (2, 1)])
        with self.assertRaisesRegex(ValueError, "sorted"):
            self.read(path)
        writeVcf(path, [("1", 1, ("1", "2", "3"))], fields="DP")
        with self.assertRaisesRegex(ValueError, "FORMAT/GT"):
            self.read(path)


class WindowTests(TemporaryTests):
    def test_small_read_blocks_support_larger_fixed_and_variable_windows(self):
        path = self.root / "blocks.vcf"
        writeVcf(path, [("1", i + 1, ("0|0", "0|1", "1|1")) for i in range(20)])
        for method, arguments in (
            (fixedWindows, (8,)),
            (physicalWindows, (7,)),
            (predefinedWindows, ([0, 8, 16],)),
        ):
            with Reader(path) as reader:
                buffer = createBuffer(
                    reader.read_into, reader.samples, reader.contigs, 19 * 16, b_read=19 * 2
                )
                self.assertEqual(fillBuffer(buffer, 1), 2)
                self.assertEqual(reader.variants, 2)
                windows = list(method(buffer, *arguments))
                self.assertEqual([len(G) for _, G, _, _ in windows], [8, 8, 4])
                np.testing.assert_array_equal(
                    np.concatenate([G for _, G, _, _ in windows]),
                    np.tile([0, 0, 0, 1, 1, 1], (20, 1)),
                )

    def windows(self, count, method, *args, chrom_boundary=None, **kwargs):
        path = self.root / "windows.vcf"
        writeVcf(
            path,
            [
                (
                    "2" if chrom_boundary and i >= chrom_boundary else "1",
                    i + 1,
                    ("0|0", "0|1", "1|1"),
                )
                for i in range(count)
            ],
        )
        with Reader(path) as reader:
            buffer = createBuffer(reader.read_into, reader.samples, reader.contigs, (6 + 13) * 40)
            return [(meta[0], meta[1], len(G)) for meta, G, _, _ in method(buffer, *args, **kwargs)]

    def test_fixed_tails_overlap_and_small_inputs(self):
        self.assertEqual(
            self.windows(32, fixedWindows, 16, 8), [(0, "1", 16), (8, "1", 16), (16, "1", 16)]
        )
        self.assertEqual(self.windows(35, fixedWindows, 16, 8)[-1], (24, "1", 11))
        self.assertEqual(self.windows(32, fixedWindows, 16), [(0, "1", 16), (16, "1", 16)])
        self.assertEqual(self.windows(5, fixedWindows, 16), [(0, "1", 5)])
        self.assertEqual(self.windows(5, fixedWindows, 16, tail="drop"), [])
        self.assertEqual(self.windows(70, fixedWindows, 16)[-1], (64, "1", 6))

    def test_chromosome_boundaries_and_custom_eof_sentinel(self):
        self.assertEqual(
            self.windows(6, fixedWindows, 4, chrom_boundary=3), [(0, "1", 3), (3, "2", 3)]
        )
        for starts in ([0, 2], [0, 2, 6]):
            self.assertEqual(self.windows(6, predefinedWindows, starts), [(0, "1", 2), (2, "1", 4)])
        with self.assertRaisesRegex(ValueError, "crosses"):
            self.windows(6, predefinedWindows, [0, 4, 6], chrom_boundary=3)
        with self.assertRaisesRegex(ValueError, "exceeds"):
            self.windows(6, predefinedWindows, [0, 8])

    def test_physical_windows_and_buffer_limit(self):
        self.assertEqual(self.windows(8, physicalWindows, 3), [(0, "1", 4), (4, "1", 4)])
        with self.assertRaisesRegex(ValueError, "input buffer"):
            self.windows(100, fixedWindows, 41)


class FormatAndPipelineTests(TemporaryTests):
    def test_thread_budget_includes_htslib_coordinator(self):
        for total in range(1, 33):
            workers, io = threadPlan(total)
            used = workers + (total > 1) + io + bool(io)
            self.assertLessEqual(used, total)
        self.assertEqual(threadPlan(4), (3, 0))
        self.assertEqual(threadPlan(8), (4, 2))
        self.assertEqual(threadPlan(4, 1), (1, 1))
        with self.assertRaisesRegex(ValueError, "coordinator"):
            threadPlan(3, 1)

    def test_single_cluster_windows_have_zero_standardization_weight(self):
        scales = scale(np.array([0, 0.5, 1], dtype=np.float32))
        np.testing.assert_allclose(scales, [0, np.sqrt(2), 0])
        with self.assertRaisesRegex(ValueError, "variable"):
            scale(np.ones(2, dtype=np.float32))

    def test_old_formats_rejected_and_missing_labels_validated(self):
        prefix = self.root / "format"
        path = prefix.with_suffix(".bca")
        for old_magic in (bytes((7, 9, 13)), bytes((7, 9, 15))):
            path.write_bytes(old_magic + bytes((0, 255)))
            with self.assertRaisesRegex(ValueError, "before 1.0.0"):
                readData([prefix], np.array([255]), [1], 1)
        path.write_bytes(MAGIC + bytes((0, 255)))
        data, _ = readData([prefix], np.array([255]), [1], 1)
        np.testing.assert_array_equal(data[0][0], [[0, 255]])
        np.testing.assert_array_equal(data[0][2], [1])
        with self.assertRaisesRegex(ValueError, "0..255"):
            readData([prefix], np.array([256]), [1], 1)
        path.write_bytes(MAGIC + bytes((0, 3)))
        with self.assertRaisesRegex(ValueError, "outside"):
            readData([prefix], np.array([2]), [1], 1)
        path.write_bytes(MAGIC + bytes((0,)))
        with self.assertRaisesRegex(ValueError, "payload"):
            readData([prefix], np.array([2]), [1], 1)

    def test_retired_cluster_and_predict_flags_are_rejected(self):
        for subcommand, flag in (
            ("cluster", "--prune"),
            ("cluster", "--memory"),
            ("predict", "--memory"),
        ):
            result = command(subcommand, flag, success=False)
            self.assertEqual(result.returncode, 2)
            self.assertIn(f"unrecognized arguments: {flag}", result.stderr)

    def test_retired_native_modules_are_not_available(self):
        for name in ("hapla.cluster_cy", "hapla.memory_cy"):
            self.assertIsNone(importlib.util.find_spec(name), name)

    def test_parallel_streaming_and_optional_outputs(self):
        vcf = self.root / "parallel.vcf"
        rng = np.random.default_rng(741)
        rows = []
        for i in range(100):
            alleles = rng.integers(0, 2, 6)
            calls = [f"{alleles[j]}|{alleles[j + 1]}" for j in range(0, 6, 2)]
            if i % 11 == 0:
                calls[0] = ".|1"
            rows.append(("1", i + 1, calls))
        writeVcf(vcf, rows)
        prefixes = [self.root / "serial", self.root / "parallel"]
        for prefix, threads in zip(prefixes, (1, 4)):
            command(
                "cluster",
                "--vcf",
                vcf,
                "--size",
                16,
                "--step",
                8,
                "--medians",
                "--plink",
                "--buffer-mb",
                1,
                "--threads",
                threads,
                "--out",
                prefix,
            )
        for suffix in (
            ".bca",
            ".win",
            ".ids",
            ".bcm",
            ".blk",
            ".wix",
            ".sites",
            ".ref.json",
            ".bed",
            ".bim",
            ".fam",
        ):
            self.assertEqual(
                Path(f"{prefixes[0]}{suffix}").read_bytes(),
                Path(f"{prefixes[1]}{suffix}").read_bytes(),
                suffix,
            )
        with Path(f"{prefixes[0]}.bca").open("rb") as handle:
            self.assertEqual(readHeader(handle), FORMAT_VERSION)

    def test_bcf_allele_recoding_preserves_clusters_and_reference_prediction(self):
        rng = np.random.default_rng(952)
        G = rng.integers(0, 2, (130, 40), dtype=np.uint8)
        G[::3] = np.tile([0, 1], 20)
        G[1, 0] = G[80, 5] = 255
        flip = rng.integers(0, 2, len(G), dtype=np.uint8)
        refs = [self.root / "original", self.root / "recoded"]
        for ref, mask, threads in zip(refs, (np.zeros_like(flip), flip), (1, 4)):
            X = np.where(G == 255, 255, G ^ mask[:, None])
            rows = [
                (
                    "1",
                    j + 1,
                    [
                        "|".join("." if a == 255 else str(a) for a in pair)
                        for pair in row.reshape(-1, 2)
                    ],
                )
                for j, row in enumerate(X)
            ]
            vcf, bcf = Path(f"{ref}.vcf"), Path(f"{ref}.bcf")
            writeVcf(vcf, rows, samples=tuple(f"s{i}" for i in range(20)))
            lines = vcf.read_text().splitlines()
            for i, line in enumerate(lines):
                if not line.startswith("#"):
                    fields = line.split("\t")
                    if mask[int(fields[1]) - 1]:
                        fields[3], fields[4] = fields[4], fields[3]
                        lines[i] = "\t".join(fields)
            vcf.write_text("\n".join(lines) + "\n")
            toBcf(vcf, bcf)
            command(
                "cluster",
                "--bcf",
                bcf,
                "--size",
                65,
                "--min-mac",
                3,
                "--max-clusters",
                7,
                "--medians",
                "--plink",
                "--threads",
                threads,
                "--out",
                ref,
            )
            pred = Path(f"{ref}.predict")
            command("predict", "--bcf", bcf, "--ref", ref, "--threads", threads, "--out", pred)
            self.assertEqual(Path(f"{ref}.bca").read_bytes(), Path(f"{pred}.bca").read_bytes())
        for suffix in (".bca", ".win", ".ids", ".blk", ".bed", ".bim", ".fam"):
            self.assertEqual(
                Path(f"{refs[0]}{suffix}").read_bytes(), Path(f"{refs[1]}{suffix}").read_bytes()
            )
        counts = [
            int(row.split()[-1]) for row in Path(f"{refs[0]}.win").read_text().splitlines()[1:]
        ]
        A, B = [
            np.frombuffer(Path(f"{ref}.bcm").read_bytes()[len(MAGIC) :], np.uint8) for ref in refs
        ]
        off = 0
        for j, count in enumerate(counts):
            end = off + count * 65
            np.testing.assert_array_equal(
                A[off:end].reshape(count, 65) ^ flip[j * 65 : (j + 1) * 65],
                B[off:end].reshape(count, 65),
            )
            off = end

    def test_failures_preserve_existing_outputs(self):
        vcf = self.root / "fail.vcf"
        writeVcf(vcf, [("1", i + 1, ("0|0", "0|1", "1|1")) for i in range(8)])
        prefix = self.root / "out"
        target = Path(f"{prefix}.bca")
        target.write_bytes(b"previous output")
        result = command(
            "cluster",
            "--vcf",
            vcf,
            "--size",
            8,
            "--max-iterations",
            1,
            "--out",
            prefix,
            success=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("did not converge", result.stderr)
        self.assertEqual(target.read_bytes(), b"previous output")
        self.assertFalse(list(self.root.glob(".hapla-*")))

    def test_output_commit_rolls_back_mid_publish(self):
        prefix = self.root / "transaction"
        for suffix in (".bca", ".win"):
            Path(f"{prefix}{suffix}").write_bytes(b"old" + suffix.encode())
        replace = os.replace
        with ExitStack() as stack:
            output = stageOutputs(stack, prefix, (".bca", ".win"))
            output[".bca"].write_bytes(b"new bca")
            output[".win"].write_bytes(b"new win")

            def fail(source, destination):
                if Path(source).name == "output.win":
                    raise OSError("injected publish failure")
                return replace(source, destination)

            with patch("hapla.runtime.os.replace", side_effect=fail):
                with self.assertRaises(OSError):
                    commitOutputs(prefix, output)
        for suffix in (".bca", ".win"):
            self.assertEqual(Path(f"{prefix}{suffix}").read_bytes(), b"old" + suffix.encode())

    def test_new_medians_predict_roundtrip(self):
        vcf, bcf = self.root / "predict.vcf", self.root / "predict.bcf"
        writeVcf(vcf, [("1", i + 1, ("0|0", "0|1", "1|1")) for i in range(35)])
        toBcf(vcf, bcf, index=True)
        reference, target = self.root / "reference", self.root / "target"
        command("cluster", "--bcf", bcf, "--size", 16, "--step", 8, "--medians", "--out", reference)
        command("predict", "--bcf", bcf, "--ref", reference, "--out", target)
        self.assertEqual(Path(f"{reference}.bca").read_bytes(), Path(f"{target}.bca").read_bytes())

    def test_downstream_complete_and_missing_aware_readers(self):
        for missing in (False, True):
            vcf = self.root / f"downstream{missing}.vcf"
            prefix = self.root / f"clusters{missing}"
            admix = self.root / f"admix{missing}"
            calls = (".|0" if missing else "0|0", "0|1", "1|1")
            writeVcf(vcf, [("1", i + 1, calls) for i in range(32)])
            command("cluster", "--vcf", vcf, "--size", 8, "--medians", "--out", prefix)
            command(
                "admix",
                "--clusters",
                prefix,
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
                admix,
            )
            qfile = f"{admix}.K2.s1.Q"
            Q = np.loadtxt(qfile)
            self.assertEqual(Q.shape, (3, 2))
            self.assertTrue(np.all(np.isfinite(Q)))
            command(
                "eval",
                "--clusters",
                prefix,
                "--qfile",
                qfile,
                "--out",
                self.root / f"eval{missing}",
            )
            result = command(
                "struct", "--clusters", prefix, "--grm", "--out", self.root / f"struct{missing}"
            )
            result = command(
                "fatash",
                "--clusters",
                prefix,
                "--qfile",
                qfile,
                "--pfile",
                f"{admix}.K2.s1.P",
                "--fixed-model",
                "--medians",
                "--alpha",
                10,
                "--out",
                self.root / f"fatash{missing}",
            )
            self.assertEqual(result.returncode, 0)

    def test_prediction_preserves_partial_window_missingness(self):
        vcf, bcf = self.root / "ref.vcf", self.root / "ref.bcf"
        reference, target = self.root / "reference", self.root / "target"
        rows = [("1", i + 1, ("0|0", "0|1", "1|1")) for i in range(16)]
        writeVcf(vcf, rows)
        toBcf(vcf, bcf, index=True)
        command("cluster", "--bcf", bcf, "--size", 8, "--medians", "--out", reference)
        query_vcf, query_bcf = self.root / "query.vcf", self.root / "query.bcf"
        rows[10] = ("1", 11, ("0|0", ".|1", "1|1"))
        writeVcf(query_vcf, rows)
        toBcf(query_vcf, query_bcf, index=True)
        command("predict", "--bcf", query_bcf, "--ref", reference, "--plink", "--out", target)
        labels = np.frombuffer(Path(f"{target}.bca").read_bytes()[len(MAGIC) :], np.uint8).reshape(
            2, 6
        )
        np.testing.assert_array_equal(
            labels == 255, [[False] * 6, [False, False, True, False, False, False]]
        )
        # Two clusters per window, one byte per PLINK variant, second sample is missing.
        bed = Path(f"{target}.bed").read_bytes()[3:]
        self.assertTrue(all((byte >> 2) & 3 == 1 for byte in bed[2:]))

    def test_all_missing_windows_are_written_and_mapped(self):
        vcf = self.root / "missing.vcf"
        prefix = self.root / "missing"
        writeVcf(vcf, [("1", i + 1, ("./.", "./.", "./.")) for i in range(8)])
        command("cluster", "--vcf", vcf, "--size", 8, "--medians", "--plink", "--out", prefix)
        data, p = readData([prefix], np.array([0]), [1], 3)
        self.assertTrue(np.all(data[0][0] == 255))
        np.testing.assert_array_equal(data[0][2], [0])
        self.assertEqual(p.size, 0)
        self.assertEqual(Path(f"{prefix}.bcm").read_bytes(), MAGIC)
        self.assertEqual(Path(f"{prefix}.bed").read_bytes(), bytes((108, 27, 1)))


if __name__ == "__main__":
    unittest.main()
