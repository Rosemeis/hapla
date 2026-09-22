"""Regression checks for the five findings in the cluster/predict/struct audit."""

import json
import shutil
from pathlib import Path

import numpy as np
from helpers import (
    TemporaryTests,
    command,
    missingFixture,
    structureFixture,
    writeClusters,
    writeVcf,
)

from hapla import struct
from hapla import struct_cy as cy
from hapla.formats import MAGIC, readMetadata
from hapla.identity import featureKeys, writeIdentity


def export(ref, out, *args, success=True):
    return command(
        "struct",
        "--clusters",
        ref,
        "--pca",
        2,
        "--power",
        2,
        "--loadings",
        "--raw",
        "--out",
        out,
        *args,
        success=success,
    )


def bind(pfx, reference, W, M):
    out = {s: Path(f"{pfx}{s}") for s in (".bca", ".ids", ".win", ".ref.json")}
    writeIdentity(out, reference, W, M)


class StructureContracts(TemporaryTests):
    def test_variation_matches_dense_dosages_and_rejects_zero_signal(self):
        for Z, c, p, D, _ in (structureFixture(), missingFixture()):
            v = np.empty_like(p)
            cy.variation(Z, c, p, v)
            np.testing.assert_allclose(v, ((D - 2 * p[:, None]) ** 2).sum(axis=1), atol=1e-13)
        N, W = 17, 12
        Z = np.tile([0, 1], (W, N)).astype(np.uint8)
        c = np.arange(0, 2 * W + 1, 2, dtype=np.int64)
        for missing in (False, True):
            if missing:
                Z.reshape(W, N, 2)[np.random.default_rng(73).random((W, N)) < 0.3] = 255
            p = np.full(2 * W, 0.5)
            obs = np.count_nonzero(Z != 255, axis=1).astype(np.int64)
            v = np.empty_like(p)
            cy.variation(Z, c, p, v)
            np.testing.assert_array_equal(v, 0)
            with self.assertRaisesRegex(ValueError, "empirical dosage variation"):
                struct.pca([(Z, c, obs)], p, 1, 7, 2, 42)
            ref = writeClusters(self.root, f"zero{missing}", Z, c)
            out = self.root / "zero"
            Path(f"{out}.eigenvecs").write_text("previous\n")
            result = command(
                "struct", "--clusters", ref, "--pca", 1, "--chunk", 7, "--out", out, success=False
            )
            self.assertIn("empirical dosage variation", result.stderr)
            self.assertEqual(Path(f"{out}.eigenvecs").read_text(), "previous\n")

    def test_pca_replacement_removes_the_entire_obsolete_model(self):
        Z, c, _, _, _ = structureFixture()
        ref = writeClusters(self.root, "ref", Z, c)
        model = self.root / "model"
        export(ref, model)
        command("struct", "--clusters", ref, "--pca", 1, "--raw", "--out", model)
        for s in (".loadings", ".freqs", ".pca.json"):
            self.assertFalse(Path(f"{model}{s}").exists())
        result = command(
            "struct",
            "--clusters",
            ref,
            "--projection",
            model,
            "--out",
            self.root / "project",
            success=False,
        )
        self.assertIn("Missing identity metadata", result.stderr)

    def test_failed_replacement_and_obsolete_input_conflicts_preserve_model(self):
        Z, c, _, _, _ = structureFixture()
        ref = writeClusters(self.root, "ref", Z, c)
        model = self.root / "model"
        export(ref, model)
        suffixes = (".eigenvals", ".eigenvecs", ".loadings", ".freqs", ".pca.json", ".log")
        previous = {s: Path(f"{model}{s}").read_bytes() for s in suffixes}
        result = command("struct", "--clusters", ref, "--pca", 999, "--out", model, success=False)
        self.assertNotEqual(result.returncode, 0)
        result = command(
            "struct",
            "--clusters",
            ref,
            "--pca",
            1,
            "--projection",
            model,
            "--out",
            model,
            success=False,
        )
        self.assertIn("conflicts with input", result.stderr)
        for s, original in previous.items():
            self.assertEqual(Path(f"{model}{s}").read_bytes(), original)

    def test_projection_rejects_reordered_files_and_different_reference_labels(self):
        Z, c, _, _, _ = structureFixture()
        a = writeClusters(self.root, "a", Z[:5], c[:6])
        b = writeClusters(self.root, "b", Z[5:], c[5:] - c[5])
        files, reverse = self.root / "files", self.root / "reverse"
        files.write_text(f"{a}\n{b}\n")
        reverse.write_text(f"{b}\n{a}\n")
        model = self.root / "model"
        command("struct", "--filelist", files, "--pca", 2, "--loadings", "--out", model)
        result = command(
            "struct",
            "--filelist",
            reverse,
            "--projection",
            model,
            "--out",
            self.root / "bad",
            success=False,
        )
        self.assertIn("Ordered cluster references", result.stderr)

        # Equal coordinates/counts alone do not establish the meaning of labels.
        ref = writeClusters(self.root, "original", Z, c)
        export(ref, model)
        other = writeClusters(self.root, "independent", structureFixture(seed=981)[0], c)
        self.assertEqual(Path(f"{ref}.win").read_bytes(), Path(f"{other}.win").read_bytes())
        result = command(
            "struct",
            "--clusters",
            other,
            "--projection",
            model,
            "--out",
            self.root / "bad",
            success=False,
        )
        self.assertIn("Ordered cluster references", result.stderr)

    def test_projection_rejects_mixed_model_and_assignment_files(self):
        Z, c, _, _, _ = structureFixture()
        ref = writeClusters(self.root, "ref", Z, c)
        other = writeClusters(self.root, "other", structureFixture(seed=85)[0], c)
        identity = json.loads(Path(f"{ref}.ref.json").read_text())["reference"]
        bind(other, identity, len(Z), int(c[-1]))
        a, b = self.root / "a", self.root / "b"
        export(ref, a)
        export(other, b)
        for suffix in (".freqs", ".loadings", ".eigenvals"):
            original = Path(f"{a}{suffix}").read_bytes()
            Path(f"{a}{suffix}").write_bytes(Path(f"{b}{suffix}").read_bytes())
            result = command(
                "struct",
                "--clusters",
                ref,
                "--projection",
                a,
                "--out",
                self.root / "bad",
                success=False,
            )
            self.assertIn("identity metadata", result.stderr)
            Path(f"{a}{suffix}").write_bytes(original)
        for suffix, change in (
            (
                ".bca",
                lambda x: (
                    x[: len(MAGIC) + 34] + bytes([1 - x[len(MAGIC) + 34]]) + x[len(MAGIC) + 35 :]
                ),
            ),
            (".win", lambda x: x.replace(b"1 2 2", b"2 2 2", 1)),
            (".ids", lambda x: x.replace(b"s0\n", b"new\n", 1)),
        ):
            original = Path(f"{ref}{suffix}").read_bytes()
            Path(f"{ref}{suffix}").write_bytes(change(original))
            result = command(
                "struct",
                "--clusters",
                ref,
                "--projection",
                a,
                "--out",
                self.root / "bad",
                success=False,
            )
            self.assertIn("identity metadata", result.stderr)
            Path(f"{ref}{suffix}").write_bytes(original)

    def test_missing_identity_is_rejected_and_moving_a_bundle_is_safe(self):
        Z, c, _, _, _ = structureFixture()
        ref = writeClusters(self.root, "ref", Z, c)
        renamed = self.root / "renamed"
        for s in (".bca", ".win", ".ids", ".ref.json"):
            shutil.copyfile(f"{ref}{s}", f"{renamed}{s}")
        self.assertEqual(
            featureKeys([ref], np.diff(c), [len(Z)]), featureKeys([renamed], np.diff(c), [len(Z)])
        )
        Path(f"{renamed}.ref.json").unlink()
        result = export(renamed, self.root / "model", success=False)
        self.assertIn("Missing identity metadata", result.stderr)
        result = command("struct", "--clusters", ref, "--projection", "", success=False)
        self.assertIn("prefix must not be empty", result.stderr)
        for ids in ("s0\n\ns1\n", "\n", "s 0\ns1\n", "s0\ns0\n"):
            Path(f"{ref}.ids").write_text(ids)
            with self.assertRaises(ValueError):
                readMetadata(ref)

    def test_grm_counts_and_scaling_metadata_match_the_defined_contrasts(self):
        for tag, parts in (("complete", structureFixture()), ("missing", missingFixture())):
            Z, c, p, D, _ = parts
            ref = writeClusters(self.root, tag, Z, c)
            out = self.root / f"{tag}-grm"
            command("struct", "--clusters", ref, "--grm", "--out", out)
            count = sum(max(0, len(set(z.tolist()) - {255}) - 1) for z in Z)
            np.testing.assert_array_equal(np.fromfile(f"{out}.grm.N.bin", np.float32), count)
            info = json.loads(Path(f"{out}.grm.meta.json").read_text())
            self.assertEqual(info["count_unit"], "categorical_contrasts")
            self.assertEqual(info["count"], count)
            self.assertFalse(info["count_is_pairwise_observed"])
            self.assertAlmostEqual(info["normalization_denominator"], 2 * np.sum(p * (1 - p)))
            E = D - 2 * p[:, None]
            raw = E.T @ E / info["normalization_denominator"]
            raw = raw - raw.mean(axis=0) - raw.mean(axis=1)[:, None] + raw.mean()
            expected = raw * info["gower_scale"]
            np.testing.assert_allclose(
                np.fromfile(f"{out}.grm.bin", np.float32),
                expected[np.tril_indices(D.shape[1])],
                atol=5e-7,
            )

    def test_native_prediction_propagates_reference_identity_to_new_missing_samples(self):
        rng = np.random.default_rng(39)
        vcf = self.root / "ref.vcf"
        G = rng.integers(0, 2, (24, 3, 2))
        rows = [("1", i + 1, tuple(f"{a}|{b}" for a, b in row)) for i, row in enumerate(G)]
        writeVcf(vcf, rows, samples=("A#1", "A#2", "B"))
        ref = self.root / "ref"
        command(
            "cluster",
            "--vcf",
            vcf,
            "--size",
            2,
            "--min-mac",
            1,
            "--medians",
            "--threads",
            4,
            "--out",
            ref,
        )
        model = self.root / "model"
        export(ref, model)
        query = self.root / "query.vcf"
        writeVcf(
            query, [(c, p, (calls[2], ".|.")) for c, p, calls in rows], samples=("new#1", "missing")
        )
        pred = self.root / "pred"
        command("predict", "--vcf", query, "--ref", ref, "--threads", 4, "--out", pred)
        a, b = [json.loads(Path(f"{x}.ref.json").read_text()) for x in (ref, pred)]
        self.assertEqual(a["reference"], b["reference"])
        self.assertNotEqual(a["files"][".bca"], b["files"][".bca"])
        self.assertEqual(readMetadata(ref)[1].tolist(), ["A#1", "A#2", "B"])
        out = self.root / "project"
        command("struct", "--clusters", pred, "--projection", model, "--raw", "--out", out)
        V = np.loadtxt(f"{out}.project.eigenvecs")
        np.testing.assert_allclose(V[1], 0, atol=1e-13)
        np.testing.assert_allclose(V[0], np.loadtxt(f"{model}.eigenvecs")[2], atol=2e-9)
