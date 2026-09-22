"""Cross-command metadata boundaries and input/output validation."""

from contextlib import ExitStack
from pathlib import Path

import numpy as np
from helpers import TemporaryTests, command, writeClusters

from hapla.formats import readMetadata
from hapla.runtime import stageOutputs


class CodebaseContracts(TemporaryTests):
    def test_window_counts_and_dimensions_are_exact_integers(self):
        Z = np.tile([0, 1, 1, 0], (2, 1)).astype(np.uint8)
        ref = writeClusters(self.root, "ref", Z, np.array([0, 2, 4]))
        win = Path(f"{ref}.win")
        for row in (
            "1 1 4 3 4 2.0",
            "1 1 4 3 4 -1",
            "1 1 4 3 4 256",
            "1 1 4 2 4 2",
            "1 1 4 3 0 2",
            "1 1 4 3 4",
        ):
            win.write_text(row + "\n")
            with self.assertRaises(ValueError):
                readMetadata(ref)

    def test_optional_input_absence_does_not_block_output_replacement(self):
        out = self.root / "result"
        Path(f"{out}.log").write_text("previous")
        with ExitStack() as stack:
            stageOutputs(stack, out, [".log"], [self.root / "absent.ref.json"], stale=())
        self.assertEqual(Path(f"{out}.log").read_text(), "previous")

    def test_admix_rejects_ambiguous_shapes_and_sample_subsets(self):
        rng = np.random.default_rng(491)
        ref = writeClusters(
            self.root, "ref", rng.integers(2, size=(4, 18), dtype=np.uint8), np.arange(5) * 2
        )
        y = self.root / "sources"
        np.savetxt(y, np.full(9, 0.5))
        res = command("admix", "--clusters", ref, "--K", 3, "--supervised", y, success=False)
        self.assertIn("integer", res.stderr)
        np.savetxt(y, np.ones((9, 2)))
        res = command("admix", "--clusters", ref, "--K", 3, "--supervised", y, success=False)
        self.assertIn("integer", res.stderr)
        P = self.root / "P"
        np.savetxt(P, np.full((3, 8), 0.5))
        res = command("admix", "--clusters", ref, "--K", 3, "--projection", P, success=False)
        self.assertIn("doesn't match", res.stderr)
        keep = self.root / "keep"
        for text in ("s0\ns0\n", "s0\nabsent\n"):
            keep.write_text(text)
            res = command(
                "admix", "--clusters", ref, "--K", 3, "--keep", keep, "--random-init", success=False
            )
            self.assertIn("Keep file", res.stderr)
        files = self.root / "files"
        files.write_text(str(ref) + "\n")
        res = command("admix", "--clusters", ref, "--filelist", files, "--K", 3, success=False)
        self.assertIn("exactly one", res.stderr)
        res = command("admix", "--clusters", ref, "--K", 3, "--prefix", "../bad", success=False)
        self.assertIn("filename", res.stderr)

    def test_filelist_cannot_be_overwritten_by_statistical_commands(self):
        rng = np.random.default_rng(17)
        ref = writeClusters(
            self.root, "ref", rng.integers(2, size=(10, 16), dtype=np.uint8), np.arange(11) * 2
        )
        out = self.root / "result"
        files = Path(f"{out}.log")
        original = str(ref) + "\n"
        files.write_text(original)
        res = command("struct", "--filelist", files, "--pca", 2, "--out", out, success=False)
        self.assertIn("conflicts", res.stderr)
        self.assertEqual(files.read_text(), original)
        Path(f"{ref}.ref.json").unlink()
        command("struct", "--clusters", ref, "--pca", 2, "--out", out)
        command("struct", "--clusters", ref, "--pca", 2, "--out", out)

    def test_direct_lists_validate_counts_and_protect_every_input(self):
        Z = np.array([[0, 1, 1, 0], [1, 0, 0, 1]], np.uint8)
        c = np.array([0, 2, 4])
        a = writeClusters(self.root, "a", Z, c)
        b = writeClusters(self.root, "b", Z, c)
        np.savetxt(f"{a}.Q", [[0.8, 0.2], [0.2, 0.8]])
        np.savetxt(f"{a}.P", np.full((4, 2), 0.5))
        np.savetxt(f"{b}.P", np.full((4, 2), 0.5))
        for name in ("struct", "admix", "fatash", "eval"):
            res = command(name, "--clusters", "--out", self.root / "bad", success=False)
            self.assertIn("expected at least one argument", res.stderr)
        base = ["fatash", "--clusters", a, b, "--qfile", f"{a}.Q", "--fixed-model"]
        res = command(*base, "--pfile", f"{a}.P", success=False)
        self.assertIn("counts differ", res.stderr)
        res = command(*base, "--pfile", f"{a}.P", f"{b}.P", "--pfilelist", "unused", success=False)
        self.assertIn("exactly one", res.stderr)
        Path(f"{b}.Q").write_text("previous\n")
        ids = Path(f"{b}.ids").read_bytes()
        res = command(*base, "--pfile", f"{a}.P", f"{b}.P", "--out", b, success=False)
        self.assertIn("conflicts with input", res.stderr)
        self.assertEqual(Path(f"{b}.ids").read_bytes(), ids)
        self.assertEqual(Path(f"{b}.Q").read_text(), "previous\n")
