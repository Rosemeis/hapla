"""Parallel scheduling must respect memory limits without starving available workers."""

import json
import os
import subprocess
import sys
import threading
import unittest
from contextlib import ExitStack, nullcontext, redirect_stderr
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from helpers import TemporaryTests, command, writeVcf

import hapla
from hapla.runtime import batches, batchSize, openReader, runBatches


class OutputTests(TemporaryTests):
    def test_header_notes_preserve_other_diagnostics_and_restore_stderr(self):
        warning = "[W::bcf_hdr_check_sanity] PP should be declared as "
        other = "[W::vcf_parse] Undefined contig\n[E::bcf_hdr_read] Invalid header"

        def reader(*args, **kwargs):
            os.write(2, f"{warning}Number=G\n{warning}Type=Integer\n{other}\n".encode())
            if fail:
                raise ValueError("Invalid header")
            return nullcontext(None)

        for fail in (False, True):
            text, fd = StringIO(), os.fstat(2)
            with ExitStack() as stack, redirect_stderr(text), patch("hapla.vcf_cy.Reader", reader):
                if fail:
                    with self.assertRaisesRegex(ValueError, "Invalid header"):
                        openReader(stack, "fixture", 0)
                else:
                    _, notes = openReader(stack, "fixture", 0)
                    self.assertEqual("\n".join(notes) + "\n", text.getvalue())
            self.assertEqual((os.fstat(2).st_dev, os.fstat(2).st_ino), (fd.st_dev, fd.st_ino))
            self.assertEqual(text.getvalue().count("Only GT is read."), 1)
            self.assertIn(other, text.getvalue())

    def test_unused_pp_does_not_change_gt_results_or_crowd_output(self):
        src, pp = self.root / "gt.vcf", self.root / "pp.vcf"
        writeVcf(src, [("1", 1, ("0|0", "0|1", "1|1"))])
        txt = src.read_text().replace(
            "#CHROM", '##FORMAT=<ID=PP,Number=1,Type=Float,Description="PP">\n#CHROM'
        )
        txt = txt.replace("\tGT\t0|0\t0|1\t1|1", "\tGT:PP\t0|0:0.1\t0|1:0.2\t1|1:0.3")
        pp.write_text(txt)
        payloads = []
        for path in (src, pp):
            out = self.root / path.stem
            res = command("cluster", "--vcf", path, "--size", 1, "--out", out)
            self.assertIn("\nhapla cluster\nSize: 1\nThreads: 1\n", res.stdout)
            payloads.append(Path(f"{out}.bca").read_bytes())
            self.assertEqual(res.stdout.count("Time elapsed:"), 1)
            self.assertNotIn("\r", res.stdout)
            self.assertNotIn("Missing assignments", res.stdout)
            self.assertNotIn(".bcm", res.stdout)
            log = Path(f"{out}.log").read_text()
            self.assertIn("Command: hapla cluster", log)
            self.assertNotIn('"statistics"', log)
            if res.stderr:
                self.assertIn("Only GT is read.", res.stderr)
                self.assertIn(res.stderr.strip(), log)
        self.assertEqual(*payloads)


class SchedulingTests(unittest.TestCase):
    def test_thread_budget_overrides_inherited_runtime_settings(self):
        code = """
import ctypes
import json
import os
from hapla.runtime import configureThreads
configureThreads(3, blas=1)
from hapla import fatash_cy
lib = ctypes.CDLL(fatash_cy.__file__)
print(json.dumps([
    lib.omp_get_max_threads(), lib.omp_get_thread_limit(),
    lib.omp_get_dynamic(), lib.omp_get_max_active_levels(),
    [os.environ[k] for k in ('OPENBLAS_NUM_THREADS', 'BLIS_NUM_THREADS',
     'GOTO_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS',
     'NUMEXPR_MAX_THREADS', 'VECLIB_MAXIMUM_THREADS')],
    'MKL_DOMAIN_NUM_THREADS' in os.environ,
]))
"""
        env = dict(
            os.environ,
            PYTHONPATH=str(Path(hapla.__file__).resolve().parent.parent),
            OMP_THREAD_LIMIT="1",
            OMP_DYNAMIC="TRUE",
            OMP_MAX_ACTIVE_LEVELS="8",
            MKL_DOMAIN_NUM_THREADS="MKL_DOMAIN_ALL=99",
            BLIS_NUM_THREADS="99",
            GOTO_NUM_THREADS="99",
        )
        res = subprocess.run(
            [sys.executable, "-c", code],
            env=env,
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
        self.assertEqual(json.loads(res.stdout), [3, 3, 0, 1, ["1"] * 7, False])

    def test_adaptive_batches_allow_all_workers_to_run(self):
        workers, window_bytes, queue_bytes = 4, 4096, 65536
        count = batchSize(window_bytes, n_req=8, b_in=32768, b_buf=queue_bytes, workers=workers)
        ready = threading.Barrier(workers, timeout=5)
        threads, output = set(), []
        lock = threading.Lock()

        def fit(batch):
            with lock:
                threads.add(threading.get_ident())
            ready.wait()
            return batch

        items = list(range(count * workers * 2))
        runBatches(
            batches(iter(items), count),
            fit,
            output.append,
            workers=workers,
            par=True,
            b_buf=queue_bytes,
            size_of=lambda batch: len(batch) * window_bytes,
        )
        self.assertEqual(len(threads), workers)
        self.assertEqual(output, items)

    def test_queue_budget_bounds_prefetch_and_preserves_order(self):
        produced, peak = 0, 0
        output = []

        def inputs():
            nonlocal produced, peak
            for value in range(100):
                produced += 1
                peak = max(peak, produced - len(output))
                yield [value]

        runBatches(
            inputs(),
            lambda batch: batch,
            output.append,
            workers=4,
            par=True,
            b_buf=8192,
            size_of=lambda batch: 4096,
        )
        # Two submitted batches and one producer batch may be retained.
        self.assertLessEqual(peak, 3)
        self.assertEqual(output, list(range(100)))


if __name__ == "__main__":
    unittest.main()
