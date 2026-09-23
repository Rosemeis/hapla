"""Thread budgets, bounded scheduling, output publication, and run summaries."""

__author__ = "Jonas Meisner"

import os
import shlex
import sys
import tempfile
import textwrap
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from numbers import Integral, Real
from pathlib import Path

from hapla import __version__

_OPTIONAL = (".bcm", ".blk", ".wix", ".sites", ".bed", ".bim", ".fam")


### Set numerical thread pools before importing NumPy or native kernels
def configureThreads(n, blas=None):
    os.environ["OMP_NUM_THREADS"] = os.environ["OMP_THREAD_LIMIT"] = str(n)
    nt = str(n if blas is None else blas)
    for key in (
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "GOTO_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "NUMEXPR_MAX_THREADS",
    ):
        os.environ[key] = nt
    os.environ["OMP_DYNAMIC"] = os.environ["MKL_DYNAMIC"] = "FALSE"
    os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"
    os.environ.pop("MKL_DOMAIN_NUM_THREADS", None)


### Divide the CPU budget between input, decompression, and window workers
def threadPlan(threads, nio=None):
    if threads < 1:
        raise ValueError("--threads must be positive")
    if nio is None:
        nio = 2 if threads >= 8 else int(threads >= 6)
    if not 0 <= nio <= max(0, threads - 3):
        raise ValueError(
            "--io-threads must be nonnegative and leave threads for reading, window processing, "
            "and HTSlib's I/O coordinator (at least 4 total for threaded decompression)"
        )

    # The caller reads input. Workers run GIL-free serial kernels. No nested OpenMP.
    # BGZF adds a coordinator thread as well as the requested decompression pool.
    workers = max(1, threads - nio - 1 - bool(nio))
    configureThreads(1)
    return workers, nio


### Stage all requested files beside their final destinations
def stageOutputs(stack, pfx, sfxs, inputs=(), *, stale=_OPTIONAL):
    if os.fspath(pfx).split(os.sep)[-1] in ("", ".", ".."):
        raise ValueError("Output prefix must include a filename")
    pfx = Path(pfx).absolute()
    pfx.parent.mkdir(parents=True, exist_ok=True)
    sources = [Path(p).resolve() for p in inputs if p is not None]
    for sfx in dict.fromkeys((*sfxs, *stale)):
        dst = Path(f"{pfx}{sfx}")
        if dst.is_symlink() or (dst.exists() and not dst.is_file()):
            raise ValueError(f"Output must be a regular file: {dst}")
        for src in sources:
            if dst.resolve() == src or (
                dst.exists() and src.exists() and os.path.samefile(dst, src)
            ):
                raise ValueError(f"Output conflicts with input: {dst}")
    tmp = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix=".hapla-", dir=pfx.parent)))
    return {sfx: tmp / f"output{sfx}" for sfx in sfxs}


### Publish outputs and remove stale optional files with rollback on failure
def commitOutputs(pfx, out, *, stale=_OPTIONAL):
    pfx = Path(pfx).absolute()
    tmp = next(iter(out.values())).parent
    moved, saved = [], []
    try:
        for sfx, pth in out.items():
            if not pth.is_file():
                raise RuntimeError(f"Incomplete output set: {sfx}")
        for sfx in dict.fromkeys((*out, *stale)):
            dst = Path(f"{pfx}{sfx}")
            if dst.exists():
                old = tmp / f"previous{sfx}"
                os.replace(dst, old)
                saved.append((old, dst))
        for sfx, pth in out.items():
            dst = Path(f"{pfx}{sfx}")
            os.replace(pth, dst)
            moved.append(dst)
    except BaseException:
        for dst in reversed(moved):
            dst.unlink()
        for old, dst in reversed(saved):
            os.replace(old, dst)
        raise


##### Console and log output


### Print one compact command header before native diagnostics
def printHeader(cmd, threads, detail=None):
    rows = [f"hapla v{__version__}", f"hapla {cmd}"]
    if detail is not None:
        rows.append(detail)
    rows.append(f"Threads: {threads}")
    bar = "-" * max(64, *map(len, rows))
    print("\n".join([bar, *rows, bar, ""]), flush=True)


### Keep completed stage timings at a shared column
def printTiming(name, sec):
    print(f"{name:<48} ({sec:,.1f}s)", flush=True)


### Show missingness only when assignments are absent
def printMissing(n, total):
    if n:
        print(f"Missing assignments: {n:,} ({100 * n / total:.2f}%)", flush=True)


### List the actual published files once without repeating their prefix
def printDone(pfx, out, sec):
    print(f"\nTime elapsed: {sec:,.1f}s\nOutput prefix: {pfx}", flush=True)
    files = ", ".join([s for s in out if s != ".log"] + [".log"])
    print(textwrap.fill(f"Files: {files}", width=100, subsequent_indent="       "), flush=True)


### Format plain log values without JSON punctuation or excessive precision
def logValue(val):
    if hasattr(val, "item"):
        val = val.item()
    if isinstance(val, bool):
        return "yes" if val else "no"
    if isinstance(val, Integral):
        return f"{val:,}"
    if isinstance(val, Real):
        return f"{val:.12g}"
    if isinstance(val, (list, tuple)):
        return ", ".join(map(logValue, val))
    return str(val)


### Write readable statistics and compact iteration tables
def logRows(data, pad="  "):
    skip = {
        "growth_passes",
        "pruning_passes",
        "distance_pairs",
        "htslib_features",
        "format_version",
    }
    empty = {
        "missing_assignments",
        "all_missing_windows",
        "capped_windows",
        "empty_windows",
        "unobserved_samples",
        "unphased_sample_windows",
        "excluded_windows",
        "phase_corrections",
        "recoveries",
        "stalled",
    }
    words = {
        "htslib": "HTSlib",
        "io": "I/O",
        "mib": "MiB",
        "em": "EM",
        "p": "P",
        "q": "Q",
        "pca": "PCA",
        "grm": "GRM",
        "loglike": "log-like",
    }
    for key, val in data.items():
        if key in skip or val is None or (isinstance(val, list) and not val):
            continue
        if key in empty and not val:
            continue
        name = " ".join(words.get(s, s) for s in key.split("_"))
        name = name[0].upper() + name[1:]
        if isinstance(val, dict):
            yield f"\n{pad}{name}:"
            yield from logRows(val, pad + "  ")
        elif key == "history":
            cols = [k for k in val[0] if any(row[k] is not None for row in val)]
            if "batches" in cols and all(r["batches"] == val[0]["batches"] for r in val):
                cols.remove("batches")
            ll = "log_likelihood" if "log_likelihood" in cols else "loglike"
            if "objective" in cols and all(r["objective"] == r[ll] for r in val):
                cols.remove("objective")
            labels = {
                "iteration": "Iteration",
                "loglike": "Log-like",
                "log_likelihood": "Log-like",
                "objective": "Objective",
                "improvement": "Gain/obs",
                "batches": "Batches",
            }
            rows = [[labels[k] for k in cols]] + [
                ["-" if r[k] is None else logValue(r[k]) for k in cols] for r in val
            ]
            widths = [max(len(row[i]) for row in rows) for i in range(len(cols))]
            yield f"\n{pad}History:"
            for row in rows:
                yield pad + "  ".join(s.rjust(w) for s, w in zip(row, widths))
            yield ""
        elif key == "files":
            for i, row in enumerate(val, 1):
                yield f"\n{pad}File {i}:"
                yield from logRows(row, pad + "  ")
        elif key == "diagnostics":
            yield f"\n{pad}Input notes:"
            yield from (pad + "  " + line for line in val)
            yield ""
        else:
            text = f"{val:,.3f}" if key.endswith("seconds") else logValue(val)
            if key == "stop":
                text = text.replace("_", " ")
            yield f"{pad}{name + ':':<30} {text}"


### Record the command and results without duplicating all default options
def writeLog(pth, cmd, args, stats):
    rows = [
        f"hapla v{__version__} | {cmd}",
        f"Date: {datetime.now().astimezone().isoformat(timespec='seconds')}",
    ]
    rows.append(f"Directory: {os.getcwd()}")
    if hasattr(args, "_cmd"):
        rows.append(f"Command: {shlex.join(args._cmd)}")
    else:
        rows.append("\nOptions:")
        rows.extend(
            logRows({k: v for k, v in vars(args).items() if v is not None and v is not False})
        )
    rows.append("\nResults:")
    rows.extend(logRows(stats))
    Path(pth).write_text("\n".join(rows) + "\n")


### Consolidate only the two known PP notices while preserving other diagnostics
def headerNotes(text):
    pp = {
        "[W::bcf_hdr_check_sanity] PP should be declared as Number=G",
        "[W::bcf_hdr_check_sanity] PP should be declared as Type=Integer",
    }
    notes = []
    for line in text.splitlines():
        if line in pp:
            line = "Note: Nonstandard FORMAT/PP header ignored. Only GT is read."
            if line in notes:
                continue
        notes.append(line)
    return notes


### Capture startup diagnostics before launching any window workers
def openReader(stack, path, threads, phased=True):
    from hapla.vcf_cy import Reader

    # HTSlib has no per-reader log callback. Restore stderr even on header failure.
    with tempfile.TemporaryFile() as tmp:
        sys.stdout.flush()
        sys.stderr.flush()
        fd = os.dup(2)
        try:
            os.dup2(tmp.fileno(), 2)
            src = stack.enter_context(Reader(path, threads, phased=phased, save=True))
        finally:
            os.dup2(fd, 2)
            os.close(fd)
            tmp.seek(0)
            notes = headerNotes(tmp.read().decode("utf-8", errors="replace"))
            for line in notes:
                print(line, file=sys.stderr, flush=True)
    return src, notes


##### Bounded scheduling


### Collect a bounded number of windows per task
def batches(items, count):
    batch = []
    for item in items:
        batch.append(item)
        if len(batch) == count:
            yield batch
            batch = []
    if batch:
        yield batch


### Leave room for two tasks per worker within the input budget
def batchSize(b_win, n_req, b_in, b_buf, workers):
    return max(1, min(n_req, b_in // b_win, b_buf // (2 * workers * b_win)))


### Bound submitted inputs and write results in their original order
def runBatches(batches, fit, write, *, workers, par, b_buf, size_of):
    if not par:
        for batch in batches:
            for result in fit(batch):
                write(result)
        return
    pending = deque()
    queued = 0
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="hapla") as pool:
        try:
            for batch in batches:
                size = size_of(batch)
                if size > b_buf:
                    raise ValueError("A task exceeds the input budget. Increase --buffer-mb")
                while pending and (len(pending) >= 2 * workers or queued + size > b_buf):
                    future, used = pending.popleft()
                    for result in future.result():
                        write(result)
                    queued -= used
                pending.append((pool.submit(fit, batch), size))
                queued += size
                del batch
            while pending:
                future, _ = pending.popleft()
                for result in future.result():
                    write(result)
        except BaseException:
            for future, _ in pending:
                future.cancel()
            raise
