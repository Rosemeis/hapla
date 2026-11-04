"""Expand local ancestry paths into ancestry-specific cluster or SNP copies."""

__author__ = "Thomas Bøggild"

import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
from pathlib import Path
from time import perf_counter

import numpy as np

from hapla.runtime import commitOutputs, printDone, printHeader, printTiming, stageOutputs, writeLog


### Check paired input lists and filtering choices
def checkArgs(args):
    from hapla.formats import readPaths

    if (args.clusters is None) == (args.filelist is None):
        raise ValueError("Provide exactly one cluster input form")
    if args.path_filelist is None or args.K is None:
        raise ValueError("Provide --path-filelist and --K")
    if not 1 <= args.K <= 255:
        raise ValueError("--K must lie between 1 and 255")
    if args.threads < 1:
        raise ValueError("--threads must be positive")
    if not 0 <= args.min_fraction <= 1:
        raise ValueError("--min-fraction must lie between zero and one")
    if args.min_call_support is not None and not 0 <= args.min_call_support <= 1:
        raise ValueError("--min-call-support must lie between zero and one")
    if args.min_tract_windows is not None and args.min_tract_windows < 1:
        raise ValueError("--min-tract-windows must be positive")
    paths, lai = readPaths(args.filelist, args.clusters), readPaths(args.path_filelist)
    if len(paths) != len(lai):
        raise ValueError("Cluster and path file lists must have the same number of rows")
    support = None if args.support_filelist is None else readPaths(args.support_filelist)
    if (support is None) != (args.min_call_support is None):
        raise ValueError("Provide both --support-filelist and --min-call-support")
    if support is not None and len(paths) != len(support):
        raise ValueError("Cluster and support file lists must have the same number of rows")
    bcf = None if args.bcf_filelist is None else readPaths(args.bcf_filelist)
    if args.format in ("bcf", "both") and bcf is None:
        raise ValueError("BCF output requires --bcf-filelist")
    if bcf is not None and len(paths) != len(bcf):
        raise ValueError("Cluster and genotype file lists must have the same number of rows")
    if not args.out or Path(args.out).name in ("", ".", ".."):
        raise ValueError("Output prefix must include a filename")
    return paths, lai, support, bcf


### Read exact matrix dimensions through bounded text blocks
def readRows(path, H, W, dtype):
    step = max(1, 8 * 1024**2 // (8 * W))
    with open(path) as src:
        for beg in range(0, H, step):
            end = min(H, beg + step)
            block = np.loadtxt(src, dtype=dtype, ndmin=2, max_rows=end - beg)
            if block.shape != (end - beg, W):
                raise ValueError(f"Invalid matrix dimensions: {path}")
            yield slice(beg, end), block
        if any(line.strip() and not line.lstrip().startswith("#") for line in src):
            raise ValueError(f"Invalid matrix dimensions: {path}")


### Mask unsupported calls and short tracts within each chromosome
def filterPath(path, support, N, W, args, breaks=()):
    arr = np.empty((2 * N, W), np.int16)
    for rows, block in readRows(path, 2 * N, W, np.int64):
        if np.any((block < -1) | (block >= args.K)):
            raise ValueError(f"Invalid ancestry path labels: {path}")
        arr[rows] = block
    if support is not None:
        for rows, prob in readRows(support, 2 * N, W, float):
            if not np.all(np.isfinite(prob)) or np.any((prob < 0) | (prob > 1)):
                raise ValueError(f"Invalid call-support matrix: {support}")
            arr[rows][prob < args.min_call_support] = -1
    if args.homozygous_only:
        left, right = arr[::2], arr[1::2]
        keep = (left == right) & (left >= 0)
        left[~keep] = right[~keep] = -1
    if args.min_tract_windows is not None:
        for row in arr:
            ends = np.unique(np.r_[np.flatnonzero(np.diff(row)) + 1, breaks, W]).astype(np.intp)
            beg = 0
            for end in ends:
                if end - beg < args.min_tract_windows:
                    row[beg:end] = -1
                beg = end
    return arr


### Give each chromosome a unique filename suffix
def suffixes(paths):
    seen, out = set(), []
    for i, path in enumerate(paths, 1):
        name = Path(path).name
        tag = name.rsplit(".", 1)[-1] if "." in name else str(i)
        tag = "".join(c if c.isalnum() or c in "_-" else "_" for c in tag) or str(i)
        while tag in seen:
            tag = f"{tag}_{i}"
        seen.add(tag)
        out.append(tag)
    return out


### Stream expanded assignments through one bounded window buffer
def writeClusters(prefix, source, labels, path, selected, ids, rows, include_original):
    from hapla.formats import MAGIC
    from hapla.identity import readIdentity, writeIdentity
    from hapla.vcf_cy import expandDeconv

    W = len(rows)
    files = {s: Path(f"{prefix}{s}") for s in (".bca", ".ids", ".win", ".ref")}
    ref = readIdentity(source, (".bca", ".ids", ".win"))
    chunk = min(W, max(1, 8 * 1024**2 // (2 * len(ids))))
    buf = np.empty((chunk, 2 * len(ids)), np.uint8)
    with files[".bca"].open("wb") as dst:
        dst.write(MAGIC)
        for beg in range(0, W, chunk):
            end = min(W, beg + chunk)
            out = buf[: end - beg]
            expandDeconv(labels[beg:end], path, selected, out, beg, include_original)
            dst.write(memoryview(out))
    files[".ids"].write_text("".join(f"{s}\n" for s in ids))
    shutil.copyfile(f"{source}.win", files[".win"])
    writeIdentity(files, ref["reference"], W, int(sum(row[5] for row in rows)))


### Time one native BCF writer within the shared worker pool
def writeBcf(*args):
    from hapla.vcf_cy import writeDeconvBcf

    start = perf_counter()
    writeDeconvBcf(*args)
    return perf_counter() - start


### Select ancestry copies globally and publish all chromosome outputs together
def main(args):
    from hapla.formats import mapLabels, readMetadata, writeMatrix
    from hapla.vcf_cy import deconvWeights

    paths, lai, support, bcf = checkArgs(args)
    start = perf_counter()
    printHeader("deconv", args)
    meta, filtered = [], []
    paths, ids, _, _ = readMetadata(paths, meta=meta)
    N = len(ids)
    weight = np.zeros((N, args.K), np.int64)
    total = 0
    base = Path(args.out).absolute()
    base.parent.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        tmp = Path(
            stack.enter_context(
                tempfile.TemporaryDirectory(prefix=".hapla-deconv-", dir=base.parent)
            )
        )
        for j, (rows, path_file) in enumerate(zip(meta, lai)):
            W = len(rows)
            breaks = [w for w in range(1, W) if rows[w][0] != rows[w - 1][0]]
            current = filterPath(
                path_file, None if support is None else support[j], N, W, args, breaks
            )
            widths = np.asarray([row[4] for row in rows], np.int64)
            total += 2 * widths.sum()
            deconvWeights(current, widths, weight)
            saved = tmp / f"path{j}.npy"
            np.save(saved, current)
            filtered.append(saved)
        selected = np.column_stack(np.nonzero(weight / total >= args.min_fraction)).astype(np.int32)
        del weight, current
        if not len(selected):
            raise ValueError("No ancestry copies satisfy --min-fraction")
        copy_ids = [f"{ids[i]}_{k}" for i, k in selected]
        if len(set(copy_ids)) != len(copy_ids) or (
            args.include_original and set(ids).intersection(copy_ids)
        ):
            raise ValueError("Ancestry copy IDs conflict with original sample IDs")
        print(f"Original samples: {N:,}\nAncestry copies retained: {len(selected):,}", flush=True)
        tags = suffixes(paths)
        sfxs, stale = [".log"], (".filelist", ".bcfs", ".paths")
        if args.format in ("clusters", "both"):
            sfxs += [".filelist"] + [
                f".{tag}{s}" for tag in tags for s in (".bca", ".ids", ".win", ".ref")
            ]
        if args.format in ("bcf", "both"):
            sfxs += [".bcfs"] + [f".{tag}.bcf" for tag in tags]
        if args.save_filtered_paths:
            sfxs += [".paths"] + [f".{tag}.path" for tag in tags]
        inputs = [f"{p}{s}" for p in paths for s in (".bca", ".ids", ".win", ".ref")]
        inputs += [
            *lai,
            *(support or []),
            *(bcf or []),
            args.filelist,
            args.path_filelist,
            args.support_filelist,
            args.bcf_filelist,
        ]
        out = stageOutputs(stack, base, sfxs, inputs, stale=stale)
        if args.format in ("clusters", "both"):
            out_ids = [*ids, *copy_ids] if args.include_original else copy_ids
            for j, (prefix, rows, tag) in enumerate(zip(paths, meta, tags)):
                writeClusters(
                    out[f".{tag}.bca"].with_suffix(""),
                    prefix,
                    mapLabels(prefix, len(rows), N),
                    np.load(filtered[j], mmap_mode="r"),
                    selected,
                    out_ids,
                    rows,
                    args.include_original,
                )
            del out_ids
            out[".filelist"].write_text("".join(f"{base}.{tag}\n" for tag in tags))
        if args.format in ("bcf", "both"):
            workers = min(4, len(bcf), args.threads)
            nio = max(0, (args.threads // workers - 3) // 2)
            print(f"Writing {len(bcf):,} BCF outputs with {workers} workers.", flush=True)
            with ThreadPoolExecutor(max_workers=workers) as pool:
                jobs = {
                    pool.submit(
                        writeBcf,
                        bcf[j],
                        out[f".{tag}.bcf"],
                        np.load(filtered[j], mmap_mode="r"),
                        selected,
                        np.asarray([row[4] for row in meta[j]], np.int64),
                        copy_ids,
                        args.include_original,
                        nio,
                        ids,
                        meta[j],
                    ): tag
                    for j, tag in enumerate(tags)
                }
                for future in as_completed(jobs):
                    printTiming(f"BCF chromosome {jobs[future]}", future.result())
            out[".bcfs"].write_text("".join(f"{base}.{tag}.bcf\n" for tag in tags))
        if args.save_filtered_paths:
            for j, tag in enumerate(tags):
                writeMatrix(out[f".{tag}.path"], np.load(filtered[j], mmap_mode="r"), fmt="%d")
            out[".paths"].write_text("".join(f"{base}.{tag}.path\n" for tag in tags))
        stats = dict(
            inputs=len(paths),
            samples=N,
            ancestry_copies=len(selected),
            include_original=args.include_original,
            format=args.format,
            min_fraction=args.min_fraction,
            min_call_support=args.min_call_support,
            min_tract_windows=args.min_tract_windows,
            homozygous_only=args.homozygous_only,
            saved_filtered_paths=args.save_filtered_paths,
            elapsed_seconds=perf_counter() - start,
        )
        writeLog(out, "deconv", args, stats, base)
        commitOutputs(base, out, stale=stale)
    printTiming("Deconvolution", stats["elapsed_seconds"])
    printDone(args.out, out, stats["elapsed_seconds"])
