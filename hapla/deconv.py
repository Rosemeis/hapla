"""Expand local-ancestry paths into ancestry-specific cluster or SNP copies."""

__author__ = "Thomas Bøggild"

import os
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from hashlib import sha256
from pathlib import Path
from time import perf_counter

import numpy as np

from hapla.runtime import printDone, printHeader, printTiming, writeLog


### Check paired input lists and filtering choices before creating temporary outputs
def checkArgs(args):
    from hapla.formats import readPaths

    if (args.clusters is None) == (args.filelist is None):
        raise ValueError("Provide exactly one cluster input form")
    if args.path_filelist is None or args.K is None:
        raise ValueError("Provide --path-filelist and --K")
    if args.K < 1 or args.K > 255:
        raise ValueError("--K must lie between 1 and 255")
    if not 0 <= args.min_fraction <= 1:
        raise ValueError("--min-fraction must lie between zero and one")
    if args.min_call_support is not None and not 0 <= args.min_call_support <= 1:
        raise ValueError("--min-call-support must lie between zero and one")
    if args.min_tract_windows is not None and args.min_tract_windows < 1:
        raise ValueError("--min-tract-windows must be positive")
    paths = readPaths(args.filelist, args.clusters)
    lai = readPaths(args.path_filelist)
    if len(paths) != len(lai):
        raise ValueError("Cluster and path file lists must have the same number of rows")
    support = None if args.support_filelist is None else readPaths(args.support_filelist)
    if support is not None and args.min_call_support is None:
        raise ValueError("--support-filelist requires --min-call-support")
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


### Read one H x W path matrix and mask low-support calls and short retained tracts
def filterPath(path, support, N, W, args):
    arr = np.loadtxt(path, dtype=np.int16, ndmin=2)
    if arr.shape != (2 * N, W) or np.any((arr < -1) | (arr >= args.K)):
        raise ValueError(f"Invalid ancestry path dimensions or labels: {path}")
    if support is not None:
        prob = np.loadtxt(support, dtype=np.float64, ndmin=2)
        if prob.shape != arr.shape or not np.all(np.isfinite(prob)):
            raise ValueError(f"Invalid call-support matrix: {support}")
        arr[prob < args.min_call_support] = -1
    if args.homozygous_only:
        left, right = arr[::2], arr[1::2]
        keep = (left == right) & (left >= 0)
        left[~keep] = right[~keep] = -1
    if args.min_tract_windows is not None:
        for row in arr:
            ends = np.r_[np.flatnonzero(np.diff(row) != 0) + 1, W]
            start = 0
            for end in ends:
                if end - start < args.min_tract_windows:
                    row[start:end] = -1
                start = end
    return arr


### Give each input chromosome a stable suffix without exposing directory names
def suffixes(paths):
    seen, out = set(), []
    for i, path in enumerate(paths, 1):
        name = Path(path).name
        tail = name.rsplit(".", 1)[-1] if "." in name else str(i)
        tail = "".join(c if c.isalnum() or c in "_-" else "_" for c in tail)
        tail = tail if tail and tail not in seen else str(i)
        seen.add(tail)
        out.append(tail)
    return out


### Write an ancestry-expanded assignment bundle that remains valid Hapla input
def writeClusters(prefix, source, labels, path, selected, ids, rows, include_original):
    from hapla.formats import MAGIC
    from hapla.identity import writeIdentity

    N, W = len(ids), len(rows)
    copy_ids = [f"{ids[i]}_{k}" for i, k in selected]
    out_ids = ([*ids] if include_original else []) + copy_ids
    expanded = np.full((W, 2 * len(out_ids)), 255, np.uint8)
    offset = 0
    if include_original:
        expanded[:, : 2 * N] = labels
        offset = N
    for j, (i, k) in enumerate(selected):
        dst = 2 * (offset + j)
        keep0, keep1 = path[2 * i] == k, path[2 * i + 1] == k
        expanded[keep0, dst] = labels[keep0, 2 * i]
        expanded[keep1, dst + 1] = labels[keep1, 2 * i + 1]
    files = {
        ".bca": Path(f"{prefix}.bca"),
        ".ids": Path(f"{prefix}.ids"),
        ".win": Path(f"{prefix}.win"),
        ".ref.json": Path(f"{prefix}.ref.json"),
    }
    files[".bca"].write_bytes(MAGIC + expanded.tobytes())
    files[".ids"].write_text("".join(f"{s}\n" for s in out_ids))
    files[".win"].write_text(
        "#CHROM\tSTART\tEND\tLENGTH\tSIZE\tK\n"
        + "".join("\t".join(map(str, row)) + "\n" for row in rows)
    )
    key = sha256(Path(f"{source}.ref.json").read_bytes() + path.tobytes()).hexdigest()
    writeIdentity(files, key, W, int(sum(row[5] for row in rows)))


### Filter each chromosome once, retain sufficiently observed copies globally, then write outputs
def main(args):
    from hapla.formats import mapLabels, readMetadata, readWindows

    paths, lai, support, bcf = checkArgs(args)
    start = perf_counter()
    printHeader("deconv", args.threads, f"Ancestries: {args.K}")
    prefixes, ids, _, sizes = readMetadata(paths)
    N = len(ids)
    labels = []
    rows_all = []
    filtered = []
    weight = np.zeros((N, args.K), np.int64)
    total = 0
    with tempfile.TemporaryDirectory(
        prefix=".hapla-deconv-", dir=Path(args.out).absolute().parent
    ) as tmp:
        tmp = Path(tmp)
        for j, (prefix, path_file) in enumerate(zip(prefixes, lai, strict=True)):
            rows = readWindows(prefix)
            W = len(rows)
            current = filterPath(path_file, None if support is None else support[j], N, W, args)
            widths = np.asarray([row[4] for row in rows], np.int64)
            total += 2 * widths.sum()
            for k in range(args.K):
                seen = (current[::2] == k).astype(np.int64)
                seen += (current[1::2] == k).astype(np.int64)
                weight[:, k] += seen @ widths
            saved = tmp / f"path{j}.npy"
            np.save(saved, current)
            filtered.append(saved)
            rows_all.append(rows)
            labels.append(mapLabels(prefix, W, N))
        selected = [
            (i, k)
            for i in range(N)
            for k in range(args.K)
            if weight[i, k] / total >= args.min_fraction
        ]
        if not selected:
            raise ValueError("No ancestry copies satisfy --min-fraction")
        print(f"Original samples: {N:,}\nAncestry copies retained: {len(selected):,}", flush=True)
        selected_arr = np.asarray(selected, dtype=np.int32)
        copy_ids = [f"{ids[i]}_{k}" for i, k in selected]
        base = Path(args.out).absolute()
        base.parent.mkdir(parents=True, exist_ok=True)
        tags = suffixes(prefixes)
        out = {}
        if args.format in ("clusters", "both"):
            filelist = tmp / "clusters.filelist"
            with filelist.open("w") as handle:
                for j, (prefix, rows, tag) in enumerate(zip(prefixes, rows_all, tags, strict=True)):
                    target = tmp / f"result.{tag}"
                    writeClusters(
                        target,
                        prefix,
                        labels[j],
                        np.load(filtered[j]),
                        selected,
                        ids,
                        rows,
                        args.include_original,
                    )
                    handle.write(f"{base}.{tag}\n")
            out[".filelist"] = filelist
            for j, tag in enumerate(tags):
                for sfx in (".bca", ".ids", ".win", ".ref.json"):
                    out[f".{tag}{sfx}"] = tmp / f"result.{tag}{sfx}"
        if args.format in ("bcf", "both"):
            from hapla.vcf_cy import writeDeconvBcf

            targets = [tmp / f"result.{tag}.bcf" for tag in tags]
            workers = min(4, len(bcf), max(1, args.threads))
            io_threads = max(0, args.threads // workers - 1)
            print(f"Writing {len(bcf):,} BCF outputs with {workers} workers.", flush=True)
            with ThreadPoolExecutor(max_workers=workers) as pool:
                jobs = {
                    pool.submit(
                        writeDeconvBcf,
                        bcf[j],
                        targets[j],
                        np.load(filtered[j]),
                        selected_arr,
                        np.asarray([row[4] for row in rows_all[j]], dtype=np.int64),
                        copy_ids,
                        args.include_original,
                        io_threads,
                    ): j
                    for j in range(len(bcf))
                }
                for future in as_completed(jobs):
                    j = jobs[future]
                    printTiming(f"BCF chromosome {tags[j]}", future.result())
            bcfs = tmp / "bcfs"
            with bcfs.open("w") as handle:
                for j, tag in enumerate(tags):
                    handle.write(f"{base}.{tag}.bcf\n")
                    out[f".{tag}.bcf"] = targets[j]
            out[".bcfs"] = bcfs
        if args.save_filtered_paths:
            listed = tmp / "paths"
            with listed.open("w") as handle:
                for j, tag in enumerate(tags):
                    target = tmp / f"result.{tag}.path"
                    np.savetxt(target, np.load(filtered[j]), fmt="%d")
                    handle.write(f"{base}.{tag}.path\n")
                    out[f".{tag}.path"] = target
            out[".paths"] = listed
        for sfx, source in out.items():
            os.replace(source, f"{base}{sfx}")
    stats = dict(
        inputs=len(prefixes),
        samples=N,
        ancestry_copies=len(selected),
        include_original=args.include_original,
        format=args.format,
        min_fraction=args.min_fraction,
        min_call_support=args.min_call_support,
        min_tract_windows=args.min_tract_windows,
        homozygous_only=args.homozygous_only,
        saved_filtered_paths=args.save_filtered_paths,
    )
    elapsed = perf_counter() - start
    printTiming("Deconvolution", elapsed)
    writeLog(f"{args.out}.log", "deconv", args, stats)
    out[".log"] = Path(f"{args.out}.log")
    printDone(args.out, out, elapsed)
