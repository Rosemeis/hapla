"""Expand local-ancestry paths into ancestry-specific cluster or SNP copies."""

__author__ = "Jonas Meisner"

import os
import tempfile
from contextlib import ExitStack
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
    if args.format in ("vcf", "both") and bcf is None:
        raise ValueError("VCF output requires --bcf-filelist")
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


### Render one VCF GT field, retaining only alleles assigned to the copy's ancestry
def genotype(a, b):
    left = "." if a == 255 else str(int(a))
    right = "." if b == 255 else str(int(b))
    return f"{left}|{right}"


### Stream a genotype input and emit original and/or ancestry-masked diploid copies
def writeVcf(out, bcf, rows, path, selected, ids, include_original):
    from hapla.vcf_cy import Reader

    N, W = len(ids), len(rows)
    with Reader(bcf, phased=True, save=True) as src:
        if src.samples != list(ids):
            raise ValueError(f"Genotype samples differ from cluster samples: {bcf}")
        names = ([*ids] if include_original else []) + [f"{ids[i]}_{k}" for i, k in selected]
        with open(out, "w", buffering=1024**2) as dst:
            dst.write("##fileformat=VCFv4.3\n")
            dst.write("##source=hapla deconv\n")
            for name in src.contigs:
                dst.write(f"##contig=<ID={name}>\n")
            dst.write('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n')
            dst.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t" + "\t".join(names) + "\n")
            G = np.empty((4096, 2 * N), np.uint8)
            pos = np.empty(4096, np.int64)
            contigs = np.empty(4096, np.int32)
            missing = np.empty(4096, np.uint8)
            w = seen = 0
            while (n := src.read_into(G, pos, contigs, missing)):
                site_rows = src.sites.decode().splitlines()
                if len(site_rows) != n:
                    raise RuntimeError("Genotype reader did not retain complete variant identities")
                for r, site in enumerate(site_rows):
                    if w >= W:
                        raise ValueError(f"Genotype input has more variants than cluster windows: {bcf}")
                    chrom, bp, ref, alt = site.split("\t")
                    values = [genotype(G[r, 2 * i], G[r, 2 * i + 1]) for i in range(N)] if include_original else []
                    for i, k in selected:
                        a = G[r, 2 * i] if path[2 * i, w] == k else 255
                        b = G[r, 2 * i + 1] if path[2 * i + 1, w] == k else 255
                        values.append(genotype(a, b))
                    dst.write(f"{chrom}\t{bp}\t.\t{ref}\t{alt}\t.\tPASS\t.\tGT\t" + "\t".join(values) + "\n")
                    seen += 1
                    if seen == rows[w][4]:
                        seen, w = 0, w + 1
    if w != W or seen:
        raise ValueError(f"Genotype input has fewer variants than cluster windows: {bcf}")


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
    with tempfile.TemporaryDirectory(prefix=".hapla-deconv-", dir=Path(args.out).absolute().parent) as tmp:
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
        selected = [(i, k) for i in range(N) for k in range(args.K) if weight[i, k] / total >= args.min_fraction]
        if not selected:
            raise ValueError("No ancestry copies satisfy --min-fraction")
        print(f"Original samples: {N:,}\nAncestry copies retained: {len(selected):,}", flush=True)
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
        if args.format in ("vcf", "both"):
            vcfs = tmp / "vcfs"
            with vcfs.open("w") as handle:
                for j, tag in enumerate(tags):
                    target = tmp / f"result.{tag}.vcf"
                    writeVcf(target, bcf[j], rows_all[j], np.load(filtered[j]), selected, ids, args.include_original)
                    handle.write(f"{base}.{tag}.vcf\n")
                    out[f".{tag}.vcf"] = target
            out[".vcfs"] = vcfs
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
