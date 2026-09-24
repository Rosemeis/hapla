"""Stream phased GT through exact packed haplotype clustering on CPU workers."""

__author__ = "Jonas Meisner"

import math
from contextlib import ExitStack
from functools import partial
from hashlib import sha256
from pathlib import Path
from time import perf_counter

from hapla.runtime import (
    batches,
    batchSize,
    commitOutputs,
    openReader,
    printDone,
    printHeader,
    printMissing,
    runBatches,
    stageOutputs,
    threadPlan,
    writeLog,
)


### Validate clustering and input limits before allocating buffers
def checkArgs(args):
    if args.vcf is None or not Path(args.vcf).is_file():
        raise ValueError("Provide an existing phased VCF/BCF with --bcf or --vcf")
    if sum(value is not None for value in (args.size, args.length, args.windows)) != 1:
        raise ValueError("Select exactly one of --size, --length, or --windows")
    if args.size is not None and args.size < 1:
        raise ValueError("--size must be positive")
    if args.length is not None and args.length < 1:
        raise ValueError("--length must be positive")
    if args.step is not None and (args.size is None or not 1 <= args.step <= args.size):
        raise ValueError("--step requires --size and must lie between 1 and the window size")
    if not math.isfinite(args.lmbda) or not 0 < args.lmbda < 1:
        raise ValueError("--lmbda must lie strictly between 0 and 1")
    if not math.isfinite(args.min_freq) or not 0 < args.min_freq < 1:
        raise ValueError("--min-freq must lie strictly between 0 and 1")
    if args.min_mac is not None and args.min_mac < 1:
        raise ValueError("--min-mac must be positive")
    if not 1 <= args.max_clusters <= 255:
        raise ValueError(
            "--max-clusters must be between 1 and 255. Byte 255 is reserved for missing"
        )
    if args.max_iterations < 1:
        raise ValueError("--max-iterations must be positive")
    if args.buffer_mb < 1 or args.batch_windows < 1:
        raise ValueError("--buffer-mb and --batch-windows must be positive")


### Fit independent windows and discard scratch arrays before returning results
def fitBatch(batch, opt, medians, missing):
    from hapla import packed_cy

    out = []
    for meta, G, miss, _ in batch:
        try:
            if miss and missing == "error":
                raise ValueError("Missing GT encountered with --missing error")
            res = packed_cy.fitWindow(G, missing=miss, **opt)
        except (ValueError, RuntimeError) as err:
            _, chrom, beg, end, _ = meta
            raise type(err)(f"{chrom}:{beg}-{end}: {err}") from err
        h = sha256(res["medians"])
        h.update(res["counts"].astype("<u8", copy=False))
        h.update(res["sizes"].astype("<u8", copy=False))
        res["identity"] = h.digest()
        if medians:
            res["likelihoods"] = packed_cy.likelihoods(res["medians"], res["counts"], res["sizes"])
        else:
            del res["medians"]
        del res["counts"], res["sizes"]
        res["window"] = meta
        out.append(res)
    return out


### Cluster bounded batches and publish the complete output set
def main(args):
    checkArgs(args)
    nt, nio = threadPlan(args.threads, args.io_threads)
    from hapla.formats import FORMAT_VERSION, openOutputs, outputSuffixes, writeWindow
    from hapla.identity import fileHash, writeIdentity
    from hapla.windows import (
        createBuffer,
        fixedWindows,
        physicalWindows,
        predefinedWindows,
        readStarts,
    )

    # Prepare options and output statistics
    start = perf_counter()
    mem = args.buffer_mb * 1024**2
    idx = readStarts(args.windows) if args.windows is not None else None
    opt = dict(
        alpha=args.lmbda,
        min_freq=args.min_freq,
        min_mac=args.min_mac,
        K_max=args.max_clusters,
        n_iter=args.max_iterations,
    )
    stats = dict(
        windows=0,
        clusters=0,
        missing_assignments=0,
        all_missing_windows=0,
        capped_windows=0,
        growth_passes=0,
        pruning_passes=0,
        distance_pairs=0,
    )
    size = (
        f"{args.size:,}"
        if args.size is not None
        else (f"{args.length:,} bp" if args.length is not None else "predefined")
    )
    printHeader("cluster", args.threads, f"Size: {size}")
    with ExitStack() as stack:
        src, stats["diagnostics"] = openReader(stack, args.vcf, nio)
        print(f"Samples: {len(src.samples):,}\nClustering windows.", flush=True)
        out = stageOutputs(
            stack,
            args.out,
            outputSuffixes(args.medians, args.plink),
            inputs=(args.vcf, args.windows),
        )
        buf = createBuffer(src.readInto, src.samples, src.contigs, mem // 4)

        # Select window boundaries and leave room for all workers
        if args.size is not None:
            windows = fixedWindows(buf, args.size, args.step, args.tail)
            n = batchSize(
                args.size * 2 * len(src.samples), args.batch_windows, mem // 4, mem // 2, nt
            )
        elif args.length is not None:
            windows, n = physicalWindows(buf, args.length), 1
        else:
            windows, n = predefinedWindows(buf, idx), 1
        with ExitStack() as io:
            files = openOutputs(io, out, src.samples, args.duplicate_fid)
            site_id, fit_key = sha256(), sha256()

            def sites(eof):
                site_id.update(src.sites)
                if args.medians:
                    files[".sites"].write(src.sites)

            buf["sites"] = sites

            # Write in input order while other workers continue fitting
            def write(res):
                fit_key.update(int(res["window"][0]).to_bytes(8, "little"))
                fit_key.update(res["identity"])
                info = res["stats"]
                K = info["K"]
                stats["windows"] += 1
                stats["clusters"] += K
                stats["missing_assignments"] += info["missing"]
                stats["all_missing_windows"] += K == 0
                stats["capped_windows"] += info["capped"]
                stats["growth_passes"] += info["growth_passes"]
                stats["pruning_passes"] += info["prune_passes"]
                stats["distance_pairs"] += info["distance_pairs"]
                writeWindow(files, res["window"], res["labels"], K, stats["windows"])
                if args.medians:
                    files[".bcm"].write(res["medians"])
                    files[".blk"].write(res["likelihoods"])
                    files[".wix"].write(f"{res['window'][0]}\n")

            runBatches(
                batches(windows, n),
                partial(fitBatch, opt=opt, medians=args.medians, missing=args.missing),
                write,
                workers=nt,
                par=args.threads > 1,
                b_buf=mem // 2,
                size_of=lambda batch: sum(G.nbytes for _, G, _, _ in batch),
            )
        if not stats["windows"]:
            raise ValueError("No windows were produced. Check the input and window/tail settings")
        key = sha256(site_id.digest() + fit_key.digest() + bytes.fromhex(fileHash(out[".win"])))
        writeIdentity(out, key.hexdigest(), stats["windows"], stats["clusters"])

        # Flush staged files before replacing previous outputs
        stats.update(
            variants=buf["variants"],
            haplotypes=2 * len(src.samples),
            read_seconds=buf["time"],
            elapsed_seconds=perf_counter() - start,
            input_buffer_mib=mem / 1024**2,
            workers=nt,
            batch_windows=n,
            io_threads=nio,
            htslib_version=src.htslib_version,
            htslib_features=src.htslib_features,
            format_version=FORMAT_VERSION,
        )
        writeLog(out[".log"], "cluster", args, stats)
        commitOutputs(args.out, out)
    print(
        f"Clustered {stats['variants']:,} variants into:\n"
        f"- {stats['windows']:,} windows\n"
        f"- {stats['clusters']:,} clusters",
        flush=True,
    )
    printMissing(stats["missing_assignments"], stats["windows"] * stats["haplotypes"])
    if stats["capped_windows"]:
        print(
            f"Cluster cap reached during growth: {stats['capped_windows']:,} windows.", flush=True
        )
    printDone(args.out, out, stats["elapsed_seconds"])
    return stats
