"""Stream phased GT through exact packed haplotype clustering on CPU workers."""

__author__ = "Jonas Meisner"

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
    if (
        sum(value is not None for value in (args.size, args.length, args.windows)) + args.adaptive
        != 1
    ):
        raise ValueError("Select exactly one of --size, --length, --windows, or --adaptive")
    if args.size is not None and args.size < 1:
        raise ValueError("--size must be positive")
    if args.length is not None and args.length < 1:
        raise ValueError("--length must be positive")
    if args.step is not None and (args.size is None or not 1 <= args.step <= args.size):
        raise ValueError("--step requires --size and must lie between 1 and the window size")
    if not 0 <= args.lmbda < 1:
        raise ValueError("--lmbda must be in [0, 1)")
    if not 0 <= args.min_freq < 1:
        raise ValueError("--min-freq must be in [0, 1)")
    if args.min_mac < 1:
        raise ValueError("--min-mac must be positive")
    if not 1 <= args.max_clusters <= 255:
        raise ValueError(
            "--max-clusters must be between 1 and 255. Byte 255 is reserved for missing"
        )
    if args.max_iterations < 1:
        raise ValueError("--max-iterations must be positive")
    if args.buffer_mb < 1 or args.batch_windows < 1:
        raise ValueError("--buffer-mb and --batch-windows must be positive")
    if args.adaptive:
        args.min_size = 8 if args.min_size is None else args.min_size
        args.max_size = 64 if args.max_size is None else args.max_size
        args.max_length = 100000 if args.max_length is None else args.max_length
        if not 1 <= args.min_size <= args.max_size or args.max_length < 1:
            raise ValueError(
                "Adaptive bounds require 1 <= --min-size <= --max-size and positive --max-length"
            )
        if args.tail != "include":
            raise ValueError("--adaptive includes all variants and cannot use --tail drop")
        if args.map is not None:
            args.max_cm = 0.1 if args.max_cm is None else args.max_cm
            from math import isfinite

            if not isfinite(args.max_cm) or args.max_cm <= 0:
                raise ValueError("--max-cm must be finite and positive")
        elif args.max_cm is not None:
            raise ValueError("--max-cm requires --map")
    elif any(
        value is not None
        for value in (args.min_size, args.max_size, args.max_length, args.map, args.max_cm)
    ):
        raise ValueError("Adaptive size, span, and map options require --adaptive")


### Choose fitted medians or smaller pieces by complete-data coding cost
def adaptiveFit(meta, G, missing, opt, minimum, pos):
    from hapla import packed_cy

    fits = 0
    packed = packed_cy.packWindow(G, missing)

    def fit(beg, end):
        nonlocal fits
        B = end - beg
        x = G[beg:end]
        obs = packed_cy.observedWindow(packed[1], beg, end) if missing else x.shape[1]
        res = None
        cost = lost = float("inf")
        if not 0 < obs < opt["min_mac"]:
            res = packed_cy.fitWindow(x, missing=obs < x.shape[1], packed=packed, offset=beg, **opt)
            fits += 1
            cost, lost = packed_cy.windowCost(
                x, res["labels"], res["medians"], res["counts"], res["sizes"]
            )
            res["window"] = (meta[0] + beg, meta[1], int(pos[beg]), int(pos[end - 1]), B)

            # A single exact pattern cannot benefit from another dictionary.
            if obs == x.shape[1] and res["stats"]["exact"] and res["stats"]["K"] == 1:
                return cost, lost, [res]
        if B >= 2 * minimum:
            mid = beg + B // 2
            lc, ll, left = fit(beg, mid)
            rc, rl, right = fit(mid, end)
            if left and right and (res is None or (cost, lost) > (lc + rc, ll + rl)):
                return lc + rc, ll + rl, left + right
        return cost, lost, [res] if res is not None else []

    cost, lost, out = fit(0, len(G))
    if not out:
        raise ValueError("Minimum cluster count exceeds the observed haplotypes in this window")
    return cost, lost, out, fits


### Fit independent windows and discard scratch arrays before returning results
def fitBatch(batch, opt, medians, missing, minimum=None):
    from hapla import packed_cy

    out = []
    for job in batch:
        meta, G, miss = job[:3]
        try:
            if miss and missing == "error":
                raise ValueError("Missing GT encountered with --missing error")
            if minimum is None:
                results = [packed_cy.fitWindow(G, missing=miss, **opt)]
                results[0]["window"] = meta
            else:
                _, _, results, fits = adaptiveFit(meta, G, miss, opt, minimum, job[3])
                results[0]["candidate_fits"] = fits
        except (ValueError, RuntimeError) as err:
            _, chrom, beg, end, _ = meta
            raise type(err)(f"{chrom}:{beg}-{end}: {err}") from err
        for res in results:
            h = sha256(res["medians"])
            h.update(res["counts"].astype("<u8", copy=False))
            h.update(res["sizes"].astype("<u8", copy=False))
            res["identity"] = h.digest()
            if medians:
                res["likelihoods"] = packed_cy.likelihoods(
                    res["medians"], res["counts"], res["sizes"]
                )
            else:
                del res["medians"]
            del res["counts"], res["sizes"]
            out.append(res)
    return out


### Cluster bounded batches and publish the complete output set
def main(args):
    checkArgs(args)
    nt, nio = threadPlan(args.threads, args.io_threads)
    from hapla.formats import openOutputs, outputSuffixes, writeWindow
    from hapla.identity import fileHash, writeIdentity
    from hapla.windows import (
        adaptiveWindows,
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
    if args.map is not None:
        from hapla.maps import readMap

        data = readMap(args.map)
    else:
        data = None
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
        exact_windows=0,
    )
    if args.adaptive:
        stats.update(candidate_fits=0, smallest_window=args.max_size, largest_window=0)
    printHeader("cluster", args)
    with ExitStack() as stack:
        src, stats["diagnostics"] = openReader(stack, args.vcf, nio)
        print(f"Samples: {len(src.samples):,}\nClustering windows.", flush=True)
        out = stageOutputs(
            stack,
            args.out,
            outputSuffixes(args.medians, args.plink),
            inputs=(args.vcf, args.windows, args.map),
        )
        buf = createBuffer(src.readInto, src.samples, src.contigs, mem // 4)

        # Select window boundaries and leave room for all workers
        if args.adaptive:
            windows = adaptiveWindows(buf, args.max_size, args.max_length, data, args.max_cm)
            n = batchSize(
                args.max_size * (2 * len(src.samples) + 8),
                args.batch_windows,
                mem // 4,
                mem // 2,
                nt,
            )
        elif args.size is not None:
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

            def sites(_):
                site_id.update(src.sites)
                if args.medians:
                    files[".sites"].write(src.sites)

            buf["sites"] = sites

            # Write in input order while other workers continue fitting
            def write(res):
                meta = res["window"]
                fit_key.update(int(meta[0]).to_bytes(8, "little"))
                fit_key.update(res["identity"])
                info = res["stats"]
                K = info["K"]
                stats["windows"] += 1
                stats["clusters"] += K
                stats["missing_assignments"] += info["missing"]
                stats["all_missing_windows"] += K == 0
                stats["capped_windows"] += info["capped"]
                stats["exact_windows"] += info["exact"]
                if args.adaptive:
                    stats["candidate_fits"] += res.get("candidate_fits", 0)
                    stats["smallest_window"] = min(stats["smallest_window"], meta[4])
                    stats["largest_window"] = max(stats["largest_window"], meta[4])
                writeWindow(files, meta, res["labels"], K, stats["windows"])
                if args.medians:
                    files[".bcm"].write(res["medians"])
                    files[".blk"].write(res["likelihoods"])
                    files[".wix"].write(f"{meta[0]}\n")

            runBatches(
                batches(windows, n),
                partial(
                    fitBatch,
                    opt=opt,
                    medians=args.medians,
                    missing=args.missing,
                    minimum=args.min_size if args.adaptive else None,
                ),
                write,
                workers=nt,
                par=args.threads > 1,
                b_buf=mem // 2,
                size_of=lambda batch: sum(
                    job[1].nbytes + (job[3].nbytes if args.adaptive else 0) for job in batch
                ),
            )
        if not stats["windows"]:
            raise ValueError("No windows were produced. Check the input and window/tail settings")
        key = sha256(site_id.digest() + fit_key.digest() + bytes.fromhex(fileHash(out[".win"])))
        writeIdentity(out, key.hexdigest(), stats["windows"], stats["clusters"])

        # Flush staged files before replacing previous outputs
        stats.update(
            samples=len(src.samples),
            variants=buf["variants"],
            haplotypes=2 * len(src.samples),
            read_seconds=buf["time"],
            elapsed_seconds=perf_counter() - start,
            input_buffer_mib=mem / 1024**2,
            workers=nt,
            batch_windows=n,
            io_threads=nio,
            htslib_version=src.htslib_version,
        )
        writeLog(out, "cluster", args, stats)
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
