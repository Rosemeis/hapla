"""Predict reference clusters with bounded native input and streamed output."""

__author__ = "Jonas Meisner"

from contextlib import ExitStack
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


### Read reference tuples: (window metadata, cluster count, median offset)
def readReference(pfx):
    from hapla.formats import readMedians, readWindows

    with open(f"{pfx}.wix") as src:
        idx = [int(line.strip()) for line in src if line.strip()]
    rows = readWindows(pfx)
    if len(idx) != len(rows):
        raise ValueError("Reference window and index counts differ or are empty")
    ref, off, prev = [], 0, -1
    for i, row in zip(idx, rows):
        chrom, beg, end, _, B, K = row
        if i <= prev:
            raise ValueError("Reference window indices must be nonnegative and increasing")
        ref.append(((i, chrom, beg, end, B), K, off))
        off += K * B
        prev = i
    R = readMedians(pfx, [K for _, K, _ in ref], [meta[4] for meta, _, _ in ref])
    return ref, R


### Align input windows and retain views into the mapped reference medians
def predictionJobs(buf, ref, R):
    from hapla.windows import advanceBuffer, chromosomeEnd, fillBuffer, takeWindow

    for meta, K, off in ref:
        idx, _, _, _, B = meta
        while buf["idx"] < idx:
            if not fillBuffer(buf, 1):
                raise ValueError("Input ends before a reference window")
            advanceBuffer(buf, min(buf["end"] - buf["beg"], idx - buf["idx"]))
        if fillBuffer(buf, B) < B or chromosomeEnd(buf) < B:
            raise ValueError("Reference window exceeds input or crosses chromosomes")
        cur, G, _ = takeWindow(buf, B)
        if cur != meta:
            raise ValueError("Input window coordinates differ from the reference")
        yield meta, G, R[off : off + K * B].reshape(K, B)

    # Check the entire site set, including variants omitted by --tail drop
    while fillBuffer(buf, 1):
        advanceBuffer(buf, buf["end"] - buf["beg"])


### Assign each observed haplotype to its nearest reference median
def predictBatch(batch):
    import numpy as np

    from hapla import packed_cy

    out = []
    for meta, G, R in batch:
        z = packed_cy.predictHaplotypes(G, R) if len(R) else np.full(G.shape[1], 255, np.uint8)
        out.append((meta, len(R), z))
    return out


### Predict bounded batches and publish the complete output set
def main(args):
    if args.vcf is None or not Path(args.vcf).is_file():
        raise ValueError("Provide an existing phased VCF/BCF with --bcf or --vcf")
    if args.ref is None:
        raise ValueError("Provide a Hapla 1.x reference with --ref")
    if args.buffer_mb < 1 or args.batch_windows < 1:
        raise ValueError("--buffer-mb and --batch-windows must be positive")

    # Balance decompression with the cheaper nearest-median worker
    io = args.io_threads
    if io is None:
        io = min(3, max(0, (args.threads - 2) // 2))
    nt, nio = threadPlan(args.threads, io)
    from hapla.formats import openOutputs, outputSuffixes, readHeader, writeWindow
    from hapla.identity import readIdentity, writeIdentity
    from hapla.windows import createBuffer

    # Map the reference and protect all input paths
    start = perf_counter()
    ident = readIdentity(args.ref, (".bcm", ".win", ".wix", ".sites"))
    ref, R = readReference(args.ref)
    if ident["windows"] != len(ref) or ident["clusters"] != sum(k for _, k, _ in ref):
        raise ValueError("Reference identity dimensions do not match the medians")
    mem = args.buffer_mb * 1024**2
    inputs = [args.vcf]
    inputs += [f"{args.ref}{s}" for s in (".bcm", ".win", ".wix", ".sites", ".ref.json")]
    stats = dict(windows=0, missing_assignments=0)
    printHeader("predict", args.threads)
    with ExitStack() as stack:
        out = stageOutputs(stack, args.out, outputSuffixes(False, args.plink), inputs)
        sites = stack.enter_context(open(f"{args.ref}.sites", "rb"))
        readHeader(sites)
        src, stats["diagnostics"] = openReader(stack, args.vcf, nio)
        ids = src.samples
        print(
            f"Data size: {len(ids):,} samples, {len(ref):,} windows\nPredicting clusters.",
            flush=True,
        )

        # Require exact chromosome, position, REF and ALT order
        def checkSites(eof):
            if sites.read(len(src.sites)) != src.sites:
                raise ValueError(
                    "Input sites differ from the reference (chromosome, position, REF/ALT order)"
                )
            if eof and sites.read(1):
                raise ValueError("Input ends before the reference variant set")

        buf = createBuffer(src.readInto, ids, src.contigs, mem // 4, sites=checkSites)
        n = batchSize(
            max(meta[4] for meta, _, _ in ref) * len(ids) * 2,
            args.batch_windows,
            mem // 4,
            mem // 2,
            nt,
        )
        with ExitStack() as io:
            files = openOutputs(io, out, ids, args.duplicate_fid)

            # Publish each result in reference order
            def write(res):
                meta, K, z = res
                stats["windows"] += 1
                stats["missing_assignments"] += int((z == 255).sum())
                writeWindow(files, meta, z, K, stats["windows"])

            runBatches(
                batches(predictionJobs(buf, ref, R), n),
                predictBatch,
                write,
                workers=nt,
                par=args.threads > 1,
                b_buf=mem // 2,
                size_of=lambda batch: sum(G.nbytes for _, G, _ in batch),
            )

        writeIdentity(out, ident["reference"], ident["windows"], ident["clusters"])

        # Flush staged files before replacing previous outputs
        stats.update(
            variants=buf["variants"],
            haplotypes=2 * len(ids),
            elapsed_seconds=perf_counter() - start,
            read_seconds=buf["time"],
            workers=nt,
            batch_windows=n,
            io_threads=nio,
            input_buffer_mib=mem / 1024**2,
            htslib_version=src.htslib_version,
        )
        writeLog(out[".log"], "predict", args, stats)
        commitOutputs(args.out, out)
    printMissing(stats["missing_assignments"], stats["windows"] * stats["haplotypes"])
    printDone(args.out, out, stats["elapsed_seconds"])
    return stats
