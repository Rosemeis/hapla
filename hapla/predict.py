"""Predict reference clusters with bounded native input and streamed output."""

from contextlib import ExitStack
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
    if not rows or len(idx) != len(rows):
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
def predictionJobs(buf, ref, R, mode):
    from hapla.windows import advanceBuffer, chromosomeEnd, fillBuffer, takeWindow

    for meta, K, off in ref:
        idx, _, _, _, B = meta
        while buf["idx"] < idx:
            if not fillBuffer(buf, 1):
                raise ValueError("Input ends before a reference window")
            advanceBuffer(buf, min(buf["end"] - buf["beg"], idx - buf["idx"]))
        if fillBuffer(buf, B) < B or chromosomeEnd(buf) < B:
            raise ValueError("Reference window exceeds input or crosses chromosomes")
        cur, G, _, phase = takeWindow(buf, B)
        if cur != meta:
            raise ValueError("Input window coordinates differ from the reference")
        if mode == "unphased":
            phase.fill(True)
        yield meta, G, phase, R[off : off + K * B].reshape(K, B)

    # Check the entire site set, including variants omitted by --tail drop
    while fillBuffer(buf, 1):
        advanceBuffer(buf, buf["end"] - buf["beg"])


### Assign phased haplotypes and apply the unphased cluster-pair heuristic
def predictBatch(batch):
    import numpy as np

    from hapla import packed_cy, shared_cy

    out = []
    for meta, G, phase, R in batch:
        n_unph = 0 if phase is None else int(np.count_nonzero(phase))
        z = (
            np.full(G.shape[1], 255, np.uint8)
            if not len(R) or (phase is not None and n_unph == len(phase))
            else packed_cy.predict_haplotypes(G, R)
        )
        if len(R) and n_unph:
            miss = np.any(G == 255, axis=0).reshape(-1, 2).any(axis=1)
            z.reshape(-1, 2)[phase & miss] = 255
            idx = np.flatnonzero(phase & ~miss)
            if len(idx):
                D = np.ascontiguousarray((G[:, 2 * idx] + G[:, 2 * idx + 1]).T)
                tmp = np.empty(2 * len(idx), np.uint8)
                shared_cy.genoCluster(D, R, tmp)
                z.reshape(-1, 2)[idx] = tmp.reshape(-1, 2)
        out.append((meta, len(R), z, n_unph))
    return out


### Predict bounded batches and publish the complete output set
def main(args):
    if (args.vcf is None) == (args.bfile is None):
        raise ValueError("Provide exactly one of --bcf/--vcf or --bfile")
    if args.ref is None:
        raise ValueError("Provide a Hapla 1.x reference with --ref")
    if args.buffer_mb < 1 or args.batch_windows < 1:
        raise ValueError("--buffer-mb and --batch-windows must be positive")
    if args.bfile is not None and args.phase_mode == "phased":
        raise ValueError("PLINK BED input is unphased")
    nt, nio = threadPlan(args.threads, args.io_threads)
    from hapla.formats import FORMAT_VERSION, openOutputs, outputSuffixes, readHeader, writeWindow
    from hapla.identity import readIdentity, writeIdentity
    from hapla.plink import openPlink
    from hapla.windows import createBuffer

    # Map the reference and protect all input paths
    start = perf_counter()
    ident = readIdentity(args.ref, (".bcm", ".win", ".wix", ".sites"))
    ref, R = readReference(args.ref)
    if ident["windows"] != len(ref) or ident["clusters"] != sum(k for _, k, _ in ref):
        raise ValueError("Reference identity dimensions do not match the medians")
    mem = args.buffer_mb * 1024**2
    inputs = [args.vcf] if args.vcf else [f"{args.bfile}{s}" for s in (".bed", ".bim", ".fam")]
    inputs += [f"{args.ref}{s}" for s in (".bcm", ".win", ".wix", ".sites", ".ref.json")]
    stats = dict(windows=0, missing_assignments=0, unphased_sample_windows=0)
    printHeader("predict", args.threads)
    with ExitStack() as stack:
        out = stageOutputs(stack, args.out, outputSuffixes(False, args.plink), inputs)
        sites = stack.enter_context(open(f"{args.ref}.sites", "rb"))
        readHeader(sites)
        if args.vcf:
            src, stats["diagnostics"] = openReader(
                stack, args.vcf, nio, phased=args.phase_mode == "phased"
            )
            read, ids, chroms = src.read_into, src.samples, src.contigs
        else:
            rows = []
            read, ids, chroms = openPlink(stack, args.bfile, rows)
        print(
            f"Data size: {len(ids):,} samples, {len(ref):,} windows\nPredicting clusters.",
            flush=True,
        )

        # Require exact chromosome, position, REF and ALT order
        def checkSites(eof):
            for row in src.sites if args.vcf else rows:
                if sites.readline() != row:
                    raise ValueError(
                        "Input sites differ from the reference (chromosome, position, REF/ALT order)"
                    )
            if eof and sites.read(1):
                raise ValueError("Input ends before the reference variant set")

        buf = createBuffer(
            read, ids, chroms, mem // 4, phase=args.phase_mode != "phased", sites=checkSites
        )
        b_sample = 2 if args.phase_mode == "phased" else 3
        n = batchSize(
            max(meta[4] for meta, _, _ in ref) * len(ids) * b_sample,
            args.batch_windows,
            mem // 4,
            mem // 2,
            nt,
        )
        with ExitStack() as io:
            files = openOutputs(io, out, ids, args.duplicate_fid)

            # Publish each result in reference order
            def write(res):
                meta, K, z, n_unph = res
                stats["windows"] += 1
                stats["missing_assignments"] += int((z == 255).sum())
                stats["unphased_sample_windows"] += n_unph
                writeWindow(files, meta, z, K, stats["windows"])

            runBatches(
                batches(predictionJobs(buf, ref, R, args.phase_mode), n),
                predictBatch,
                write,
                workers=nt,
                par=args.threads > 1,
                b_buf=mem // 2,
                size_of=lambda batch: sum(
                    G.nbytes + (0 if phase is None else phase.nbytes) for _, G, phase, _ in batch
                ),
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
            format_version=FORMAT_VERSION,
        )
        if args.vcf:
            stats.update(htslib_version=src.htslib_version, htslib_features=src.htslib_features)
        writeLog(out[".log"], "predict", args, stats)
        commitOutputs(args.out, out)
    printMissing(stats["missing_assignments"], stats["windows"] * stats["haplotypes"])
    printDone(args.out, out, stats["elapsed_seconds"])
    return stats
