"""Hapla 1.x binary boundaries. Analyses from earlier versions must be rerun."""

__author__ = "Jonas Meisner"

from pathlib import Path

import numpy as np

MAX_CLUSTERS = 255
FORMAT_VERSION = 1
MAGIC = b"HAPLA\x01\r\n"


### Resolve an ordered path list or read one from a file
def readPaths(pth=None, values=None):
    if (pth is None) == (values is None):
        raise ValueError("Provide exactly one path list or filelist")
    if values is None:
        paths = [s.strip() for s in Path(pth).read_text().splitlines() if s.strip()]
    else:
        paths = [values] if isinstance(values, (str, Path)) else list(values)
    if not paths or any(not str(p).strip() for p in paths):
        raise ValueError(f"Input list is empty: {pth if pth is not None else values}")
    return paths


### Read one sample ID per line, preserving order
def readIds(pth):
    ids = Path(pth).read_text().splitlines()
    if not ids or any(not s or any(c.isspace() for c in s) for s in ids):
        raise ValueError("Sample file must contain one nonempty ID per line, without whitespace")
    return np.asarray(ids, dtype=np.str_)


### Check the sample sidecar when reading a saved Hapla Q matrix
def checkQIds(pth, ids):
    path = Path(pth)
    side = path.with_suffix(".ids") if path.suffix == ".Q" else None
    if side is None or not side.is_file():
        return None
    if not np.array_equal(readIds(side), ids):
        raise ValueError("Q sample IDs differ from cluster sample order")
    return side


### Read six exact window fields before passing counts to native kernels
def readWindows(pfx):
    rows = []
    with open(f"{pfx}.win") as src:
        for line in src:
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) != 6:
                raise ValueError("Window metadata requires six fields per row")
            chrom, beg, end, length, size, count = fields[0], *map(int, fields[1:])
            if beg < 1 or end < beg or length != end - beg or size < 1 or not 0 <= count <= 255:
                raise ValueError(
                    "Invalid window coordinates, size, or cluster count (expected 0..255)"
                )
            rows.append((chrom, beg, end, length, size, count))
    if not rows:
        raise ValueError("Window metadata is empty")
    return rows


### Resolve sample subsets in Q order for evaluation and input order for fitting
def sampleIndices(ids, pth, *, ordered=False):
    keep = readIds(pth)
    index = {s: i for i, s in enumerate(ids)}
    if len(np.unique(keep)) != len(keep):
        raise ValueError("Keep file contains duplicate sample IDs")
    if any(s not in index for s in keep):
        raise ValueError("Keep file contains sample IDs absent from the input")
    idx = np.array([index[s] for s in keep], dtype=np.uint32)
    return idx if ordered else np.sort(idx)


### Collect chromosome metadata without repeatedly growing NumPy arrays
def readMetadata(pfx, files=None, *, likes=False):
    paths = readPaths(files, pfx)
    ids, counts = None, []
    for pth in paths:
        for sfx in (".bca", ".ids", ".win") + ((".blk",) if likes else ()):
            if not Path(f"{pth}{sfx}").is_file():
                raise ValueError(f"{sfx[1:]} file doesn't exist: {pth}{sfx}")
        cur = readIds(f"{pth}.ids")
        if not len(cur) or len(np.unique(cur)) != len(cur):
            raise ValueError("Sample IDs must be nonempty and unique")
        if ids is None:
            ids = cur
        elif not np.array_equal(ids, cur):
            raise ValueError("Samples do not match across files!")
        counts.append(np.array([r[5] for r in readWindows(pth)], dtype=np.uint32))
    return paths, ids, np.concatenate(counts), np.array([len(k) for k in counts], np.uint32)


### Reject incompatible binary headers before reading payloads
def readHeader(handle):
    if handle.read(len(MAGIC)) != MAGIC:
        raise ValueError(
            f"Not a Hapla 1.x file: {handle.name}. Analyses from versions before 1.0.0 "
            "are incompatible. Rerun clustering and downstream analyses with Hapla 1.0.0+."
        )
    return FORMAT_VERSION


### Map assignments after checking their header and exact payload dimensions
def mapLabels(pfx, W, N):
    W = int(W)
    pth = Path(f"{pfx}.bca")
    with pth.open("rb") as src:
        readHeader(src)
    if W < 1 or N < 1 or pth.stat().st_size != len(MAGIC) + W * 2 * N:
        raise ValueError("Assignment payload does not match the window and sample metadata")
    return np.memmap(pth, mode="r", dtype=np.uint8, offset=len(MAGIC), shape=(W, 2 * N))


### Map reference medians for each prediction kernel to validate its slice
def readMedians(prefix, K, sizes):
    counts = np.asarray(K, dtype=np.int64).reshape(-1)
    widths = np.asarray(sizes, dtype=np.int64).reshape(-1)
    path = Path(f"{prefix}.bcm")
    with path.open("rb") as handle:
        readHeader(handle)
    if counts.shape != widths.shape or np.any(widths < 1) or np.any(counts < 0):
        raise ValueError("Invalid median metadata")
    if np.any(counts > MAX_CLUSTERS):
        raise ValueError("Cluster count exceeds the median format's limit")
    size = sum(int(k) * int(b) for k, b in zip(counts, widths))
    if path.stat().st_size != len(MAGIC) + size:
        raise ValueError("Invalid binary median payload dimensions")
    return (
        np.memmap(path, mode="r", dtype=np.uint8, offset=len(MAGIC), shape=(size,))
        if size
        else np.empty(0, np.uint8)
    )


### Declare the cluster or prediction output set
def outputSuffixes(medians, plink):
    out = [".bca", ".ids", ".win", ".ref.json", ".log"]
    if medians:
        out += [".bcm", ".blk", ".wix", ".sites"]
    if plink:
        out += [".bed", ".bim", ".fam"]
    return out


### Open buffered outputs and write shared sample metadata and headers
def openOutputs(stack, out, ids, dup=False):
    out[".ids"].write_text("".join(f"{s}\n" for s in ids))
    if ".fam" in out:
        with out[".fam"].open("w") as fam:
            for s in ids:
                fam.write(f"{s if dup else '0'}\t{s}\t0\t0\t0\t-9\n")
    files = {
        s: stack.enter_context(
            p.open(
                "wb" if s in (".bca", ".bcm", ".blk", ".bed", ".sites") else "w", buffering=1024**2
            )
        )
        for s, p in out.items()
        if s not in (".ids", ".fam", ".log", ".ref.json")
    }
    for s in (".bca", ".bcm", ".blk", ".sites"):
        if s in files:
            files[s].write(MAGIC)
    if ".bed" in files:
        files[".bed"].write(bytes((108, 27, 1)))
    files[".win"].write("#CHROM\tSTART\tEND\tLENGTH\tSIZE\tK\n")
    return files


### Write one window and optional PLINK cluster dosages
def writeWindow(files, meta, z, K, w):
    _, chrom, beg, end, B = meta
    files[".bca"].write(z)
    files[".win"].write(f"{chrom}\t{beg}\t{end}\t{end - beg}\t{B}\t{K}\n")
    if ".bed" in files:
        from hapla import packed_cy

        files[".bed"].write(packed_cy.plink_window(z, K))
        for k in range(K):
            label = f"{chrom}_W{w}_K{k + 1}_B{B}"
            files[".bim"].write(f"{chrom}\t{label}\t0\t{beg}\tK\t0\n")
