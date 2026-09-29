"""Bind assignment bundles and projection models to ordered cluster definitions."""

__author__ = "Jonas Meisner"

from hashlib import sha256
from pathlib import Path


### Hash files in bounded blocks without expanding genotype or loading arrays
def fileHash(pth):
    h = sha256()
    with Path(pth).open("rb") as src:
        for block in iter(lambda: src.read(1024**2), b""):
            h.update(block)
    return h.hexdigest()


### Read a versioned table of identity fields and file hashes
def readRecord(pth, kind):
    if not Path(pth).is_file():
        raise ValueError(f"Missing identity metadata: {pth}. Regenerate clusters or the PCA model")
    rows = Path(pth).read_text().splitlines()
    if not rows or rows[0] != f"#HAPLA {kind} 1":
        raise ValueError(f"Invalid identity metadata: {pth}")
    obj = {}
    for line in rows[1:]:
        row = line.split()
        if len(row) != 2 or row[0] in obj:
            raise ValueError(f"Invalid identity metadata: {pth}")
        obj[row[0]] = row[1]
    return obj


### Write compact identity metadata without paths or sample-dependent feature keys
def writeRecord(pth, kind, obj):
    Path(pth).write_text(f"#HAPLA {kind} 1\n" + "".join(f"{k}\t{v}\n" for k, v in obj.items()))


### Check each file against its recorded hash
def checkFiles(pfx, obj, sfxs):
    for s in sfxs:
        if obj.get(s) != fileHash(f"{pfx}{s}"):
            raise ValueError(f"File does not match its identity metadata: {pfx}{s}")


### Preserve reference identity while binding each query's own assignments and IDs
def writeIdentity(out, ref, windows, K):
    obj = dict(reference=ref, windows=int(windows), clusters=int(K))
    obj.update(
        {
            s: fileHash(out[s])
            for s in (".bca", ".ids", ".win", ".bcm", ".wix", ".sites")
            if s in out
        }
    )
    writeRecord(out[".ref"], "REF", obj)


### Validate reference dimensions and assignment files
def readIdentity(pfx, sfxs):
    obj = readRecord(f"{pfx}.ref", "REF")
    try:
        ref = obj["reference"]
        obj["windows"], obj["clusters"] = int(obj["windows"]), int(obj["clusters"])
        if (
            len(ref) != 64
            or any(c not in "0123456789abcdef" for c in ref)
            or obj["windows"] < 1
            or obj["clusters"] < 0
        ):
            raise ValueError
    except (KeyError, ValueError) as exc:
        raise ValueError("Invalid cluster reference identity") from exc
    checkFiles(pfx, obj, sfxs)
    return obj


### Compare references in file order, independently of the query sample set
def featureKeys(paths, counts, sizes):
    keys, start = [], 0
    for pfx, size in zip(paths, sizes):
        obj = readIdentity(pfx, (".bca", ".ids", ".win"))
        end = start + int(size)
        if obj["windows"] != int(size) or obj["clusters"] != int(counts[start:end].sum()):
            raise ValueError("Cluster identity dimensions do not match the window metadata")
        keys.append(f"{obj['reference']} {obj['windows']} {obj['clusters']} {obj['.win']}")
        start = end
    return keys


### All projection parameters belong to one published model generation
def writeModel(out, keys):
    obj = dict(features=sha256("\n".join(keys).encode()).hexdigest())
    obj.update({s: fileHash(out[s]) for s in (".freq", ".load", ".val")})
    writeRecord(out[".pca"], "PCA", obj)


### Match the ordered references and saved projection parameters
def checkModel(pfx, keys):
    obj = readRecord(f"{pfx}.pca", "PCA")
    if obj.get("features") != sha256("\n".join(keys).encode()).hexdigest():
        raise ValueError("Ordered cluster references do not match the PCA model")
    checkFiles(pfx, obj, (".freq", ".load", ".val"))
