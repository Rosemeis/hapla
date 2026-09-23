"""Bind assignment bundles and projection models to ordered cluster definitions."""

__author__ = "Jonas Meisner"

import json
from hashlib import sha256
from pathlib import Path


### Hash files in bounded blocks without expanding genotype or loading arrays
def fileHash(pth):
    h = sha256()
    with Path(pth).open("rb") as src:
        for block in iter(lambda: src.read(1024**2), b""):
            h.update(block)
    return h.hexdigest()


### Read and validate identity metadata
def readRecord(pth):
    if not Path(pth).is_file():
        raise ValueError(f"Missing identity metadata: {pth}. Regenerate clusters or the PCA model")
    obj = json.loads(Path(pth).read_text())
    if (
        not isinstance(obj, dict)
        or obj.get("version") != 1
        or not isinstance(obj.get("files"), dict)
    ):
        raise ValueError(f"Invalid identity metadata: {pth}")
    return obj


### Check each file against its recorded hash
def checkFiles(pfx, obj, sfxs):
    for s in sfxs:
        if obj["files"].get(s) != fileHash(f"{pfx}{s}"):
            raise ValueError(f"File does not match its identity metadata: {pfx}{s}")


### Preserve reference identity while binding each query's own assignments and IDs
def writeIdentity(out, ref, windows, K):
    obj = dict(version=1, reference=ref, windows=int(windows), clusters=int(K))
    obj["files"] = {
        s: fileHash(out[s]) for s in (".bca", ".ids", ".win", ".bcm", ".wix", ".sites") if s in out
    }
    out[".ref.json"].write_text(json.dumps(obj, indent=2) + "\n")


### Validate reference dimensions and assignment files
def readIdentity(pfx, sfxs):
    obj = readRecord(f"{pfx}.ref.json")
    ref = obj.get("reference")
    if (
        not isinstance(ref, str)
        or len(ref) != 64
        or any(c not in "0123456789abcdef" for c in ref)
        or type(obj.get("windows")) is not int
        or obj["windows"] < 1
        or type(obj.get("clusters")) is not int
        or obj["clusters"] < 0
    ):
        raise ValueError("Invalid cluster reference identity")
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
        keys.append(
            {k: obj[k] for k in ("reference", "windows", "clusters")}
            | {"window_sha256": obj["files"][".win"]}
        )
        start = end
    return keys


### All projection parameters belong to one published model generation
def writeModel(out, keys):
    obj = dict(
        version=1,
        features=keys,
        files={s: fileHash(out[s]) for s in (".freqs", ".loadings", ".eigenvals")},
    )
    out[".pca.json"].write_text(json.dumps(obj, indent=2) + "\n")


### Match the ordered references and saved projection parameters
def checkModel(pfx, keys):
    obj = readRecord(f"{pfx}.pca.json")
    if obj.get("features") != keys:
        raise ValueError("Ordered cluster references do not match the PCA model")
    checkFiles(pfx, obj, (".freqs", ".loadings", ".eigenvals"))
