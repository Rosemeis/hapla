"""Bounded PLINK BED input. BIM A2 is REF and A1 is ALT."""

import numpy as np

from hapla import shared_cy


### Open PLINK files on the caller's cleanup stack and return a block reader
def openPlink(stack, pfx, sites):
    with open(f"{pfx}.fam") as src:
        rows = [line.split() for line in src if line.strip()]
    if not rows or any(len(row) != 6 for row in rows):
        raise ValueError("PLINK FAM requires six fields per sample")
    ids = [row[1] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("Sample IDs must be unique")
    width = (len(ids) + 3) // 4
    bed = stack.enter_context(open(f"{pfx}.bed", "rb"))
    if bed.read(3) != bytes((108, 27, 1)):
        raise ValueError("Expected a SNP-major PLINK BED file")
    bim = stack.enter_context(open(f"{pfx}.bim", "rb"))
    chroms, rid = [], {}
    prev = (-1, -1)

    # Decode one block and check its site order against the reference
    def read(G, pos, contigs, miss, phase):
        nonlocal prev
        if phase is None:
            raise ValueError("PLINK prediction requires a phase buffer")
        raw = bed.read(len(G) * width)
        if len(raw) % width:
            raise ValueError("Truncated PLINK BED row")
        n = len(raw) // width
        if n:
            X = np.frombuffer(raw, np.uint8).reshape(n, width)
            D = np.empty((n, len(ids)), np.uint8)
            shared_cy.readPlink(X, D)
            G[:n, 0::2], G[:n, 1::2] = D == 2, D >= 1
            mask = D == 9
            G[:n, 0::2][mask] = 255
            G[:n, 1::2][mask] = 255
            miss[:n], phase[:n] = np.any(mask, axis=1), 1
        sites.clear()
        for i in range(n):
            row = bim.readline().split()
            if len(row) != 6 or row[4] in (b"0", b".") or row[5] in (b"0", b"."):
                raise ValueError("Missing or invalid PLINK BIM row/alleles")
            chrom, bp = row[0].decode("utf-8"), int(row[3])
            if chrom not in rid:
                rid[chrom] = len(chroms)
                chroms.append(chrom)
            cur = (rid[chrom], bp)
            if bp < 1 or cur < prev:
                raise ValueError("PLINK variants must be sorted by chromosome and position")
            contigs[i], pos[i] = prev = cur
            sites.append(b"\t".join((row[0], str(bp).encode(), row[5], row[4])) + b"\n")
        if n < len(G) and bim.read().strip():
            raise ValueError("PLINK BIM has more variants than BED")
        return n

    return read, ids, chroms
