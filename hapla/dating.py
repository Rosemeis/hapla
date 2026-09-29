"""Fixed-frequency single-pulse dating from the ancestry HMM likelihood."""

__author__ = "Jonas Meisner"

from pathlib import Path
from time import perf_counter

import numpy as np

from hapla import fatash_cy as cy
from hapla.formats import sampleIndices
from hapla.maps import chromosome
from hapla.runtime import printTiming


### Bracket the best grid value and refine in log generations
def maximize(score, lo, hi):
    grid = np.linspace(np.log(lo), np.log(hi), 17)
    vals = np.array([score(x) for x in grid])
    if not np.isfinite(vals).all():
        raise ValueError("Date likelihood is not finite")
    if np.ptp(vals) <= 1e-6:
        return None, vals[0], "flat"
    i = int(vals.argmax())
    a, b = grid[max(0, i - 1)], grid[min(len(grid) - 1, i + 1)]
    best = grid[i], vals[i]
    r = (np.sqrt(5) - 1) / 2
    x, y = b - r * (b - a), a + r * (b - a)
    fx, fy = score(x), score(y)
    for _ in range(40):
        best = max(best, (x, fx), (y, fy), key=lambda v: v[1])
        if b - a <= 1e-3:
            break
        if fx > fy:
            b, y, fy = y, x, fx
            x = b - r * (b - a)
            fx = score(x)
        else:
            a, x, fx = x, y, fy
            y = a + r * (b - a)
            fy = score(y)
    t = float(np.exp(best[0]))
    status = "converged"
    if best[0] == grid[0]:
        status = "lower_bound"
    elif best[0] == grid[-1]:
        status = "upper_bound"
    return t, best[1], status


### Cache compact target labels once, then score chromosomes with bounded scratch
def fit(data, P, Q, chroms, ids, dist, args, tmp):
    tick = perf_counter()
    pick = (
        sampleIndices(ids, args.date_samples, ordered=True)
        if args.date_samples
        else np.arange(len(ids))
    )
    q = np.repeat(Q[pick], 2, axis=0)
    K = Q.shape[1]
    h = (2 * pick[:, None] + [0, 1]).ravel()
    parts, index = [], {}
    for (Z, c, use, regions, size), p, names, d in zip(data, P, chroms, dist):
        for (beg, end), name in zip(regions, names):
            z = np.memmap(
                Path(tmp) / str(len(parts)), mode="w+", dtype=np.uint8, shape=(len(h), end - beg)
            )
            for i in range(0, len(h), size):
                z[i : i + size] = Z[beg:end, h[i : i + size]].T
            j = index.setdefault(chromosome(name), len(index))
            a, b = c[beg], c[end]
            f, k = p[a * K : b * K], c[beg : end + 1] - a
            parts.append((z, f, k, use[beg:end], d[beg:end], j))
    C = len(index)
    if args.date_jackknife and C < 3:
        raise ValueError("Date jackknife requires at least three chromosomes")
    cache = {}

    def scores(x):
        if x not in cache:
            ll = np.zeros(C)
            t = np.exp(x)
            for z, p, c, use, d, j in parts:
                ll[j] += cy.mapScore(z, p, c, use, q, d, t).sum()
            cache[x] = ll
        return cache[x]

    t, ll, status = maximize(lambda x: scores(x).sum(), args.date_min, args.date_max)
    if t is None:
        raise ValueError("Admixture time is not identifiable from the selected samples and windows")
    info = dict(
        generations=t,
        status=status,
        log_likelihood=ll,
        samples=len(pick),
        chromosomes=C,
        min_generations=args.date_min,
        max_generations=args.date_max,
    )
    printTiming(f"Date: {t:,.3f} generations ({status.replace('_', ' ')}).", perf_counter() - tick)
    if args.date_jackknife:
        lap = perf_counter()
        dates, valid = [], status == "converged"
        for j in range(C):
            keep = np.arange(C) != j
            tj, _, stop = maximize(lambda x: scores(x)[keep].sum(), args.date_min, args.date_max)
            dates.append(float("nan") if tj is None else tj)
            valid &= stop == "converged"
        info["jackknife_dates"] = dates
        info["jackknife_status"] = "complete" if valid else "unresolved"
        if valid:
            x = np.log(dates)
            se = np.sqrt((C - 1) / C * np.sum((x - x.mean()) ** 2))
            info.update(
                log_time_se=float(se),
                lower=float(t * np.exp(-1.96 * se)),
                upper=float(t * np.exp(1.96 * se)),
            )
        printTiming(
            "Chromosome jackknife complete." if valid else "Chromosome jackknife unresolved.",
            perf_counter() - lap,
        )
    info.update(evaluations=len(cache), seconds=perf_counter() - tick)
    return info


### Write the fitted time and optional chromosome uncertainty in one row
def writeDate(path, info):
    keys = (
        "generations",
        "lower",
        "upper",
        "log_time_se",
        "status",
        "samples",
        "chromosomes",
        "log_likelihood",
    )
    row = [
        f"{info[k]:.10g}"
        if isinstance(info.get(k), (float, np.floating))
        else str(info.get(k, "NA"))
        for k in keys
    ]
    Path(path).write_text("\t".join(keys) + "\n" + "\t".join(row) + "\n")
