"""Individual groups from the implicit haplotype sharing kernel."""

__author__ = "Jonas Meisner"

import numpy as np

from hapla import sharing

##### Centroids


### Apply the complete kernel to normalized group indicators
def state(data, Z, K, mean=None):
    N = len(data[0][0]) - 1
    Z = np.ascontiguousarray(Z, np.int32)
    if Z.ndim != 1 or len(Z) != N:
        raise ValueError("Sharing group labels differ from the sample count")
    n = np.bincount(Z, minlength=K)
    if len(n) != K or np.any(n == 0):
        raise ValueError("Sharing groups must be nonempty")
    if mean is None:
        mean, _ = sharing.moments(data)
    if np.shape(mean) != (N,):
        raise ValueError("Sharing means differ from the profile dimensions")
    F = np.empty((N, K))
    sharing.cy.centroids(*data[1], Z, 1 / n, mean, F)
    A = sharing.product(data, F)
    b = np.bincount(Z, weights=A[np.arange(N), Z], minlength=K) / n
    A *= -2
    A += b
    return A, b, n


### Number groups by their first sample
def labels(Z, K):
    first = np.full(K, len(Z), np.int64)
    np.minimum.at(first, Z, np.arange(len(Z)))
    inv = np.empty(K, np.int32)
    inv[np.argsort(first)] = np.arange(K)
    return inv[Z]


### Retain ties and refill empty groups with a singleton
def assign(D, Z, K, X):
    N = len(Z)
    idx = np.arange(N)
    out = np.argmin(D, axis=1).astype(np.int32)
    old, new = D[idx, Z], D[idx, out]
    tol = 32 * np.finfo(float).eps * np.maximum(np.abs(old), np.abs(new))
    keep = old <= new + tol
    out[keep] = Z[keep]
    n = np.bincount(out, minlength=K)
    empty = np.flatnonzero(n == 0)
    if len(empty):
        # PCA distances choose a split without another kernel product.
        A = np.zeros((K, X.shape[1]))
        np.add.at(A, out, X)
        A /= np.maximum(n, 1)[:, None]
        d = np.sum((X - A[out]) ** 2, axis=1)
        for g in empty:
            i = np.argmax(np.where(n[out] > 1, d, -np.inf))
            n[out[i]] -= 1
            out[i], n[g], d[i] = g, 1, 0
    return out


##### Initialization


### Seed PCA scores with k-means++ and retain duplicate seed identities
def initial(X, K, rng):
    N = len(X)
    picks = np.empty(K, np.int64)
    picks[0] = rng.integers(N)
    d = np.sum((X - X[picks[0]]) ** 2, axis=1)
    for g in range(1, K):
        d[picks[:g]] = 0
        total = d.sum()
        if total > 0:
            picks[g] = rng.choice(N, p=d / total)
        else:
            free = np.ones(N, bool)
            free[picks[:g]] = False
            picks[g] = rng.choice(np.flatnonzero(free))
        np.minimum(d, np.sum((X - X[picks[g]]) ** 2, axis=1), out=d)
    D = np.sum(X * X, axis=1)[:, None] + np.sum(X[picks] ** 2, axis=1)
    D -= 2 * X @ X[picks].T
    Z = np.argmin(D, axis=1).astype(np.int32)
    Z[picks] = np.arange(K)
    for it in range(41):
        n = np.bincount(Z, minlength=K)
        A = np.zeros((K, X.shape[1]))
        np.add.at(A, Z, X)
        A /= n[:, None]
        if it == 40:
            break
        D = X @ A.T
        D *= -2
        D += np.sum(A * A, axis=1)
        out = assign(D, Z, K, X)
        if np.array_equal(out, Z):
            break
        Z = out
    return float(np.sum((X - A[Z]) ** 2)), Z


##### Kernel grouping


### Refine a flat partition against all sharing features
def refine(data, Z, K, X, ss, mean=None):
    if mean is None:
        mean, _ = sharing.moments(data)
    history = []
    for it in range(101):
        D, b, n = state(data, Z, K, mean)
        loss = max(0.0, float(ss - n @ b))
        history.append(loss)
        if it == 100:
            return Z, D, loss, it, False, history
        out = assign(D, Z, K, X)
        if np.array_equal(out, Z):
            return Z, D, loss, it, True, history
        Z = out


### Initialize in PCA space, then optimize the complete implicit kernel
def fit(data, K, power=11, seed=42, scores=None, mom=None):
    N = len(data[0][0]) - 1
    if not isinstance(K, (int, np.integer)) or not 1 <= K < N:
        raise ValueError("Sharing groups must be between one and samples minus one")
    if not isinstance(power, (int, np.integer)) or power < 1 or seed < 0:
        raise ValueError("Sharing power must be positive and seed nonnegative")
    mean, ss = sharing.moments(data) if mom is None else mom
    scale = (N - 1) / ss
    if K == 1:
        return (
            np.zeros(N, np.int32),
            np.zeros(N),
            dict(
                iterations=0,
                converged=True,
                stop="unchanged",
                objective=float(N - 1),
                group_sizes=[N],
            ),
        )
    if len(data) != 2:
        raise ValueError("Sharing grouping requires cached transpose profiles")
    if scores is None:
        r = min(max(K, 10), 20, N - 1)
        V, S = sharing.pca(data, r, power, seed, strict=False)
        X = V * S
    else:
        X = np.asarray(scores, dtype=np.float64)
        if X.ndim != 2 or len(X) != N or X.shape[1] == 0 or not np.isfinite(X).all():
            raise ValueError("Sharing initialization scores must be finite sample rows")
    rng = np.random.default_rng(seed)
    starts = sorted((initial(X, K, rng) for _ in range(4)), key=lambda x: x[0])
    best, seen = None, set()
    for _, Z in starts:
        key = labels(Z, K).tobytes()
        if key in seen:
            continue
        seen.add(key)
        cur = refine(data, Z, K, X, ss, mean)
        if best is None or cur[2] < best[2]:
            best = cur
        if len(seen) == 2:
            break
    Z, D, loss, it, done, _ = best
    idx = np.arange(N)
    old = D[idx, Z]
    D[idx, Z] = np.inf
    gap = (np.min(D, axis=1) - old) * scale
    gap[np.abs(gap) <= 64 * np.finfo(float).eps * (N - 1)] = 0
    Z = labels(Z, K)
    return (
        Z,
        gap,
        dict(
            iterations=it,
            converged=done,
            stop="unchanged" if done else "iteration limit",
            objective=loss * scale,
            group_sizes=np.bincount(Z, minlength=K).tolist(),
        ),
    )
