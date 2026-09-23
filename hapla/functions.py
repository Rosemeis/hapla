"""Admixture updates and initialization from centered cluster labels."""

__author__ = "Jonas Meisner"

import numpy as np

from hapla import admix_cy

##### Admixture updates


### Target 8 MiB of Q scratch, with at least one sample per partition
def emWorkspace(N, K, k):
    B = min(64, len(k))
    tile = min(N, max(1, 8 * 1024**2 // (8 * B * K)))
    return np.empty((B, int(k.max()) * K)), np.empty((B, tile, K))


### One EM update, sharing scratch across full, batch, and projection modes
def emStep(P, Q, Pn, Qn, ctx, rows=None, qo=None, pool=None, prior=0.0, scratch=None):
    Z, k, c, T, pt, qt, wo, y = ctx
    dst = scratch if Pn is P and qt.shape[1] < len(Q) and scratch is not None else Pn
    if dst is not Pn and rows is not None:
        dst[:] = Pn
    admix_cy.em(Z, P, dst, Q, T, k, c, pt, qt, rows, wo, pool, prior)
    if dst is not Pn:
        Pn[:] = dst
    if qo is None:
        admix_cy.accelQ(Q, Qn, T, len(Z) if rows is None else len(rows))
    else:
        admix_cy.accelQMiss(Q, Qn, T, qo)
    if y is not None:
        admix_cy.superQ(Qn, y)


### Two EM updates followed by quasi-Newton extrapolation
def emQuasi(P, Q, P1, P2, Q1, Q2, ctx, rows=None, qo=None, pool=None, prior=0.0, scratch=None):
    emStep(P, Q, P1, Q1, ctx, rows, qo, pool, prior, scratch)
    emStep(P if P1 is None else P1, Q1, P2, Q2, ctx, rows, qo, pool, prior, scratch)
    if P1 is not None:
        if rows is None:
            admix_cy.jumpP(P, P1, P2, ctx[1], ctx[2], Q.shape[1])
        else:
            admix_cy.jumpBatchP(P, P1, P2, ctx[1], ctx[2], rows, Q.shape[1])
    admix_cy.jumpQ(Q, Q1, Q2)
    if ctx[-1] is not None:
        admix_cy.superQ(Q, ctx[-1])


##### Initialization


### Centered label products for SVD/ALS initialization, without dosage expansion
def centerSVD(Z, p_vec, c_vec, W, K, chunk, power, rng, obs=None, extra=0):
    from hapla import struct, struct_cy

    N, base, M = Z.shape[1] // 2, K - 1, int(c_vec[W])
    D = min(base + extra, N - 1, M)
    L = min(max(D + 10, 20), N - 1, M)
    if base > L:
        raise ValueError("K exceeds the centered SVD dimensions. Use --random-init")
    data = [
        (Z[:W], c_vec[: W + 1].astype(np.int64), None if obs is None else obs[:W].astype(np.int64))
    ]
    p = p_vec[:M].astype(float)
    a = np.ones(M)
    Q, _ = np.linalg.qr(struct.product(data, p, a, L, chunk, rng=rng), mode="reduced")
    shift = 0.0
    for _ in range(power):
        H = struct.product(data, p, a, L, chunk, Q=np.ascontiguousarray(Q))
        H -= shift * Q
        Q, R = np.linalg.qr(H, mode="reduced")
        low = np.linalg.svd(R, compute_uv=False)[-1]
        if low > shift:
            shift = 0.5 * (low + shift)
    Q = np.ascontiguousarray(Q)
    A = np.empty((M, L))
    sums = Q.sum(axis=0)
    for z, c, s, o in struct.blocks(data, chunk):
        struct_cy.leftProduct(z, c, p[s], a[s], Q, sums, A[s], o)
    U, S, R = np.linalg.svd(A, full_matrices=False)
    rank = np.count_nonzero(S > np.finfo(float).eps * max(M, N) * S[0])
    if rank < base:
        raise ValueError("Insufficient variation for SVD initialization. Use --random-init")
    D = min(D, rank)
    return (
        np.ascontiguousarray(U[:, :D], dtype=np.float32),
        S[:D].astype(np.float32),
        np.ascontiguousarray(Q @ R[:D].T, dtype=np.float32),
    )


### One alternating least squares update for P and Q
def _alsStep(Y, V, p_vec, k_vec, c_vec, Q, P=None):
    H = np.dot(Q, np.linalg.pinv(np.dot(Q.T, Q)))
    P = np.dot(Y, np.dot(V.T, H), out=P)
    P *= 0.5
    P += np.outer(p_vec, np.sum(H, axis=0))
    admix_cy.projectP(P, k_vec, c_vec)
    H = np.dot(P, np.linalg.pinv(np.dot(P.T, P)))
    Q = 0.5 * np.dot(V, np.dot(Y.T, H))
    H *= p_vec[:, None]
    Q += H.sum(axis=0)
    admix_cy.projectQ(Q)
    return P, Q


### Select supported extremes in sample scores and solve for ancestry coordinates
def _sourceQ(V, S, K):
    X = np.ascontiguousarray(V * S, dtype=float)
    N, D = X.shape
    h = min(8, max(1, N // (3 * K)))
    norm = np.einsum("ij,ij->i", X, X)
    mean = X.mean(axis=0)
    R = X - mean
    A = np.empty((K, D))
    raw = np.empty_like(A)
    B = np.empty((D, K - 1))
    used = np.zeros(N, dtype=bool)
    taken = np.zeros(N, dtype=bool)
    cache = {}
    rank = 0

    # Favor neighborhood centers over isolated extreme samples
    for j in range(K):
        score = np.einsum("ij,ij->i", R, R)
        score[used] = -np.inf
        n = min(48, max(16, 3 * K), N - j)
        cand = np.argpartition(score, -n)[-n:]
        best = -1.0
        for i in cand:
            if i not in cache:
                dist = norm + norm[i] - 2 * (X @ X[i])
                near = np.argpartition(dist, h - 1)[:h]

                # Copy the small neighborhood so the full partition array can be freed
                cache[i] = X[near].mean(axis=0), near.copy()
            center, near = cache[i]
            v = center - (mean if j == 0 else A[0])
            for b in range(rank):
                v -= (v @ B[:, b]) * B[:, b]
            value = (v @ v) * (1 - np.count_nonzero(taken[near]) / h)
            if value > best:
                best, idx, point, group = value, i, center, near
        A[j] = point
        raw[j] = X[idx]
        used[idx] = True
        taken[group] = True
        if j == 0:
            R = X - point
        elif j < K - 1:
            v = point - A[0]
            for b in range(rank):
                v -= (v @ B[:, b]) * B[:, b]
            length = np.linalg.norm(v)
            if length > 1e-6 * max(np.linalg.norm(point), 1.0):
                v /= length
                B[:, rank] = v
                rank += 1
                R -= (R @ v)[:, None] * v

    H = A[1:] - A[:1]
    s = np.linalg.svd(H, compute_uv=False)
    if s[-1] <= 1e-4 * s[0]:
        A = raw
        H = A[1:] - A[:1]
        s = np.linalg.svd(H, compute_uv=False)
        if s[-1] <= 1e-4 * s[0]:
            raise ValueError("Source anchors are poorly conditioned. Omit --source-init")
    H = np.linalg.pinv(H, rcond=1e-4)
    Q = np.empty((N, K), dtype=np.float32)
    Q[:, 1:] = (X - A[0]) @ H
    Q[:, 0] = 1 - Q[:, 1:].sum(axis=1)
    admix_cy.projectQ(Q)
    return Q


### Alternating least square (ALS) for initializing Q and P
def factorALS(U, S, V, p_vec, k_vec, c_vec, iter, tole, rng, K, source=False):
    M = U.shape[0]
    Y = np.ascontiguousarray(U * S)
    if source:
        P = np.empty((M, K), dtype=np.float32)
        Q = _sourceQ(V, S, K)
    else:
        P = rng.random(size=(M, K), dtype=np.float32)
        admix_cy.projectP(P, k_vec, c_vec)
        H = np.dot(P, np.linalg.pinv(np.dot(P.T, P)))
        Q = 0.5 * np.dot(V, np.dot(Y.T, H))
        H *= p_vec[:, None]
        Q += H.sum(axis=0)
        admix_cy.projectQ(Q)
    Q0 = np.copy(Q)

    # Perform ALS iterations
    for _ in range(iter):
        P, Q = _alsStep(Y, V, p_vec, k_vec, c_vec, Q, P)

        # Check convergence
        if admix_cy.rmseQ(Q, Q0) < tole:
            break
        Q0[:] = Q
    return P, Q


### Project the remaining centered labels onto the initialization basis
def centerSub(Z, S, V, p_vec, c_vec, W_sub, chunk, obs=None):
    from hapla import struct, struct_cy

    beg = int(c_vec[W_sub])
    p = p_vec[beg:].astype(float)
    a = np.ones(len(p))
    Q = np.ascontiguousarray(V / S, dtype=float)
    sums = Q.sum(axis=0)
    U = np.empty((len(p), Q.shape[1]), dtype=np.float32)
    data = [
        (
            Z[W_sub:],
            (c_vec[W_sub:] - beg).astype(np.int64),
            None if obs is None else obs[W_sub:].astype(np.int64),
        )
    ]
    for z, c, s, o in struct.blocks(data, chunk):
        A = np.empty((int(c[-1]), Q.shape[1]))
        struct_cy.leftProduct(z, c, p[s], a[s], Q, sums, A, o)
        U[s] = A
    return U


### Least square (ALS) for subsampled P and Q followed by standard iteration
def factorSub(U_sub, U_rem, S, V, p_vec, k_vec, c_vec, W_sub, iter, tole, rng, K, source=False):
    # Fit the same ALS updates on the selected windows
    M = U_sub.shape[0]
    _, Q = factorALS(
        U_sub, S, V, p_vec[:M], k_vec[:W_sub], c_vec[:W_sub], iter, tole, rng, K, source
    )

    # Perform extra full ALS iteration
    Y = np.ascontiguousarray(np.concatenate((U_sub, U_rem), axis=0) * S)
    P, Q = _alsStep(Y, V, p_vec, k_vec, c_vec, Q)
    return P, Q
