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
    admix_cy.accelQ(Q, Qn, T, len(Z) if rows is None else len(rows), qo)
    if y is not None:
        admix_cy.superQ(Qn, y)


### Two EM updates followed by quasi-Newton extrapolation
def emQuasi(P, Q, P1, P2, Q1, Q2, ctx, rows=None, qo=None, pool=None, prior=0.0, scratch=None):
    emStep(P, Q, P1, Q1, ctx, rows, qo, pool, prior, scratch)
    emStep(P if P1 is None else P1, Q1, P2, Q2, ctx, rows, qo, pool, prior, scratch)
    if P1 is not None:
        admix_cy.jumpP(P, P1, P2, ctx[1], ctx[2], Q.shape[1], rows)
    admix_cy.jumpQ(Q, Q1, Q2)
    if ctx[-1] is not None:
        admix_cy.superQ(Q, ctx[-1])


### One full-data Q correction, leaving the shared P update unchanged
def looStep(P, Q, Pn, Qn, ctx, pool, prior, qo):
    Z, k, c, T, pt, qt, wo, y = ctx
    admix_cy.loo(Z, P, Pn, Q, T, k, c, pt, qt, pool, wo, prior)
    admix_cy.accelQ(Q, Qn, T, len(Z), qo)
    if y is not None:
        admix_cy.superQ(Qn, y)


##### Initialization


### Centered label products for SVD/ALS initialization, without dosage expansion
def centerSVD(Z, p_vec, c_vec, W, K, chunk, power, rng, obs=None):
    from hapla import struct, struct_cy

    N, D, M = Z.shape[1] // 2, K - 1, int(c_vec[W])
    L = min(max(D + 10, 20), N - 1, M)
    if D > L:
        raise ValueError("K exceeds the centered SVD dimensions. Use --random-init")
    c = np.ascontiguousarray(c_vec[: W + 1], dtype=np.int64)
    o = None if obs is None else np.ascontiguousarray(obs[:W], dtype=np.int64)
    data = [(Z[:W], c, o)]
    p = np.ascontiguousarray(p_vec[:M], dtype=float)
    a = np.ones(M)
    Q = struct.subspace(data, p, a, L, chunk, power, rng)
    A = np.empty((M, L))
    sums = Q.sum(axis=0)
    for z, c, s, o in struct.blocks(data, chunk):
        struct_cy.leftProduct(z, c, p[s], a[s], Q, sums, A[s], o)
    U, S, R = np.linalg.svd(A, full_matrices=False)
    rank = np.count_nonzero(S > np.finfo(float).eps * max(M, N) * S[0])
    if rank < D:
        raise ValueError("Insufficient variation for SVD initialization. Use --random-init")
    return (
        np.ascontiguousarray(U[:, :D], dtype=np.float32),
        S[:D].astype(np.float32),
        np.ascontiguousarray(Q @ R[:D].T, dtype=np.float32),
    )


### Update Q from P, or both parameters after initialization
def alsStep(Y, V, p_vec, k_vec, c_vec, Q, P=None):
    if Q is not None:
        H = np.dot(Q, np.linalg.pinv(np.dot(Q.T, Q)))
        P = np.dot(Y, np.dot(V.T, H), out=P)
        admix_cy.projectP(P, k_vec, c_vec, p_vec, np.sum(H, axis=0))
    H = np.dot(P, np.linalg.pinv(np.dot(P.T, P)))
    Q = np.dot(V, np.dot(Y.T, H))
    Q *= 0.5
    H *= p_vec[:, None]
    Q += H.sum(axis=0)
    admix_cy.projectQ(Q)
    return P, Q


### Alternating least square (ALS) for initializing Q and P
def factorALS(U, S, V, p_vec, k_vec, c_vec, iter, tole, rng):
    M, K = U.shape[0], len(S) + 1
    Y = np.ascontiguousarray(U * S)
    P = rng.random(size=(M, K), dtype=np.float32)
    admix_cy.projectP(P, k_vec, c_vec)
    P, Q = alsStep(Y, V, p_vec, k_vec, c_vec, None, P)
    Q0 = np.copy(Q)

    # Perform ALS iterations
    for _ in range(iter):
        P, Q = alsStep(Y, V, p_vec, k_vec, c_vec, Q, P)

        # Check convergence
        if admix_cy.rmseQ(Q, Q0) < tole:
            break
        Q0[:] = Q
    return P, Q


### Project the remaining centered labels onto the initialization basis
def centerSub(Z, S, V, p_vec, c_vec, W_sub, chunk, obs=None):
    from hapla import struct, struct_cy

    beg = int(c_vec[W_sub])
    p = np.ascontiguousarray(p_vec[beg:], dtype=float)
    a = np.ones(len(p))
    Q = np.ascontiguousarray(V / S, dtype=float)
    sums = Q.sum(axis=0)
    U = np.empty((len(p), Q.shape[1]), dtype=np.float32)
    c = np.ascontiguousarray(c_vec[W_sub:] - beg, dtype=np.int64)
    o = None if obs is None else np.ascontiguousarray(obs[W_sub:], dtype=np.int64)
    data = [(Z[W_sub:], c, o)]
    buf = np.empty((min(len(p), max(chunk, 255)), Q.shape[1]))
    for z, c, s, o in struct.blocks(data, chunk):
        A = buf[: c[-1]]
        struct_cy.leftProduct(z, c, p[s], a[s], Q, sums, A, o)
        U[s] = A
    return U
