"""Admixture updates and initialization from centered cluster labels."""

import numpy as np

from hapla import admix_cy

##### Admixture updates


### One EM update, sharing scratch across full, batch, and projection modes
def emStep(P, Q, Pn, Qn, ctx, rows=None, qo=None):
    Z, k, c, T, pt, qt, wo, y = ctx
    admix_cy.em(Z, P, Pn, Q, T, k, c, pt, qt, rows, wo)
    if qo is None:
        admix_cy.accelQ(Q, Qn, T, len(Z) if rows is None else len(rows))
    else:
        admix_cy.accelQMiss(Q, Qn, T, qo)
    if y is not None:
        admix_cy.superQ(Qn, y)


### Two EM updates followed by quasi-Newton extrapolation
def emQuasi(P, Q, P1, P2, Q1, Q2, ctx, rows=None, qo=None):
    emStep(P, Q, P1, Q1, ctx, rows, qo)
    emStep(P if P1 is None else P1, Q1, P2, Q2, ctx, rows, qo)
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
def centerSVD(Z, p_vec, c_vec, W, K, chunk, power, rng, obs=None):
    from hapla import struct, struct_cy

    N, D, M = Z.shape[1] // 2, K - 1, int(c_vec[W])
    L = min(max(D + 10, 20), N - 1, M)
    if D > L:
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
    if S[D - 1] <= np.finfo(float).eps * max(M, N) * S[0]:
        raise ValueError("Insufficient variation for SVD initialization. Use --random-init")
    return (
        np.ascontiguousarray(U[:, :D], dtype=np.float32),
        S[:D].astype(np.float32),
        np.ascontiguousarray(Q @ R[:D].T, dtype=np.float32),
    )


### Alternating least square (ALS) for initializing Q and P
def factorALS(U, S, V, p_vec, k_vec, c_vec, iter, tole, rng):
    M, D = U.shape
    Y = np.ascontiguousarray(U * S)
    P = rng.random(size=(M, D + 1), dtype=np.float32)
    admix_cy.projectP(P, k_vec, c_vec)
    H = np.dot(P, np.linalg.pinv(np.dot(P.T, P)))
    Q = 0.5 * np.dot(V, np.dot(Y.T, H))
    H *= p_vec[:, None]
    Q += H.sum(axis=0)
    admix_cy.projectQ(Q)
    Q0 = np.copy(Q)

    # Perform ALS iterations
    for _ in range(iter):
        # Update P
        H = np.dot(Q, np.linalg.pinv(np.dot(Q.T, Q)))
        np.dot(Y, np.dot(V.T, H), out=P)
        P *= 0.5
        P += np.outer(p_vec, np.sum(H, axis=0))
        admix_cy.projectP(P, k_vec, c_vec)

        # Update Q
        H = np.dot(P, np.linalg.pinv(np.dot(P.T, P)))
        Q = 0.5 * np.dot(V, np.dot(Y.T, H))
        H *= p_vec[:, None]
        Q += H.sum(axis=0)
        admix_cy.projectQ(Q)

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
def factorSub(U_sub, U_rem, S, V, p_vec, k_vec, c_vec, W_sub, iter, tole, rng):
    # Fit the same ALS updates on the selected windows
    M = U_sub.shape[0]
    _, Q = factorALS(U_sub, S, V, p_vec[:M], k_vec[:W_sub], c_vec[:W_sub], iter, tole, rng)

    # Perform extra full ALS iteration
    Y = np.ascontiguousarray(np.concatenate((U_sub, U_rem), axis=0) * S)
    H = np.dot(Q, np.linalg.pinv(np.dot(Q.T, Q)))
    P = np.dot(Y, np.dot(V.T, H))
    P *= 0.5
    P += np.outer(p_vec, np.sum(H, axis=0))
    admix_cy.projectP(P, k_vec, c_vec)
    H = np.dot(P, np.linalg.pinv(np.dot(P.T, P)))
    Q = 0.5 * np.dot(V, np.dot(Y.T, H))
    H *= p_vec[:, None]
    Q += H.sum(axis=0)
    admix_cy.projectQ(Q)
    return P, Q
