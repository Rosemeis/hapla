# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
from cython.parallel import prange
import numpy as np
from libc.math cimport fmax, fmaxf, fmin, fminf, log, sqrtf
from libc.stdint cimport uint8_t, uint32_t

ctypedef uint8_t u8
ctypedef uint32_t u32
ctypedef float f32
ctypedef double f64

cdef f64 PRO_MIN = 1e-5
cdef f64 ACC_MIN = 1.0
cdef f64 ACC_MAX = 100.0
cdef f32 FLT_MIN = 1e-5
cdef f32 FLT_MAX = 1.0 - (1e-5)


##### Admixture updates

### Bound the quasi-Newton step
cdef inline f64 _clamp2(f64 a) noexcept nogil: return fmax(ACC_MIN, fmin(a, ACC_MAX))


cdef inline f32 _clamp3(f32 a) noexcept nogil: return fmaxf(FLT_MIN, fminf(a, FLT_MAX))


### Exact count M-step on the simplex with a fixed probability floor
cdef inline void _simplex(f64* x, Py_ssize_t n, Py_ssize_t stride) noexcept nogil:
    cdef Py_ssize_t j, active = 0, removed
    cdef f64 total = 0.0, scale, value
    if n == 0:
        return
    for j in range(n):
        value = fmax(0.0, x[j*stride])
        x[j*stride] = value
        total += value
        active += value > 0.0
    if total == 0.0:
        for j in range(n):
            x[j*stride] = 1.0 / n
        return
    while True:
        scale = (1.0 - (n-active)*PRO_MIN) / total
        removed = 0
        for j in range(n):
            value = x[j*stride]
            if value > 0.0 and value*scale < PRO_MIN:
                total -= value
                x[j*stride] = 0.0
                removed += 1
        if removed == 0:
            break
        active -= removed
    for j in range(n):
        x[j*stride] = x[j*stride]*scale if x[j*stride] > 0.0 else PRO_MIN


### Estimate inverse individual allele frequency
cdef inline f64 _computeH(
        const f64* p,
        const f64* q,
        const Py_ssize_t K
    ) noexcept nogil:
    cdef:
        size_t k
        f64 h = 0.0
    for k in range(K):
        h += p[k] * q[k]
    return 1.0 / h


### Estimate individual allele frequency
cdef inline f64 _computeL(
        const f64* p,
        const f64* q,
        const Py_ssize_t K
    ) noexcept nogil:
    cdef:
        size_t k
        f64 h = 0.0
    for k in range(K):
        h += p[k] * q[k]
    return log(h)


### Outer loop accelerated update for Q
cdef inline void _outerAccelQ(
        const f64* q,
        f64* q_new,
        f64* q_tmp,
        const f64 S,
        const Py_ssize_t K
    ) noexcept nogil:
    cdef:
        size_t k
        f64 a, b
    for k in range(K):
        a = q[k] * q_tmp[k] * S
        b = a
        q_new[k] = b
        q_tmp[k] = 0.0
    _simplex(q_new, K, 1)


### Accumulate one fixed partition of the quasi-Newton norm
cdef void _norm(const f64* a, const f64* b, const f64* c,
                Py_ssize_t beg, Py_ssize_t end, f64* x, f64* y) noexcept nogil:
    cdef Py_ssize_t i
    cdef f64 u, v, s = 0.0, t = 0.0
    for i in range(beg, end):
        u = b[i] - a[i]
        v = c[i] - b[i] - u
        s += u*u
        t += u*v
    x[0], y[0] = s, t


### Estimate the QN factor in a fixed reduction order
cdef f64 _qnC(const f64* a, const f64* b, const f64* c, Py_ssize_t I) noexcept nogil:
    cdef Py_ssize_t j, B = min(64, I)
    cdef f64 x[64]
    cdef f64 y[64]
    cdef f64 s = 0.0, t = 0.0
    for j in prange(B, schedule='static'):
        _norm(a, b, c, j*I//B, (j+1)*I//B, &x[j], &y[j])
    for j in range(B):
        s += x[j]
        t += y[j]
    return _clamp2(-s/t) if s > 0 and t < 0 else 1.0


### Accumulate selected windows within one fixed norm partition
cdef void _batchNorm(const f64* a, const f64* b, const f64* c,
                     const u32* counts, const u32* off, const u32* rows,
                     Py_ssize_t beg, Py_ssize_t end, Py_ssize_t K,
                     f64* x, f64* y) noexcept nogil:
    cdef Py_ssize_t j, w
    cdef f64 u, v, s = 0.0, t = 0.0
    for j in range(beg, end):
        w = rows[j]
        _norm(a, b, c, off[w], off[w] + counts[w]*K, &u, &v)
        s += u
        t += v
    x[0], y[0] = s, t


### Estimate the batch QN factor independently of thread scheduling
cdef f64 _qnBatch(const f64* a, const f64* b, const f64* c,
                   const u32* counts, const u32* off, const u32* rows,
                   Py_ssize_t W, Py_ssize_t K) noexcept nogil:
    cdef Py_ssize_t j, B = min(64, W)
    cdef f64 x[64]
    cdef f64 y[64]
    cdef f64 s = 0.0, t = 0.0
    for j in prange(B, schedule='static'):
        _batchNorm(a, b, c, counts, off, rows, j*W//B, (j+1)*W//B, K, &x[j], &y[j])
    for j in range(B):
        s += x[j]
        t += y[j]
    return _clamp2(-s/t) if s > 0 and t < 0 else 1.0


### Estimate QN jump in P without temporary allocation
cdef inline void _computeP(f64* P0, const f64* P1, const f64* P2,
                           f64 c1, f64 c2, Py_ssize_t B, Py_ssize_t K) noexcept nogil:
    cdef Py_ssize_t c, k
    if B == 0:
        return
    for c in range(B*K):
        P0[c] = fmax(0.0, c2 * P1[c] + c1 * P2[c])
    for k in range(K):
        _simplex(&P0[k], B, K)


### Estimate QN jump in Q
cdef inline void _computeQ(
        f64* q0,
        const f64* q1,
        const f64* q2,
        const f64 c1,
        const f64 c2,
        const Py_ssize_t K
    ) noexcept nogil:
    cdef:
        size_t k
        f64 a, b
    for k in range(K):
        a = c2 * q1[k] + c1 * q2[k]
        b = fmax(0.0, a)
        q0[k] = b
    _simplex(q0, K, 1)


### Project P to domain
cdef inline void _projectP(
        f32* p,
        const Py_ssize_t B,
        const Py_ssize_t K
    ) noexcept nogil:
    cdef Py_ssize_t c, k
    cdef f32 total
    for k in range(K):
        total = 0.0
        for c in range(B):
            p[c*K+k] = _clamp3(p[c*K+k])
            total += p[c*K+k]
        for c in range(B):
            p[c*K+k] /= total


### Project Q to domain
cdef inline void _projectQ(
        f32* q,
        const Py_ssize_t K
    ) noexcept nogil:
    cdef:
        size_t k
        f32 sumQ = 0.0
        f32 a, b
    for k in range(K):
        a = q[k]
        b = _clamp3(a)
        sumQ += b
        q[k] = b
    for k in range(K):
        q[k] /= sumQ


### Compute the squared difference
cdef inline f32 _computeR(
        const f32* a,
        const f32* b,
        const Py_ssize_t I
    ) noexcept nogil:
    cdef:
        f32 r = 0.0
        f32 c
        size_t i
    for i in range(I):
        c = a[i] - b[i]
        r += c * c
    return r


### Accumulate a diploid sample and reuse the denominator for matching labels
cdef inline void _pair(const f64* p, const f64* q, f64* pt, f64* qt,
                       Py_ssize_t r, Py_ssize_t s, Py_ssize_t K,
                       bint first, bint second) noexcept nogil:
    cdef Py_ssize_t k
    cdef f64 a = 0.0, b = 0.0
    if first and second and r == s:
        a = 2.0 * _computeH(&p[r], q, K)
        if pt != NULL:
            for k in range(K):
                pt[r+k] += q[k] * a
                qt[k] += p[r+k] * a
        else:
            for k in range(K):
                qt[k] += p[r+k] * a
        return
    if first:
        a = _computeH(&p[r], q, K)
    if second:
        b = a if first and r == s else _computeH(&p[s], q, K)
    if first:
        if pt != NULL:
            for k in range(K):
                pt[r+k] += q[k] * a
                qt[k] += p[r+k] * a
        else:
            for k in range(K):
                qt[k] += p[r+k] * a
    if second:
        if pt != NULL:
            for k in range(K):
                pt[s+k] += q[k] * b
                qt[k] += p[s+k] * b
        else:
            for k in range(K):
                qt[k] += p[s+k] * b


### Compile constant-width loops for K=5 and K=6, plus a generic fallback
ctypedef fused width:
    u8
    u32
    f64


### Fit one window with reusable counts and no heap allocation
cdef inline void _emWindow(width mode, const u8* z, bint missing, const f64* p, f64* out,
                           const f64* q, f64* pt, f64* qt,
                           Py_ssize_t N, Py_ssize_t C, Py_ssize_t K) noexcept nogil:
    cdef Py_ssize_t i, k, c
    if width is u8:
        K = 5
    elif width is u32:
        K = 6
    if out != NULL:
        for c in range(C*K):
            pt[c] = 0.0
    if not missing:
        for i in range(N):
            _pair(p, &q[i*K], pt if out != NULL else NULL, &qt[i*K],
                  z[2*i]*K, z[2*i+1]*K, K, True, True)
    else:
        for i in range(N):
            _pair(p, &q[i*K], pt if out != NULL else NULL, &qt[i*K],
                  z[2*i]*K, z[2*i+1]*K, K, z[2*i] != 255, z[2*i+1] != 255)
    if out != NULL:
        for c in range(C*K):
            out[c] = p[c] * pt[c]
        for k in range(K):
            _simplex(&out[k], C, K)


### Fixed window partitions make the Q reduction independent of thread count
cpdef void em(const u8[:, ::1] Z, const f64[::1] P, f64[::1] P_new,
              const f64[:, ::1] Q, f64[:, ::1] T,
              const u32[::1] k_vec, const u32[::1] c_vec,
              f64[:, ::1] pt, f64[:, :, ::1] qt,
              const u32[::1] rows=None, const u32[::1] obs=None) noexcept nogil:
    cdef:
        Py_ssize_t W = Z.shape[0] if rows is None else rows.shape[0]
        Py_ssize_t N = Q.shape[0], K = Q.shape[1], B = min(W, qt.shape[0])
        Py_ssize_t b, t, w, i, k, l
        bint missing
        f64* out
    for b in prange(B, schedule='dynamic', chunksize=1):
        for i in range(N):
            for k in range(K):
                qt[b, i, k] = 0.0
        for t in range(b*W//B, (b+1)*W//B):
            w = t if rows is None else rows[t]
            l = c_vec[w]
            if k_vec[w] == 0 or (obs is not None and obs[w] == 0):
                if P_new is not None:
                    for k in range(k_vec[w]*K):
                        P_new[l+k] = P[l+k]
                continue
            missing = obs is not None and obs[w] != 2*N
            out = NULL if P_new is None else &P_new[l]
            if K == 5:
                _emWindow[u8](0, &Z[w, 0], missing, &P[l], out, &Q[0, 0], &pt[b, 0],
                              &qt[b, 0, 0], N, k_vec[w], 5)
            elif K == 6:
                _emWindow[u32](0, &Z[w, 0], missing, &P[l], out, &Q[0, 0], &pt[b, 0],
                               &qt[b, 0, 0], N, k_vec[w], 6)
            else:
                _emWindow[f64](0, &Z[w, 0], missing, &P[l], out, &Q[0, 0], &pt[b, 0],
                               &qt[b, 0, 0], N, k_vec[w], K)
    for t in prange((N+31)//32, schedule='static'):
        for i in range(t*32, min(N, (t+1)*32)):
            for k in range(K):
                T[i, k] = 0.0
        for b in range(B):
            for i in range(t*32, min(N, (t+1)*32)):
                for k in range(K):
                    T[i, k] += qt[b, i, k]


### Count observed haplotypes in sample tiles without a window-by-sample mask
def observedCounts(const u8[:, ::1] Z, const u32[::1] rows=None):
    cdef Py_ssize_t N = Z.shape[1]//2, W = Z.shape[0] if rows is None else rows.shape[0]
    cdef Py_ssize_t b, t, w, i, end
    cdef u32[::1] out = np.zeros(N, np.uint32)
    for b in prange((N+255)//256, nogil=True, schedule='static', use_threads_if=N*W >= 1048576):
        end = min(N, (b+1)*256)
        for t in range(W):
            w = t if rows is None else rows[t]
            for i in range(b*256, end):
                out[i] += (Z[w, 2*i] != 255) + (Z[w, 2*i+1] != 255)
    return np.asarray(out)


### Validate mapped labels without allocating chromosome-sized boolean arrays
cpdef Py_ssize_t validateLabels(const u8[:, ::1] Z, const u32[::1] k_vec):
    cdef Py_ssize_t w, i, count = 0
    cdef int bad = 0
    if Z.shape[0] != k_vec.shape[0] or Z.shape[1] == 0 or Z.shape[1] % 2:
        raise ValueError("Invalid assignment dimensions")
    for w in range(k_vec.shape[0]):
        if not 0 <= k_vec[w] <= 255:
            raise ValueError("Admixture requires 0..255 clusters per window")
    for w in prange(Z.shape[0], nogil=True, schedule='static'):
        for i in range(Z.shape[1]):
            if Z[w, i] == 255:
                count += 1
            elif Z[w, i] >= k_vec[w]:
                bad |= 1
    if bad:
        raise ValueError("Observed assignment is outside its window's cluster range")
    return count


### Accelerated jump for P (QN)
cpdef void jumpP(
        f64[::1] P0,
        f64[::1] P1,
        f64[::1] P2,
        const u32[::1] k_vec,
        const u32[::1] c_vec,
        const Py_ssize_t K
    ) noexcept nogil:
    cdef:
        Py_ssize_t M = P0.shape[0]
        Py_ssize_t W = k_vec.shape[0]
        Py_ssize_t B
        size_t l, w
        f64 c1, c2
    c1 = _qnC(&P0[0], &P1[0], &P2[0], M)
    c2 = 1.0 - c1
    for w in prange(W, schedule='guided'):
        l = c_vec[w]
        B = k_vec[w]
        _computeP(&P0[l], &P1[l], &P2[l], c1, c2, B, K)


### Batch accelerated jump for P (QN)
cpdef void jumpBatchP(
        f64[::1] P0,
        const f64[::1] P1,
        const f64[::1] P2,
        const u32[::1] k_vec,
        const u32[::1] c_vec,
        const u32[::1] s_bat,
        const Py_ssize_t K
    ) noexcept nogil:
    cdef:
        Py_ssize_t W = s_bat.shape[0]
        Py_ssize_t B
        size_t l, r, w
        f64 c1, c2
    c1 = _qnBatch(&P0[0], &P1[0], &P2[0], &k_vec[0], &c_vec[0], &s_bat[0], W, K)
    c2 = 1.0 - c1
    for w in prange(W, schedule='guided'):
        r = s_bat[w]
        l = c_vec[r]
        B = k_vec[r]
        _computeP(&P0[l], &P1[l], &P2[l], c1, c2, B, K)


### Accelerated update Q
cpdef void accelQ(
        const f64[:, ::1] Q,
        f64[:, ::1] Q_new,
        f64[:, ::1] Q_tmp,
        const Py_ssize_t W
    ) noexcept nogil:
    cdef:
        Py_ssize_t N = Q.shape[0]
        Py_ssize_t K = Q.shape[1]
        size_t i, k
        f64 S = 1.0 / <f64>(W << 1)
    for i in prange(N, schedule='guided'):
        _outerAccelQ(&Q[i, 0], &Q_new[i, 0], &Q_tmp[i, 0], S, K)


### Accelerated update Q with observed assignment counts
cpdef void accelQMiss(
        const f64[:, ::1] Q,
        f64[:, ::1] Q_new,
        f64[:, ::1] Q_tmp,
        const u32[::1] q_obs
    ) noexcept nogil:
    cdef:
        Py_ssize_t N = Q.shape[0]
        Py_ssize_t K = Q.shape[1]
        size_t i, k
        f64 S
    for i in prange(N, schedule='guided'):
        if q_obs[i] > 0:
            S = 1.0 / <f64>q_obs[i]
            _outerAccelQ(&Q[i, 0], &Q_new[i, 0], &Q_tmp[i, 0], S, K)
        else:
            for k in range(K):
                Q_new[i, k] = Q[i, k]
                Q_tmp[i, k] = 0.0


### Accelerated jump for Q (QN)
cpdef void jumpQ(
        f64[:, ::1] Q0,
        const f64[:, ::1] Q1,
        const f64[:, ::1] Q2
    ) noexcept nogil:
    cdef:
        Py_ssize_t N = Q0.shape[0]
        Py_ssize_t K = Q0.shape[1]
        size_t i, k
        f64 a, c1, c2
    c1 = _qnC(&Q0[0, 0], &Q1[0, 0], &Q2[0, 0], N * K)
    c2 = 1.0 - c1
    for i in prange(N, schedule='guided'):
        _computeQ(&Q0[i, 0], &Q1[i, 0], &Q2[i, 0], c1, c2, K)


##### Initialization and scoring

### Normalize P and Q once after initialization
cpdef void createP(f64[::1] P, const u32[::1] k_vec, const u32[::1] c_vec,
                   Py_ssize_t K) noexcept nogil:
    cdef Py_ssize_t w, k
    for w in prange(k_vec.shape[0], schedule='static'):
        if k_vec[w]:
            for k in range(K):
                _simplex(&P[c_vec[w]+k], k_vec[w], K)


### Normalize each individual's initial ancestry proportions
cpdef void createQ(f64[:, ::1] Q) noexcept nogil:
    cdef Py_ssize_t i
    for i in prange(Q.shape[0], schedule='static'):
        _simplex(&Q[i, 0], Q.shape[1], 1)


### Score one window in a fixed order, reusing homozygous probabilities
cdef f64 _likelihood(width mode, const u8* z, const f64* p, const f64* q,
                      Py_ssize_t N, Py_ssize_t K, bint missing) noexcept nogil:
    cdef Py_ssize_t i, r, s
    cdef f64 value, total = 0.0
    if width is u8:
        K = 5
    elif width is u32:
        K = 6
    for i in range(N):
        r, s = z[2*i]*K, z[2*i+1]*K
        if not missing or z[2*i] != 255:
            value = _computeL(&p[r], &q[i*K], K)
            total += value
            if r == s and (not missing or z[2*i+1] != 255):
                total += value
                continue
        if not missing or z[2*i+1] != 255:
            total += _computeL(&p[s], &q[i*K], K)
    return total


### Reuse one scalar per window for deterministic likelihood reduction
cpdef f64 likelihood(const u8[:, ::1] Z, const f64[::1] P,
                      const f64[:, ::1] Q, const u32[::1] c_vec,
                      f64[::1] work, const u32[::1] obs=None) noexcept nogil:
    cdef Py_ssize_t w, N = Q.shape[0], K = Q.shape[1]
    cdef f64 total = 0.0
    for w in prange(Z.shape[0], schedule='static'):
        if c_vec[w+1] == c_vec[w] or (obs is not None and obs[w] == 0):
            work[w] = 0.0
        elif K == 5:
            work[w] = _likelihood[u8](0, &Z[w, 0], &P[c_vec[w]], &Q[0, 0], N, 5,
                                      obs is not None and obs[w] != 2*N)
        elif K == 6:
            work[w] = _likelihood[u32](0, &Z[w, 0], &P[c_vec[w]], &Q[0, 0], N, 6,
                                       obs is not None and obs[w] != 2*N)
        else:
            work[w] = _likelihood[f64](0, &Z[w, 0], &P[c_vec[w]], &Q[0, 0], N, K,
                                       obs is not None and obs[w] != 2*N)
    for w in range(Z.shape[0]):
        total += work[w]
    return total * (<f64>K / (<f64>P.shape[0] * (2*N)))


### Projection function for P (f32)
cpdef void projectP(
        f32[:, ::1] P,
        const u32[::1] k_vec,
        const u32[::1] c_vec
    ) noexcept nogil:
    cdef:
        Py_ssize_t W = k_vec.shape[0]
        Py_ssize_t K = P.shape[1]
        Py_ssize_t B
        size_t l, w
    for w in prange(W, schedule='guided'):
        l = c_vec[w]
        B = k_vec[w]
        _projectP(&P[l, 0], B, K)


### Projection function for Q (f32)
cpdef void projectQ(
        f32[:, ::1] Q
    ) noexcept nogil:
    cdef:
        Py_ssize_t N = Q.shape[0]
        Py_ssize_t K = Q.shape[1]
        size_t i
    for i in prange(N, schedule='guided'):
        _projectQ(&Q[i, 0], K)


### Root-mean square error between two Q matrices
cpdef f32 rmseQ(
        f32[:, ::1] A,
        f32[:, ::1] B
    ) noexcept nogil:
    cdef:
        Py_ssize_t N = A.shape[0]
        Py_ssize_t K = A.shape[1]
        f32 r
    r = _computeR(&A[0, 0], &B[0, 0], N * K)
    return sqrtf(r / (<f32>(N) * <f32>(K)))


##### Supervised mode

### Initialize supervised frequencies from observed labels
cpdef void superP(const u8[:, ::1] Z, f64[:, ::1] P,
                  const u32[::1] k_vec, const u32[::1] c_vec,
                  const u8[::1] y) noexcept nogil:
    cdef Py_ssize_t w, h, k, l, K = P.shape[1]
    for w in prange(Z.shape[0], schedule='static'):
        l = c_vec[w]
        if k_vec[w] == 0:
            continue
        for h in range(Z.shape[1]):
            if y[h//2] > 0 and Z[w, h] != 255:
                P[l+Z[w, h], y[h//2]-1] += 1.0
        for k in range(K):
            _simplex(&P[l, k], k_vec[w], K)


### Update Q in supervised mode
cpdef void superQ(
        f64[:, ::1] Q,
        const u8[::1] y
    ) noexcept nogil:
    cdef:
        Py_ssize_t K = Q.shape[1]
        Py_ssize_t N = Q.shape[0]
        size_t i, k
    for i in prange(N, schedule='guided'):
        if y[i] > 0:
            for k in range(K):
                Q[i, k] = 1.0 - (K-1)*PRO_MIN if k == y[i]-1 else PRO_MIN


##### Projection mode

### Check haplotype cluster frequencies input
cpdef void checkP(
        f64[::1] P,
        f64[:, ::1] p_sum,
        const u32[::1] k_vec,
        const u32[::1] c_vec,
        const Py_ssize_t K
    ) noexcept nogil:
    cdef:
        Py_ssize_t W = k_vec.shape[0]
        Py_ssize_t B
        f64* p
        size_t c, k, l, w
    for w in prange(W, schedule='guided'):
        l = c_vec[w]
        B = k_vec[w]
        for c in range(B):
            p = &P[l + c * K]
            for k in range(K):
                p_sum[w, k] += p[k]
