# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False
# cython: cdivision=True
"""Direct products of validated cluster labels, without expanded dosage matrices."""

from cython.parallel cimport prange, threadid
cimport openmp as omp
import numpy as np
from libc.math cimport sqrt
from libc.stdint cimport uint8_t, uint64_t, int64_t

ctypedef uint8_t u8
ctypedef uint64_t u64
ctypedef int64_t i64
ctypedef float f32
ctypedef double f64


### Validate in a linear scan and build a histogram only when frequencies are needed
cdef int _frequency(const u8* z, Py_ssize_t H, Py_ssize_t K, f64* p, i64* obs) noexcept nogil:
    cdef Py_ssize_t h, k, total = 0
    cdef u64 cnt[256]
    cdef u8 limit = K
    cdef int bad = 0
    if p == NULL:
        for h in range(H):
            total += z[h] != 255
            bad |= (z[h] != 255) & (z[h] >= limit)
        obs[0] = total
        return bad
    for k in range(256):
        cnt[k] = 0
    for h in range(H):
        cnt[z[h]] += 1
    for k in range(K, 255):
        if cnt[k]:
            bad = 1
    obs[0] = H - cnt[255]
    for k in range(K):
        p[k] = <f64>cnt[k] / obs[0] if obs[0] else 0.0
    return bad


### Validate labels while counting each window once
cpdef void frequencies(const u8[:, ::1] Z, const i64[::1] c, f64[::1] p,
                       i64[::1] obs):
    cdef Py_ssize_t W = Z.shape[0], H = Z.shape[1], w
    cdef int bad = 0
    if H == 0 or H % 2 or c.shape[0] != W + 1 or c[0] != 0 or obs.shape[0] != W:
        raise ValueError("Invalid assignment or frequency dimensions")
    if p is not None and c[W] != p.shape[0]:
        raise ValueError("Invalid frequency dimensions")
    for w in range(W):
        if not 0 <= c[w+1] - c[w] <= 255:
            raise ValueError("Population structure requires 0..255 clusters per window")
    for w in prange(W, nogil=True, schedule='static'):
        bad |= _frequency(&Z[w, 0], H, c[w+1]-c[w],
                          &p[c[w]] if p is not None and c[w+1] > c[w] else NULL, &obs[w])
    if bad:
        raise ValueError("Observed assignment is outside its window's cluster range")


### Sum nonnegative squared dosage deviations without subtracting large moments
cdef void _variation(const u8* z, const f64* p, f64* out,
                     Py_ssize_t N, Py_ssize_t K) noexcept nogil:
    cdef u64 hom[255]
    cdef u64 het[255]
    cdef u64 one[255]
    cdef Py_ssize_t i, k, r, s, full = 0, partial = 0
    cdef f64 q, x
    for k in range(K): hom[k], het[k], one[k] = 0, 0, 0
    for i in range(N):
        r, s = z[2*i], z[2*i+1]
        if r != 255 and s != 255:
            full += 1
            if r == s: hom[r] += 1
            else:
                het[r] += 1
                het[s] += 1
        elif r != 255 or s != 255:
            partial += 1
            one[r if r != 255 else s] += 1
    for k in range(K):
        q, x = p[k], 1-2*p[k]
        out[k] = (4*hom[k]*(1-q)*(1-q) + het[k]*x*x
                  + 4*(full-hom[k]-het[k])*q*q
                  + one[k]*(1-q)*(1-q) + (partial-one[k])*q*q)


### Measure dosage variation independently for each window
cpdef void variation(const u8[:, ::1] Z, const i64[::1] c, const f64[::1] p, f64[::1] out):
    cdef Py_ssize_t W = Z.shape[0], N = Z.shape[1]//2, w
    if Z.shape[1] % 2 or N < 1 or c.shape[0] != W+1 or c[0] != 0 or c[W] != p.shape[0]:
        raise ValueError("Invalid dosage variation dimensions")
    if out.shape[0] != p.shape[0]:
        raise ValueError("Invalid dosage variation output dimensions")
    for w in prange(W, nogil=True, schedule='static'):
        if c[w+1] > c[w]:
            _variation(&Z[w, 0], &p[c[w]], &out[c[w]], N, c[w+1]-c[w])


### Compute X Q by accumulating two nonzero dosages per sample
cpdef void leftProduct(const u8[:, ::1] Z, const i64[::1] c,
                       const f64[::1] p, const f64[::1] a,
                       const f64[:, ::1] Q, const f64[::1] sums,
                       f64[:, ::1] A, const i64[::1] obs=None):
    cdef:
        Py_ssize_t w, i, l, k, r, s, W = Z.shape[0], N = Q.shape[0], L = Q.shape[1]
        Py_ssize_t t, nt = max(1, min(W, omp.omp_get_max_threads()))
        f64 u
        f64[:, ::1] work = np.empty((nt, L)) if obs is not None else None
    for w in prange(W, nogil=True, schedule='static', num_threads=nt):
        for k in range(c[w], c[w+1]):
            for l in range(L):
                A[k, l] = 0.0
        if obs is None or obs[w] == 2*N:
            for i in range(N):
                r, s = c[w] + Z[w, 2*i], c[w] + Z[w, 2*i+1]
                for l in range(L):
                    A[r, l] += Q[i, l]
                    A[s, l] += Q[i, l]
        else:
            t = threadid()
            for l in range(L):
                work[t, l] = 2.0*sums[l] if obs[w] else 0.0
            if obs[w] == 0:
                continue
            for i in range(N):
                r, s = Z[w, 2*i], Z[w, 2*i+1]
                if r != 255 and s != 255:
                    for l in range(L):
                        A[c[w]+r, l] += Q[i, l]
                        A[c[w]+s, l] += Q[i, l]
                else:
                    for l in range(L):
                        if r == 255:
                            work[t, l] -= Q[i, l]
                        else:
                            A[c[w]+r, l] += Q[i, l]
                        if s == 255:
                            work[t, l] -= Q[i, l]
                        else:
                            A[c[w]+s, l] += Q[i, l]
        for k in range(c[w], c[w+1]):
            for l in range(L):
                u = 2.0*sums[l] if obs is None or obs[w] == 2*N else work[t, l]
                A[k, l] = (A[k, l] - p[k] * u) * a[k]


### Add D' A in independent sample blocks with row scaling already in A
cpdef void rightProduct(const u8[:, ::1] Z, const i64[::1] c,
                        const f64[:, ::1] A, f64[:, ::1] Q,
                        const f64[::1] p, const i64[::1] obs=None):
    cdef:
        Py_ssize_t b, w, i, l, k, r, s, end, W = Z.shape[0], N = Q.shape[0], L = Q.shape[1]
        f64[:, ::1] mean = np.zeros((W, L)) if obs is not None else None
    if obs is not None:
        for w in prange(W, nogil=True, schedule='static'):
            if obs[w] < 2*N:
                for k in range(c[w], c[w+1]):
                    for l in range(L):
                        mean[w, l] += p[k] * A[k, l]
    for b in prange((N + 31) // 32, nogil=True, schedule='static'):
        end = min(N, (b+1)*32)
        for w in range(W):
            if obs is None or obs[w] == 2*N:
                for i in range(b*32, end):
                    r, s = c[w] + Z[w, 2*i], c[w] + Z[w, 2*i+1]
                    for l in range(L):
                        Q[i, l] += A[r, l] + A[s, l]
            else:
                for i in range(b*32, end):
                    r, s = Z[w, 2*i], Z[w, 2*i+1]
                    for l in range(L):
                        Q[i, l] += (A[c[w]+r, l] if r != 255 else mean[w, l]) + (A[c[w]+s, l] if s != 255 else mean[w, l])


### Center and scale the packed GRM without allocating a square matrix
cpdef f64 normalizeGram(f64[::1] G, f64 den, f64[::1] u, bint center):
    cdef:
        Py_ssize_t N = u.shape[0], i, j, off
        f64 val, trace = 0.0, mean = 0.0, scale = 1.0
    with nogil:
        for i in range(N):
            off = i*(i+1)//2
            for j in range(i+1):
                val = G[off+j] / (2.0*den)
                G[off+j] = val
                u[i] += val
                if j != i:
                    u[j] += val
            trace += G[off+i]
        if center:
            for i in range(N):
                u[i] /= N
                mean += u[i] / N
            trace -= N * mean
    if trace <= 0:
        raise ValueError("Population structure requires positive empirical dosage variation")
    if center:
        scale = (N-1) / trace
        for i in prange(N, nogil=True, schedule='static'):
            off = i*(i+1)//2
            for j in range(i+1):
                G[off+j] = (G[off+j] - u[i] - u[j] + mean) * scale
    return scale


### Use K-1 orthonormal Helmert contrasts for each centered categorical window
cdef void _contrasts(const u8* z, const f64* p, f32[:, ::1] X,
                     Py_ssize_t K, Py_ssize_t off) noexcept nogil:
    cdef Py_ssize_t N = X.shape[1], j, i
    cdef f64 d, u, total = 0.0
    cdef int a, b
    for j in range(K-1):
        total += p[j]
        d = 1.0 / sqrt(<f64>(j+1)*(j+2))
        u = (total - (j+1)*p[j+1]) * d
        for i in range(N):
            a, b = z[2*i], z[2*i+1]
            X[off+j, i] = ((a <= j) + (b <= j) - (j+1)*((a == j+1) + (b == j+1))) * d - ((a != 255) + (b != 255)) * u


### Expand one bounded block and omit GRM rows for fixed windows
cpdef void contrastBlock(const u8[:, ::1] Z, const i64[::1] c,
                         const f64[::1] p, const i64[::1] rows, f32[:, ::1] X) noexcept nogil:
    cdef Py_ssize_t w
    for w in prange(Z.shape[0], schedule='static'):
        if c[w+1] > c[w]:
            _contrasts(&Z[w, 0], &p[c[w]], X, c[w+1]-c[w], rows[w])


### Accumulate only the lower triangle, in double precision
cpdef void addGram(const f32[:, ::1] A, f64[::1] G, Py_ssize_t start) noexcept nogil:
    cdef Py_ssize_t N = A.shape[0], i, j, off
    for i in prange(N, schedule='static'):
        off = (start+i)*(start+i+1)//2
        for j in range(start+i+1):
            G[off+j] += A[i, j]
