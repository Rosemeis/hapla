# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
"""Linear-time ancestry HMMs, bounded scratch, and independent haplotype fits."""

import numpy as np
cimport openmp as omp
from cython.parallel cimport prange, threadid
from libc.math cimport exp, expm1, log, log1p, INFINITY, isfinite, fabs
from libc.stdio cimport FILE, fdopen, fclose, fwrite, snprintf
from posix.unistd cimport dup, close
from libc.stdint cimport uint8_t, uint32_t, int64_t

ctypedef uint8_t u8
ctypedef uint32_t u32
ctypedef int64_t i64
ctypedef double f64
ctypedef float f32


### Add log probabilities, including exact zeros
cdef inline f64 _add(f64 a, f64 b) noexcept nogil:
    if a == -INFINITY: return b
    if b == -INFINITY: return a
    if a < b: a, b = b, a
    return a + log1p(exp(b-a))


### Sum log probabilities after subtracting their maximum
cdef f64 _sum(const f64* x, Py_ssize_t K) noexcept nogil:
    cdef Py_ssize_t k
    cdef f64 m = x[0], s = 0
    for k in range(1, K): m = max(m, x[k])
    if m == -INFINITY: return m
    for k in range(K): s += exp(x[k]-m)
    return m + log(s)


### Sum all other states with prefix and suffix passes
cdef void _exclude(const f64* x, f64* out, Py_ssize_t K) noexcept nogil:
    cdef Py_ssize_t k
    cdef f64 s = -INFINITY
    for k in range(K):
        out[k] = s
        s = _add(s, x[k])
    s = -INFINITY
    for k in range(K-1, -1, -1):
        out[k] = _add(out[k], s)
        s = _add(s, x[k])


### Normalize file frequencies after checking each ancestral simplex
def normalizeP(f64[:, ::1] P, const i64[::1] c):
    cdef Py_ssize_t W = c.shape[0]-1, K = P.shape[1], w, k, a
    cdef f64 total
    cdef int bad = 0
    if W < 1 or c[0] != 0 or c[W] != P.shape[0] or K < 1:
        raise ValueError("Invalid frequency dimensions")
    for w in prange(W, nogil=True, schedule='static'):
        if c[w+1] > c[w]:
            for k in range(K):
                total = 0
                for a in range(c[w], c[w+1]): total = total + P[a, k]
                if not isfinite(total) or fabs(total-1) > 1e-5:
                    bad |= 1
                else:
                    for a in range(c[w], c[w+1]): P[a, k] /= total
    if bad: raise ValueError("P must sum to one within each window and ancestry")


### Precompute each cluster's log emission once, including stable soft mixtures
cdef int _table(const f64* p, const f32* likes, f64* out,
                Py_ssize_t C, Py_ssize_t K) noexcept nogil:
    cdef Py_ssize_t a, b, k
    cdef f64 weights[255]
    cdef f64 m, total, value, largest, term, sub
    for a in range(C):
        if likes != NULL:
            m = -INFINITY
            for b in range(C):
                value = likes[a*C+b]
                if value != value or value == INFINITY: return 1
                m = max(m, value)
            if m == -INFINITY: return 1
            total = 0
            for b in range(C):
                weights[b] = exp(likes[a*C+b]-m)
                total += weights[b]
        for k in range(K):
            if likes == NULL:
                value = p[a*K+k]
            else:
                value = 0
                for b in range(C): value += weights[b]*p[b*K+k]
                value /= total
            if likes != NULL and value < 1e-200:
                largest = -INFINITY
                for b in range(C):
                    if p[b*K+k] > 0:
                        term = likes[a*C+b]-m+log(p[b*K+k])
                        largest = max(largest, term)
                sub = 0
                if largest != -INFINITY:
                    for b in range(C):
                        if p[b*K+k] > 0:
                            sub += exp(likes[a*C+b]-m+log(p[b*K+k])-largest)
                out[a*K+k] = largest+log(sub)-log(total) if sub > 0 else -INFINITY
            else:
                out[a*K+k] = log(value) if value > 0 else -INFINITY
    return 0


### Build cluster emission tables in parallel across windows
def emissionTable(const f64[::1] P, const i64[::1] c, Py_ssize_t K,
                  const f32[::1] likes=None):
    cdef Py_ssize_t W = c.shape[0]-1, w
    cdef int bad = 0
    if W < 1 or not 1 <= K <= 255 or c[0] != 0 or c[W]*K != P.shape[0]:
        raise ValueError("Invalid emission table dimensions")
    counts = np.diff(c)
    if np.any((counts < 0) | (counts > 255)):
        raise ValueError("Emission tables require 0..255 clusters per window")
    cdef i64[::1] x = np.r_[0, np.cumsum(counts*counts, dtype=np.int64)]
    if likes is not None and likes.shape[0] != x[W]:
        raise ValueError("Median likelihood payload does not match the windows")
    cdef f64[::1] table = np.empty(P.shape[0])
    for w in prange(W, nogil=True, schedule='static'):
        if c[w+1] > c[w]:
            if likes is None:
                bad |= _table(&P[c[w]*K], NULL, &table[c[w]*K], c[w+1]-c[w], K)
            else:
                bad |= _table(&P[c[w]*K], &likes[x[w]], &table[c[w]*K], c[w+1]-c[w], K)
    if bad: raise ValueError("Each median likelihood row must have finite support and no NaN/+inf")
    return np.asarray(table)


### Gather window emissions and leave missing or excluded assignments neutral
def emissions(const u8[:, ::1] Z, const f64[::1] table, const i64[::1] c,
              const u8[::1] use, Py_ssize_t K, Py_ssize_t beg, Py_ssize_t end,
              Py_ssize_t block=1):
    cdef Py_ssize_t N = Z.shape[0], W = Z.shape[1], i, w, k, b, z
    if not 0 <= beg < end <= W or block < 1 or not 1 <= K <= 255:
        raise ValueError("Invalid HMM window bounds or block size")
    if c.shape[0] != W+1 or use.shape[0] != W or table.shape[0] != c[W]*K:
        raise ValueError("Emission metadata dimensions differ")
    cdef bint direct = block == 1 and N > 1 and omp.omp_get_max_threads() > 1
    shape = (N, (end-beg+block-1)//block, K)
    cdef f64[:, :, ::1] E = np.empty(shape) if direct else np.zeros(shape)
    if direct:
        # Fill in parallel. The zero-initialized loop is faster with one thread.
        for i in prange(N, nogil=True, schedule='static'):
            for w in range(beg, end):
                z = Z[i, w]
                if use[w] and z != 255:
                    b = (c[w]+z)*K
                    for k in range(K): E[i, w-beg, k] = table[b+k]
                else:
                    for k in range(K): E[i, w-beg, k] = 0
    else:
        for i in prange(N, nogil=True, schedule='static'):
            for w in range(beg, end):
                z = Z[i, w]
                if use[w] and z != 255:
                    b = (w-beg)//block
                    for k in range(K): E[i, b, k] += table[(c[w]+z)*K+k]
    return np.asarray(E)


### Check priors and alpha once before entering unchecked recursions
def _arguments(E, Q, alpha):
    if E.ndim != 3 or min(E.shape) < 1 or E.shape[2] > 255:
        raise ValueError("HMM emissions require positive dimensions and 1..255 ancestries")
    Q = np.asarray(Q, dtype=np.float64)
    if Q.shape != (E.shape[0], E.shape[2]) or not np.all(np.isfinite(Q)) or np.any(Q < 0):
        raise ValueError("Invalid HMM ancestry proportions")
    sums = Q.sum(axis=1)
    if not np.allclose(sums, 1, atol=1e-5, rtol=0):
        raise ValueError("HMM ancestry proportions must sum to one")
    alpha = np.atleast_1d(np.asarray(alpha, dtype=np.float64))
    if alpha.ndim != 1 or not len(alpha) or not np.all(np.isfinite(alpha)) or np.any(alpha <= 0):
        raise ValueError("HMM alpha values must be finite and positive")
    return np.ascontiguousarray(Q/sums[:, None]), np.ascontiguousarray(alpha)


### Rescale emissions once per haplotype and use log recursion for extreme ranges
cdef int _prepare(const f64* E, f64* S, f64* shift, const f64* q,
                  Py_ssize_t W, Py_ssize_t K, f64 amin) noexcept nogil:
    cdef Py_ssize_t w, k
    cdef f64 m, lo, value, span = 0, qm = 1
    for k in range(K):
        if q[k] > 0: qm = min(qm, q[k])
    for w in range(W):
        m, lo = -INFINITY, INFINITY
        for k in range(K):
            value = E[w*K+k]
            if value != value or value == INFINITY: return -1
            if q[k] > 0:
                m = max(m, value)
                if value != -INFINITY: lo = min(lo, value)
        if m == -INFINITY: return -1
        shift[w] = m
        span = max(span, m-lo)
        for k in range(K):
            S[w*K+k] = exp(E[w*K+k]-m) if q[k] > 0 else 0
    return log(qm) + log(-expm1(-amin)) - span < -300


### Scaled forward/backward: T = exp(-alpha) I + (1-exp(-alpha)) q 1'
cdef f64 _fb_prob(const f64* E, const f64* shift, f64* scale, f64* F, f64* work,
                   const f64* q, f64* G, f64* count, Py_ssize_t W,
                   Py_ssize_t K, f64 alpha, f64 weight) noexcept nogil:
    cdef Py_ssize_t w, k
    cdef f64 e = exp(-alpha), s = -expm1(-alpha), norm, total, ll = 0, value
    cdef f64* beta = work
    cdef f64* v = work+K
    for w in range(W):
        norm = 0
        for k in range(K):
            F[w*K+k] = E[w*K+k]*(q[k] if w == 0 else e*F[(w-1)*K+k]+s*q[k])
            norm += F[w*K+k]
        if norm <= 0: return -INFINITY
        ll += log(norm)+shift[w]
        if scale != NULL: scale[w] = norm
        for k in range(K): F[w*K+k] /= norm
    if G == NULL: return ll
    for k in range(K): beta[k] = 1
    for w in range(W-1, -1, -1):
        for k in range(K):
            value = F[w*K+k]*beta[k]
            G[w*K+k] += weight*value
            if count != NULL and w == 0: count[k] += weight*value
            v[k] = E[w*K+k]*beta[k]
        if w:
            total = 0
            for k in range(K): total += q[k]*v[k]
            norm = 1.0/scale[w]
            if count != NULL:
                for k in range(K):
                    count[k] += weight*s*q[k]*v[k]*norm
                    beta[k] = (e*v[k]+s*total)*norm
            else:
                for k in range(K): beta[k] = (e*v[k]+s*total)*norm
    return ll


### Logarithmic fallback and normalized simplified transitions, still O(W K)
cdef f64 _fb_log(const f64* E, f64* shift, f64* F, f64* work, const f64* q,
                  f64* G, f64* count, Py_ssize_t W, Py_ssize_t K,
                  f64 alpha, f64 weight, bint simple) noexcept nogil:
    cdef Py_ssize_t w, k
    cdef f64 ls = log(-expm1(-alpha)), norm, total, ll = 0, value
    cdef f64* beta = work
    cdef f64* v = work+K
    cdef f64* ex = work+2*K
    cdef f64* lq = work+3*K
    cdef f64* den = work+4*K
    for k in range(K):
        lq[k] = log(q[k]) if q[k] > 0 else -INFINITY
        den[k] = _add(-alpha, ls+log1p(-q[k])) if simple else 0
    for w in range(W):
        if simple and w:
            for k in range(K): v[k] = F[(w-1)*K+k]-den[k]
            _exclude(v, ex, K)
        for k in range(K):
            if w == 0:
                value = lq[k]
            elif simple:
                value = _add(-alpha+v[k], ls+lq[k]+ex[k])
            else:
                value = _add(-alpha+F[(w-1)*K+k], ls+lq[k])
            F[w*K+k] = E[w*K+k]+value
        norm = _sum(&F[w*K], K)
        if norm == -INFINITY: return norm
        ll += norm
        shift[w] = norm
        for k in range(K): F[w*K+k] -= norm
    if G == NULL: return ll
    for k in range(K): beta[k] = 0
    for w in range(W-1, -1, -1):
        for k in range(K): v[k] = F[w*K+k]+beta[k]
        norm = _sum(v, K)
        for k in range(K):
            value = exp(v[k]-norm)
            G[w*K+k] += weight*value
            if count != NULL and w == 0: count[k] += weight*value
        if w:
            if count != NULL:
                # Reuse the forward scale and posterior normalizer for reset counts.
                norm += shift[w]
                for k in range(K): count[k] += weight*exp(ls+lq[k]+E[w*K+k]+beta[k]-norm)
            for k in range(K): v[k] = lq[k]+E[w*K+k]+beta[k]
            if simple:
                _exclude(v, ex, K)
            else:
                total = _sum(v, K)
            for k in range(K):
                beta[k] = _add(-alpha+E[w*K+k]+beta[k], ls+(ex[k] if simple else total))-den[k]
            norm = _sum(beta, K)
            for k in range(K): beta[k] -= norm
    return ll


### Average ancestry posteriors and reset counts across alpha values
def posterior(const f64[:, :, ::1] E, Q, alpha, bint simple=False, bint resets=False, bint score=False):
    if score and resets:
        raise ValueError("Reset counts require posterior decoding")
    if simple and resets:
        raise ValueError("Q refinement requires the standard refresh transition model")
    Q, alpha = _arguments(np.asarray(E), Q, alpha)
    cdef const f64[:, ::1] q = Q
    cdef const f64[::1] rates = alpha
    cdef Py_ssize_t N = E.shape[0], W = E.shape[1], K = E.shape[2], A = len(alpha)
    cdef Py_ssize_t nt = min(N, omp.omp_get_max_threads()), i, a, t
    cdef int mode, bad = 0
    cdef f64 amin = min(alpha), value
    cdef f64* count
    cdef f64* g
    cdef f64[:, :, ::1] G = np.empty((0, 0, 0)) if score else np.zeros((N, W, K))
    cdef f64[:, ::1] C = np.zeros((N, K)) if resets else np.empty((0, 0))
    cdef f64[:, ::1] ll = np.empty((N, A))
    cdef f64[:, ::1] F = np.empty((nt, W*K))
    cdef f64[:, ::1] S = np.empty((nt, W*K))
    cdef f64[:, ::1] shift = np.empty((nt, W))
    cdef f64[:, ::1] scale = np.empty((nt, W)) if not score else np.empty((0, 0))
    cdef f64[:, ::1] tmp = np.empty((nt, 5*K))
    for i in prange(N, nogil=True, schedule='static', num_threads=nt):
        t = threadid()
        mode = _prepare(&E[i, 0, 0], &S[t, 0], &shift[t, 0], &q[i, 0], W, K, amin)
        if mode < 0:
            bad |= 1
            continue
        count = &C[i, 0] if resets else NULL
        g = NULL if score else &G[i, 0, 0]
        for a in range(A):
            if mode or simple:
                value = _fb_log(&E[i, 0, 0], &shift[t, 0], &F[t, 0], &tmp[t, 0], &q[i, 0],
                                g, count, W, K, rates[a], 1.0/A, simple)
            else:
                value = _fb_prob(&S[t, 0], &shift[t, 0], &scale[t, 0] if not score else NULL,
                                 &F[t, 0], &tmp[t, 0], &q[i, 0],
                                 g, count, W, K, rates[a], 1.0/A)
            ll[i, a] = value
            if not isfinite(value): bad |= 1
    if bad: raise ValueError("Nonfinite emissions or no supported HMM path for an observed haplotype")
    return np.asarray(G), np.asarray(C), np.asarray(ll)


### Decode the mean posterior from bounded per-thread scratch
def posteriorDecode(const f64[:, :, ::1] E, Q, alpha, bint simple=False, bint confidence=False):
    Q, alpha = _arguments(np.asarray(E), Q, alpha)
    cdef const f64[:, ::1] q = Q
    cdef const f64[::1] rates = alpha
    cdef Py_ssize_t N = E.shape[0], W = E.shape[1], K = E.shape[2], A = len(alpha)
    cdef Py_ssize_t nt = min(N, omp.omp_get_max_threads()), i, a, t, w, k, dst
    cdef int mode, bad = 0
    cdef f64 amin = min(alpha), value, best
    cdef u8[:, ::1] D = np.empty((N, W), np.uint8)
    cdef f64[:, ::1] P = np.empty((N, W)) if confidence else np.empty((0, 0))
    cdef f64[:, ::1] ll = np.empty((N, A))
    cdef f64[:, ::1] F = np.empty((nt, W*K))
    cdef f64[:, ::1] S = np.empty((nt, W*K))
    cdef f64[:, ::1] G = np.empty((nt, W*K))
    cdef f64[:, ::1] shift = np.empty((nt, W))
    cdef f64[:, ::1] scale = np.empty((nt, W))
    cdef f64[:, ::1] tmp = np.empty((nt, 5*K))
    for i in prange(N, nogil=True, schedule='static', num_threads=nt):
        t = threadid()
        mode = _prepare(&E[i, 0, 0], &S[t, 0], &shift[t, 0], &q[i, 0], W, K, amin)
        if mode < 0:
            bad |= 1
            continue
        for w in range(W*K): G[t, w] = 0
        for a in range(A):
            if mode or simple:
                value = _fb_log(&E[i, 0, 0], &shift[t, 0], &F[t, 0], &tmp[t, 0], &q[i, 0],
                                &G[t, 0], NULL, W, K, rates[a], 1.0/A, simple)
            else:
                value = _fb_prob(&S[t, 0], &shift[t, 0], &scale[t, 0], &F[t, 0],
                                 &tmp[t, 0], &q[i, 0], &G[t, 0], NULL, W, K, rates[a], 1.0/A)
            ll[i, a] = value
            if not isfinite(value): bad |= 1
        for w in range(W):
            dst = 0
            best = G[t, w*K]
            for k in range(1, K):
                value = G[t, w*K+k]
                if value > best:
                    best = value
                    dst = k
            D[i, w] = dst
            if confidence: P[i, w] = best
    if bad: raise ValueError("Nonfinite emissions or no supported HMM path for an observed haplotype")
    return np.asarray(D), np.asarray(P), np.asarray(ll)


### Small state spaces favor a tight dense loop with rolling score vectors
cdef f64 _viterbi_small(const f64* E, const f64* q, u8* path, u8* prev,
                         f64* work, Py_ssize_t W, Py_ssize_t K,
                         f64 alpha) noexcept nogil:
    cdef f64 T[64]
    cdef f64* a = work
    cdef f64* b = work+K
    cdef f64* swap
    cdef f64 ls = log(-expm1(-alpha)), best, value, norm
    cdef Py_ssize_t w, k, j, src
    for k in range(K):
        value = ls+log(q[k]) if q[k] > 0 else -INFINITY
        for j in range(K): T[k*K+j] = _add(-alpha, value) if k == j else value
        if E[k] != E[k] or E[k] == INFINITY: return -INFINITY
        a[k] = E[k]+(log(q[k]) if q[k] > 0 else -INFINITY)
    for w in range(1, W):
        norm = -INFINITY
        for k in range(K):
            src, best = 0, a[0]+T[k*K]
            for j in range(1, K):
                value = a[j]+T[k*K+j]
                if value > best: src, best = j, value
            value = E[w*K+k]
            if value != value or value == INFINITY: return -INFINITY
            b[k] = best+value
            prev[w*K+k] = src
            norm = max(norm, b[k])
        if norm == -INFINITY: return norm

        # Periodic centering avoids growth of accumulated scores. Huge blocks center immediately.
        if w % 64 == 0 or norm < -1e4 or norm > 1e4:
            for k in range(K): b[k] -= norm
        swap, a, b = a, b, a
    src, best = 0, a[0]
    for k in range(1, K):
        if a[k] > best: src, best = k, a[k]
    if best == -INFINITY: return best
    path[W-1] = src
    for w in range(W-2, -1, -1): path[w] = prev[(w+1)*K+path[w+1]]
    return best


### Viterbi needs only the two best predecessors and one traceback byte per state
cdef f64 _viterbi(const f64* E, const f64* q, u8* path, u8* prev,
                   f64* work, Py_ssize_t W, Py_ssize_t K, f64 alpha,
                   bint simple) noexcept nogil:
    cdef Py_ssize_t w, k, first, second, src, dst
    cdef f64 ls = log(-expm1(-alpha)), best, runner, stay, jump, norm, score = 0
    cdef f64* v = work
    cdef f64* u = work+K
    cdef f64* diag = work+2*K
    cdef f64* off = work+3*K
    cdef f64* den = work+4*K
    if K <= 8 and not simple:
        return _viterbi_small(E, q, path, prev, work, W, K, alpha)
    for k in range(K):
        off[k] = ls+log(q[k]) if q[k] > 0 else -INFINITY
        den[k] = _add(-alpha, ls+log1p(-q[k])) if simple else 0
        diag[k] = -alpha-den[k] if simple else _add(-alpha, off[k])
        if E[k] != E[k] or E[k] == INFINITY: return -INFINITY
        v[k] = E[k]+(log(q[k]) if q[k] > 0 else -INFINITY)
    for w in range(1, W):
        first, best = 0, v[0]
        if simple:
            first, second, best, runner = 0, 0, -INFINITY, -INFINITY
            for k in range(K):
                u[k] = v[k]-den[k]
                if u[k] > best:
                    second, runner, first, best = first, best, k, u[k]
                elif u[k] > runner:
                    second, runner = k, u[k]
        else:
            for k in range(1, K):
                if v[k] > best: first, best = k, v[k]
        norm = -INFINITY
        for k in range(K):
            src = second if simple and first == k else first
            jump = (runner if simple and first == k else best)+off[k]
            stay = v[k]+diag[k]
            dst = k
            if jump > stay or (jump == stay and src < k):
                stay, dst = jump, src
            prev[w*K+k] = dst
            if E[w*K+k] != E[w*K+k] or E[w*K+k] == INFINITY: return -INFINITY
            u[k] = stay+E[w*K+k]
            norm = max(norm, u[k])
        if norm == -INFINITY: return norm
        score += norm
        for k in range(K): v[k] = u[k]-norm
    dst, best = 0, v[0]
    for k in range(1, K):
        if v[k] > best: dst, best = k, v[k]
    if best == -INFINITY: return best
    path[W-1] = dst
    for w in range(W-2, -1, -1): path[w] = prev[(w+1)*K+path[w+1]]
    return score+best


### Decode exact paths per alpha and use plurality for an ensemble
def viterbi(const f64[:, :, ::1] E, Q, alpha, bint simple=False):
    Q, alpha = _arguments(np.asarray(E), Q, alpha)
    cdef const f64[:, ::1] q = Q
    cdef const f64[::1] rates = alpha
    cdef Py_ssize_t N = E.shape[0], W = E.shape[1], K = E.shape[2], A = len(alpha)
    cdef Py_ssize_t nt = min(N, omp.omp_get_max_threads()), i, a, t, w, k, dst
    cdef int bad = 0
    cdef f64 value
    cdef u8[:, ::1] D = np.empty((N, W), np.uint8)
    cdef u8[:, ::1] I = np.empty((nt, W*K), np.uint8)
    cdef u8[:, ::1] path = np.empty((nt, W), np.uint8)
    cdef u32[:, ::1] votes = np.zeros((nt, W*K), np.uint32) if A > 1 else np.empty((nt, 0), np.uint32)
    cdef f64[:, ::1] tmp = np.empty((nt, 5*K))
    for i in prange(N, nogil=True, schedule='static', num_threads=nt):
        t = threadid()
        for a in range(A):
            value = _viterbi(&E[i, 0, 0], &q[i, 0], &path[t, 0], &I[t, 0],
                             &tmp[t, 0], W, K, rates[a], simple)
            if not isfinite(value):
                bad |= 1
                continue
            for w in range(W):
                if A == 1: D[i, w] = path[t, w]
                else: votes[t, w*K+path[t, w]] += 1
        if A > 1:
            for w in range(W):
                dst = 0
                for k in range(1, K):
                    if votes[t, w*K+k] > votes[t, w*K+dst]: dst = k
                D[i, w] = dst
                for k in range(K): votes[t, w*K+k] = 0
    if bad: raise ValueError("No supported HMM path for an observed haplotype")
    return np.asarray(D)


### Count observed assignments once for convergence scaling and empty samples
def observations(const u8[:, ::1] Z, const u8[::1] use):
    cdef Py_ssize_t W = Z.shape[0], N = Z.shape[1] // 2, i, w, t
    cdef Py_ssize_t nt = max(1, min(N, omp.omp_get_max_threads()))
    if Z.shape[1] % 2 or use.shape[0] != W:
        raise ValueError("Invalid observation dimensions")
    cdef i64[::1] obs = np.zeros(N, np.int64)
    for t in prange(nt, nogil=True, schedule='static', num_threads=nt):
        for w in range(W):
            if use[w]:
                for i in range(N*t//nt, N*(t+1)//nt):
                    obs[i] += (Z[w, 2*i] != 255) + (Z[w, 2*i+1] != 255)
    return np.asarray(obs)


### Log prior relative to its mode: -mass * KL(base || current)
def penalty(const f64[::1] base, const f64[::1] p, f64 mass):
    cdef Py_ssize_t a
    cdef f64 value = 0
    if base.shape[0] != p.shape[0] or not isfinite(mass) or mass < 0:
        raise ValueError("Invalid prior dimensions or mass")
    if mass == 0: return 0.0
    with nogil:
        for a in range(base.shape[0]):
            if base[a] > 0:
                if p[a] <= 0:
                    value = -INFINITY
                    break
                value += base[a] * (log(p[a]) - log(base[a]))
    return mass * value


### Accumulate P sufficient statistics from bounded posterior batches
def accumulate(const u8[:, ::1] Z, const f64[:, :, ::1] G,
               const u8[::1] use, const i64[::1] c, Py_ssize_t beg,
               Py_ssize_t end, f64[::1] counts):
    cdef Py_ssize_t N = Z.shape[0], K = G.shape[2], w, i, k, z
    if G.shape[0] != N or G.shape[1] != end-beg or counts.shape[0] != c[c.shape[0]-1]*K:
        raise ValueError("Refinement dimensions differ")
    for w in prange(beg, end, nogil=True, schedule='static'):
        if use[w]:
            for i in range(N):
                z = Z[i, w]
                if z != 255:
                    for k in range(K): counts[(c[w]+z)*K+k] += G[i, w-beg, k]


### Normalize emission counts with optional prior mass
def refineP(const f64[::1] base, const f64[::1] counts,
            const i64[::1] c, Py_ssize_t K, f64 mass):
    cdef Py_ssize_t W = c.shape[0]-1, w, k, a
    cdef f64 total
    cdef f64[::1] P = np.empty(base.shape[0])
    if counts.shape[0] != base.shape[0] or base.shape[0] != c[W]*K or not isfinite(mass) or mass < 0:
        raise ValueError("Invalid refinement counts or prior mass")
    for w in prange(W, nogil=True, schedule='static'):
        for k in range(K):
            total = 0
            for a in range(c[w], c[w+1]): total = total + counts[a*K+k]
            for a in range(c[w], c[w+1]):
                P[a*K+k] = ((counts[a*K+k] + mass*base[a*K+k])/(total + mass)
                            if total > 0 else base[a*K+k])
    return np.asarray(P)


### Correct reciprocal phase switches in pairs of haplotypes
def phaseCorrect(
        u8[:, ::1] D,
        f64[:, ::1] L,
        const u32 dist,
        bint probs
    ):
    cdef:
        Py_ssize_t N = D.shape[0] // 2
        Py_ssize_t W = D.shape[1]
        Py_ssize_t pending, pos
        Py_ssize_t i, j, w
        i64 n_fix = 0
        u8 p0, p1, x0, x1, old, new, pend0, pend1
        f64 tmp
        bint change0, change1, match, phase
    if D.shape[0] % 2 or W < 1:
        raise ValueError("Phase correction requires paired haplotypes and nonempty windows")
    if probs and (L.shape[0] != D.shape[0] or L.shape[1] != W):
        raise ValueError("Phase confidence and path dimensions differ")
    for i in prange(N, nogil=True, schedule='static'):
        j = 2 * i
        p0 = D[j, 0]
        p1 = D[j + 1, 0]
        pending = -1
        pos = 0
        phase = False
        for w in range(1, W):
            x0 = D[j, w]
            x1 = D[j + 1, w]
            change0 = p0 != x0
            change1 = p1 != x1
            match = False

            if change0 and change1:
                match = (p0 == x1) and (p1 == x0)
                pending = -1
            elif change0 != change1:
                if change0:
                    old = p0
                    new = x0
                    if (pending == 1) and (w - pos <= dist):
                        match = (pend0 == new) and (pend1 == old)
                else:
                    old = p1
                    new = x1
                    if (pending == 0) and (w - pos <= dist):
                        match = (pend0 == new) and (pend1 == old)

                if match:
                    pending = -1
                else:
                    pending = 0 if change0 else 1
                    pos = w
                    pend0 = old
                    pend1 = new
            elif (pending >= 0) and (w - pos > dist):
                pending = -1

            if match:
                phase = not phase
                n_fix += 1
            if phase:
                D[j, w] = x1
                D[j + 1, w] = x0
                if probs:
                    tmp = L[j, w]
                    L[j, w] = L[j + 1, w]
                    L[j + 1, w] = tmp
            p0 = x0
            p1 = x1
    return n_fix


### Format text output in one fixed buffer, without Python objects per value
def writeRows(int fd, const u8[:, ::1] D=None, const f64[:, ::1] P=None):
    cdef Py_ssize_t N, W, i, w, d, ndec
    cdef size_t n = 0, size = 1024*1024
    cdef int copy, value, digits, error = 0
    cdef unsigned long long rounded, place, digit, scale
    cdef f64 x, scaled
    cdef FILE* out
    cdef u8[::1] buffer = np.empty(size, np.uint8)
    if (D is None) == (P is None):
        raise ValueError("Provide either paths or posterior confidence for output")
    N, W = (D.shape[0], D.shape[1]) if D is not None else (P.shape[0], P.shape[1])
    copy = dup(fd)
    if copy < 0: raise OSError("Cannot duplicate local ancestry output handle")
    out = fdopen(copy, "a")
    if out == NULL:
        close(copy)
        raise OSError("Cannot open local ancestry output stream")
    with nogil:
        for i in range(N):
            for w in range(W):
                if n > size-64:
                    if fwrite(&buffer[0], 1, n, out) != n:
                        error = 1
                        break
                    n = 0
                if D is not None:
                    value = D[i, w]
                    if value >= 100:
                        buffer[n] = 48+value//100
                        n += 1
                    if value >= 10:
                        buffer[n] = 48+(value//10)%10
                        n += 1
                    buffer[n] = 48+value%10
                    n += 1
                else:
                    x = P[i, w]
                    if isfinite(x) and x >= 0.001 and x <= 1:
                        if x >= 0.1:
                            scale, place, ndec = 100000000, 10000000, 8
                        elif x >= 0.01:
                            scale, place, ndec = 1000000000, 100000000, 9
                        else:
                            scale, place, ndec = 10000000000, 1000000000, 10
                        scaled = x * scale
                        rounded = <unsigned long long>(scaled + 0.5)
                        if fabs(scaled - (<f64>rounded - 0.5)) >= 1e-7:
                            if rounded == scale:
                                buffer[n] = 49
                                n += 1
                            else:
                                buffer[n] = 48
                                buffer[n+1] = 46
                                n += 2
                                for d in range(ndec):
                                    digit = rounded // place
                                    buffer[n] = 48 + digit
                                    n += 1
                                    rounded -= digit*place
                                    place //= 10
                                while buffer[n-1] == 48: n -= 1
                                if buffer[n-1] == 46: n -= 1
                            buffer[n] = 10 if w+1 == W else 32
                            n += 1
                            continue
                    digits = snprintf(<char*>&buffer[n], 32, "%.8g", x)
                    if digits < 0 or digits >= 32:
                        error = 1
                        break
                    n += digits
                buffer[n] = 10 if w+1 == W else 32
                n += 1
            if error: break
        if n and not error and fwrite(&buffer[0], 1, n, out) != n: error = 1
        if fclose(out) != 0: error = 1
    if error: raise OSError("Failed to write local ancestry output")
