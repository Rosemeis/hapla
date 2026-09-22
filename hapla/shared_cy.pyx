# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
"""PLINK decoding and packed cluster-pair distances for unphased prediction."""

import numpy as np
cimport openmp as omp
from cython.parallel cimport prange, threadid
from libc.stdint cimport uint8_t, uint32_t, uint64_t

ctypedef uint8_t u8
ctypedef uint32_t u32
ctypedef uint64_t u64

cdef extern from *:
    """
    static inline unsigned hapla_popcount(unsigned long long value) {
        return (unsigned)__builtin_popcountll(value);
    }
    """
    unsigned hapla_popcount(unsigned long long) noexcept nogil


### Read unphased 2-bit genotype array for haplotype cluster prediction
cpdef void readPlink(const u8[:, ::1] D, u8[:, ::1] G) noexcept nogil:
    cdef:
        Py_ssize_t M = G.shape[0]
        Py_ssize_t N = G.shape[1]
        Py_ssize_t B = D.shape[1]
        size_t b, i, j, part
        u8[4] recode = [2, 9, 1, 0]
        u8 mask = 3
        u8 byte
    for j in prange(M, schedule='guided'):
        i = 0
        for b in range(B):
            byte = D[j, b]
            for part in range(4):
                G[j, i] = recode[byte & mask]
                byte = byte >> 2
                i = i + 1
                if i == N:
                    break


### Share homozygous errors across pairs and require opposite alleles for heterozygotes
cdef void _pair(const u8* x, const u64[:, ::1] R, u64* hom, u64* het,
                u64* mask, u8* out, Py_ssize_t M) noexcept nogil:
    cdef Py_ssize_t j, q, a, b, K = R.shape[0], Q = R.shape[1]
    cdef u32 errors[255]
    cdef u32 d
    cdef float best = M+1
    cdef u64 bit
    for q in range(Q): hom[q], het[q], mask[q] = 0, 0, 0
    for j in range(M):
        bit = <u64>1 << (j & 63)
        q = j >> 6
        if x[j] == 1:
            het[q] |= bit
        elif x[j] == 0 or x[j] == 2:
            mask[q] |= bit
            if x[j] == 2: hom[q] |= bit
    for a in range(K):
        errors[a] = 0
        for q in range(Q): errors[a] += hapla_popcount((hom[q] ^ R[a, q]) & mask[q])
    out[0], out[1] = 0, 0
    for a in range(K):
        for b in range(K):
            d = errors[b]
            for q in range(Q): d += hapla_popcount(het[q] & ~(R[a, q] ^ R[b, q]))

            # Preserve the existing weights, traversal order, and float32 tie behavior.
            if 0.66 * <float>errors[a] + 0.33 * <float>d < best:
                best = 0.66 * <float>errors[a] + 0.33 * <float>d
                out[0], out[1] = a, b


### Pack reference medians once and reuse bounded scratch for each sample
cpdef void genoCluster(const u8[:, ::1] X, const u8[:, ::1] R, u8[::1] Z):
    cdef Py_ssize_t N = X.shape[0], M = X.shape[1], K = R.shape[0]
    cdef Py_ssize_t Q = (M+63)//64, nt = min(max(1, N), omp.omp_get_max_threads())
    cdef Py_ssize_t i, k, j, t
    if M < 1 or R.shape[1] != M or not 1 <= K <= 255 or Z.shape[0] != 2*N:
        raise ValueError("Invalid unphased prediction dimensions")
    if K == 1:
        with nogil:
            for i in range(2*N): Z[i] = 0
        return
    cdef u64[:, ::1] packed = np.zeros((K, Q), np.uint64)
    cdef u64[:, :, ::1] work = np.empty((nt, 3, Q), np.uint64)
    with nogil:
        for k in range(K):
            for j in range(M): packed[k, j >> 6] |= <u64>R[k, j] << (j & 63)
        for i in prange(N, schedule='static', num_threads=nt, use_threads_if=N*K*K*Q >= 131072):
            t = threadid()
            _pair(&X[i, 0], packed, &work[t, 0, 0], &work[t, 1, 0],
                  &work[t, 2, 0], &Z[2*i], M)
