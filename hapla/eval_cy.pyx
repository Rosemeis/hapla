# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False
# cython: cdivision=True
"""Direct label projections, residual variances, and bounded correlation output."""

__author__ = "Jonas Meisner"

import numpy as np
cimport openmp as omp
from cython.parallel cimport prange
from libc.math cimport sqrt
from libc.stdint cimport uint8_t, uint64_t, int64_t
from libc.stdio cimport FILE, fdopen, fclose, fwrite, snprintf
from libc.string cimport memcpy

cdef extern from "unistd.h" nogil:
    int dup(int)
    int close(int)

ctypedef uint8_t u8
ctypedef uint64_t u64
ctypedef int64_t i64
ctypedef double f64


### Project the two observed cluster labels directly onto a small fitted basis
def project(const u8[:, ::1] Z, const i64[::1] c, const f64[:, ::1] U):
    cdef Py_ssize_t w, i, k, a, b, N = Z.shape[1]//2, K = U.shape[1]
    cdef f64[:, ::1] A = np.zeros((c[Z.shape[0]], K))
    for w in prange(Z.shape[0], nogil=True, schedule='static',
                    num_threads=max(1, min(Z.shape[0], omp.omp_get_max_threads())),
                    use_threads_if=Z.shape[0]*N*K >= 262144):
        for i in range(N):
            a, b = Z[w, 2*i], Z[w, 2*i+1]
            for k in range(K):
                if a != 255: A[c[w]+a, k] += U[i, k]
                if b != 255: A[c[w]+b, k] += U[i, k]
    return np.asarray(A)


### Replace fitted means with residuals in sample tiles and weight windows by 1/K
def residuals(f64[:, ::1] R, const u8[:, ::1] Z, const i64[::1] c):
    cdef Py_ssize_t i, w, a, b, end, K, N = R.shape[1]
    cdef f64 h, p, weight
    cdef f64[::1] v = np.zeros(N)
    for b in prange((N+63)//64, nogil=True, schedule='static',
                    num_threads=max(1, min((N+63)//64, omp.omp_get_max_threads())),
                    use_threads_if=R.shape[0]*N >= 262144):
        end = min(N, (b+1)*64)
        for w in range(Z.shape[0]):
            K = c[w+1]-c[w]
            if K == 0: continue
            weight = 1.0/sqrt(K)
            for a in range(c[w], c[w+1]):
                for i in range(b*64, end):
                    h = (Z[w, 2*i] != 255) + (Z[w, 2*i+1] != 255)
                    p = R[a, i]
                    if h > 0 and 0 < p < h:
                        v[i] += p*(1-p/h)/K
                    R[a, i] = ((Z[w, 2*i] == a-c[w]) + (Z[w, 2*i+1] == a-c[w])-p)*weight
    return np.asarray(v)


### Convert each symmetric covariance entry once using saved standard deviations
cpdef void correlation(f64[:, ::1] C, const f64[::1] d) noexcept nogil:
    cdef Py_ssize_t i, j
    cdef f64 val
    for i in prange(C.shape[0], schedule='static'):
        for j in range(i, C.shape[0]):
            val = C[i, j]/(d[i]*d[j]) if d[i] > 0 and d[j] > 0 else 0
            C[i, j] = val
            C[j, i] = val


### Round bounded correlations exactly with integer arithmetic for decimal halfway cases
cdef int _fixed(f64 val, char* buf) noexcept nogil:
    cdef u64 bits = 0, mant, rem, half
    cdef int shift, q = 0, n = 0
    if not -2 <= val <= 2:
        return snprintf(buf, 32, "%.4f", val)
    memcpy(&bits, &val, sizeof(f64))
    if bits >> 63:
        buf[0] = 45
        n = 1

    # IEEE binary64: abs(val)*10000 = mantissa*625 / 2**(1071-exponent), within uint64.
    shift = 1071 - <int>((bits >> 52) & 2047)
    if shift < 64:
        mant = ((bits & ((<u64>1 << 52)-1)) | (<u64>1 << 52))*625
        q = <int>(mant >> shift)
        rem, half = mant & ((<u64>1 << shift)-1), <u64>1 << (shift-1)
        if rem > half or (rem == half and (q & 1)):
            q += 1
    buf[n] = 48+q//10000
    buf[n+1] = 46
    buf[n+2] = 48+(q//1000)%10
    buf[n+3] = 48+(q//100)%10
    buf[n+4] = 48+(q//10)%10
    buf[n+5] = 48+q%10
    return n+6


### Format independent rows in parallel, retaining the exact four-decimal output
cdef int _row(const f64* a, const f64* b, char* buf, Py_ssize_t N,
              i64* size) noexcept nogil:
    cdef Py_ssize_t j, n = 0
    cdef int digits
    for j in range(N):
        digits = _fixed(a[j] if b == NULL else a[j]-b[j], buf+n)
        if digits < 0 or digits >= 32: return 1
        n += digits
        buf[n] = 10 if j+1 == N else 32
        n += 1
    size[0] = n
    return 0


### Stream bounded text blocks with optional subtraction to avoid another square matrix
def writeMatrix(int fd, const f64[:, ::1] A, const f64[:, ::1] B=None):
    cdef Py_ssize_t M = A.shape[0], N = A.shape[1], start, i, row, end
    cdef Py_ssize_t width = 32*N+1, batch = max(1, min(128, 8*1024**2//width))
    cdef u8[:, ::1] buf = np.empty((batch, width), np.uint8)
    cdef i64[::1] size = np.empty(batch, np.int64)
    cdef int copy, bad = 0
    cdef FILE* out
    if N < 1:
        raise ValueError("Residual output requires nonempty rows")
    if B is not None and (B.shape[0] != M or B.shape[1] != N):
        raise ValueError("Correlation output dimensions differ")
    copy = dup(fd)
    if copy < 0: raise OSError("Cannot duplicate residual output handle")
    out = fdopen(copy, "w")
    if out == NULL:
        close(copy)
        raise OSError("Cannot open residual output stream")
    with nogil:
        start = 0
        while start < M:
            end = min(batch, M-start)
            for i in prange(end, schedule='static', num_threads=min(end, omp.omp_get_max_threads()),
                            use_threads_if=end*N >= 65536):
                row = start+i
                bad |= _row(&A[row, 0], &B[row, 0] if B is not None else NULL,
                            <char*>&buf[i, 0], N, &size[i])
            if bad: break
            for i in range(end):
                if fwrite(&buf[i, 0], 1, size[i], out) != <size_t>size[i]:
                    bad = 1
                    break
            if bad: break
            start += end
        if fclose(out) != 0: bad = 1
    if bad: raise OSError("Failed to write residual correlations")
