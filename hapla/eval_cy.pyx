# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
import numpy as np
cimport numpy as np
cimport openmp as omp
from cython.parallel import prange
from libc.math cimport sqrt
from libc.stdint cimport uint8_t, uint32_t

ctypedef uint8_t u8
ctypedef uint32_t u32
ctypedef float f32
ctypedef double f64

##### hapla - evaluation of admixture model fit #####
### Standard functions
# Add residual covariances from chromosome to accumulator
cpdef void covar(
        f64[:,::1] C, f64[::1] V, const f64[:,::1] Q, const f64[:,::1] A,
        const u8[:,::1] Z, const u32[::1] k_chr
    ):
    cdef:
        Py_ssize_t W = Z.shape[0]
        Py_ssize_t N = Q.shape[0]
        Py_ssize_t K = Q.shape[1]
        Py_ssize_t i,j,w,t,c,k,l,n
        int nthreads = omp.omp_get_max_threads()
        f64 norm = 0.0

    # thread-local
    cdef f64[:,:,:] C_priv = np.zeros((nthreads, N, N), dtype=np.float64)
    cdef f64[:,:] V_priv = np.zeros((nthreads, N), dtype=np.float64)
    cdef f64[:,:] R = np.zeros((nthreads, N), dtype=np.float64)
    cdef f64[:,:] XQ = np.zeros((nthreads, K), dtype=np.float64)
    cdef f64[:,:] B = np.zeros((nthreads, K), dtype=np.float64)

    with nogil:
        for w in prange(W, schedule='guided'):
            t = omp.omp_get_thread_num()
            norm = 1.0 / k_chr[w]

            for c in range(k_chr[w]):
                for k in range(K):
                    XQ[t,k] = 0.0
                    B[t,k] = 0.0

                # Regress the cluster count feature on Q.
                for n in range(N):
                    for k in range(K):
                        XQ[t,k] += (
                            (Z[w,2*n] == c) + (Z[w,2*n+1] == c)
                        ) * Q[n,k]

                for k in range(K):
                    for l in range(K):
                        B[t,k] += XQ[t,l] * A[l,k]

                # Compute residuals and binomial variance under the fitted mean.
                for n in range(N):
                    R[t,n] = 0.0
                    for k in range(K):
                        R[t,n] += Q[n,k] * B[t,k]
                    if R[t,n] > 0.0 and R[t,n] < 2.0:
                        V_priv[t,n] += R[t,n] * (1.0 - 0.5 * R[t,n]) * norm
                    R[t,n] = (Z[w,2*n] == c) + (Z[w,2*n+1] == c) - R[t,n]

                # Get covariances.
                for i in range(N):
                    for j in range(i, N):
                        C_priv[t,i,j] += R[t,i] * R[t,j] * norm

        for t in range(nthreads):
            for i in range(N):
                V[i] += V_priv[t,i]
                for j in range(i, N):
                    C[i,j] += C_priv[t,i,j]


# Add residual covariances from chromosome to accumulator, skipping missing haplotypes
cpdef void covarMiss(
        f64[:,::1] C, f64[::1] V, const f64[:,::1] Q, const f64[:,:,::1] A_chr,
        const u8[:,::1] Z, const u8[:,::1] Z_miss, const u32[::1] k_chr
    ):
    cdef:
        Py_ssize_t W = Z.shape[0]
        Py_ssize_t N = Q.shape[0]
        Py_ssize_t K = Q.shape[1]
        Py_ssize_t i,j,w,t,c,k,l,n
        int nthreads = omp.omp_get_max_threads()
        f64 norm = 0.0
        f64 obs = 0.0
        f64 x = 0.0
        f64 pred = 0.0
        f64 var = 0.0

    cdef f64[:,:,:] C_priv = np.zeros((nthreads, N, N), dtype=np.float64)
    cdef f64[:,:] V_priv = np.zeros((nthreads, N), dtype=np.float64)
    cdef f64[:,:] R = np.zeros((nthreads, N), dtype=np.float64)
    cdef f64[:,:] Var = np.zeros((nthreads, N), dtype=np.float64)
    cdef f64[:,:] Obs = np.zeros((nthreads, N), dtype=np.float64)
    cdef f64[:,:,:] D = np.zeros((nthreads, N, K), dtype=np.float64)
    cdef f64[:,:] XQ = np.zeros((nthreads, K), dtype=np.float64)
    cdef f64[:,:] B = np.zeros((nthreads, K), dtype=np.float64)

    with nogil:
        for w in prange(W, schedule='guided'):
            t = omp.omp_get_thread_num()
            norm = 1.0 / k_chr[w]

            for n in range(N):
                obs = <f64>((Z_miss[w,2*n] == 0) + (Z_miss[w,2*n+1] == 0))
                Obs[t,n] = obs
                for k in range(K):
                    D[t,n,k] = obs * Q[n,k]

            for c in range(k_chr[w]):
                for k in range(K):
                    XQ[t,k] = 0.0
                    B[t,k] = 0.0

                for n in range(N):
                    x = <f64>(
                        (Z_miss[w,2*n] == 0 and Z[w,2*n] == c) +
                        (Z_miss[w,2*n+1] == 0 and Z[w,2*n+1] == c)
                    )
                    for k in range(K):
                        XQ[t,k] += D[t,n,k] * x

                for k in range(K):
                    for l in range(K):
                        B[t,k] += XQ[t,l] * A_chr[w,l,k]

                for n in range(N):
                    pred = 0.0
                    for k in range(K):
                        pred = pred + D[t,n,k] * B[t,k]
                    var = 0.0
                    if Obs[t,n] > 0.0 and pred > 0.0 and pred < Obs[t,n]:
                        var = pred * (1.0 - pred / Obs[t,n])
                    Var[t,n] = var
                    V_priv[t,n] += var * norm
                    x = <f64>(
                        (Z_miss[w,2*n] == 0 and Z[w,2*n] == c) +
                        (Z_miss[w,2*n+1] == 0 and Z[w,2*n+1] == c)
                    )
                    R[t,n] = x - pred

                for i in range(N):
                    for j in range(i, N):
                        C_priv[t,i,j] += R[t,i] * R[t,j] * norm

        for t in range(nthreads):
            for i in range(N):
                V[i] += V_priv[t,i]
                for j in range(i, N):
                    C[i,j] += C_priv[t,i,j]


# Estimate expected residual covariances when samples have missing haplotypes
cpdef void expectedMiss(
        f64[:,::1] C, const f64[:,::1] Q, const f64[:,:,::1] A_chr,
        const u8[:,::1] Z, const u8[:,::1] Z_miss, const u32[::1] k_chr
    ):
    cdef:
        Py_ssize_t W = Z.shape[0]
        Py_ssize_t N = Q.shape[0]
        Py_ssize_t K = Q.shape[1]
        Py_ssize_t i,j,w,t,c,k,l,m,n
        int nthreads = omp.omp_get_max_threads()
        f64 norm = 0.0
        f64 obs = 0.0
        f64 x = 0.0
        f64 pred = 0.0
        f64 var = 0.0
        f64 hij = 0.0
        f64 eij = 0.0

    cdef f64[:,:,:] C_priv = np.zeros((nthreads, N, N), dtype=np.float64)
    cdef f64[:,:] Var = np.zeros((nthreads, N), dtype=np.float64)
    cdef f64[:,:] Obs = np.zeros((nthreads, N), dtype=np.float64)
    cdef f64[:,:,:] D = np.zeros((nthreads, N, K), dtype=np.float64)
    cdef f64[:,:] XQ = np.zeros((nthreads, K), dtype=np.float64)
    cdef f64[:,:] B = np.zeros((nthreads, K), dtype=np.float64)
    cdef f64[:,:,:] S = np.zeros((nthreads, K, K), dtype=np.float64)
    cdef f64[:,:,:] Tm = np.zeros((nthreads, K, K), dtype=np.float64)
    cdef f64[:,:,:] Bv = np.zeros((nthreads, K, K), dtype=np.float64)

    with nogil:
        for w in prange(W, schedule='guided'):
            t = omp.omp_get_thread_num()
            norm = 1.0 / k_chr[w]

            for n in range(N):
                obs = <f64>((Z_miss[w,2*n] == 0) + (Z_miss[w,2*n+1] == 0))
                Obs[t,n] = obs
                for k in range(K):
                    D[t,n,k] = obs * Q[n,k]

            for c in range(k_chr[w]):
                for k in range(K):
                    XQ[t,k] = 0.0
                    B[t,k] = 0.0
                    for l in range(K):
                        S[t,k,l] = 0.0
                        Tm[t,k,l] = 0.0
                        Bv[t,k,l] = 0.0

                for n in range(N):
                    x = <f64>(
                        (Z_miss[w,2*n] == 0 and Z[w,2*n] == c) +
                        (Z_miss[w,2*n+1] == 0 and Z[w,2*n+1] == c)
                    )
                    for k in range(K):
                        XQ[t,k] += D[t,n,k] * x

                for k in range(K):
                    for l in range(K):
                        B[t,k] += XQ[t,l] * A_chr[w,l,k]

                for n in range(N):
                    pred = 0.0
                    for k in range(K):
                        pred = pred + D[t,n,k] * B[t,k]
                    var = 0.0
                    if Obs[t,n] > 0.0 and pred > 0.0 and pred < Obs[t,n]:
                        var = pred * (1.0 - pred / Obs[t,n])
                    Var[t,n] = var
                    for k in range(K):
                        for l in range(K):
                            S[t,k,l] += D[t,n,k] * var * D[t,n,l]

                for k in range(K):
                    for l in range(K):
                        for m in range(K):
                            Tm[t,k,l] += A_chr[w,k,m] * S[t,m,l]

                for k in range(K):
                    for l in range(K):
                        for m in range(K):
                            Bv[t,k,l] += Tm[t,k,m] * A_chr[w,m,l]

                for i in range(N):
                    for j in range(i, N):
                        hij = 0.0
                        eij = 0.0
                        for k in range(K):
                            for l in range(K):
                                hij += D[t,i,k] * A_chr[w,k,l] * D[t,j,l]
                                eij += D[t,i,k] * Bv[t,k,l] * D[t,j,l]
                        C_priv[t,i,j] += (
                            eij - hij * (Var[t,i] + Var[t,j])
                        ) * norm
                        if i == j:
                            C_priv[t,i,j] += Var[t,i] * norm

        for t in range(nthreads):
            for i in range(N):
                for j in range(i, N):
                    C[i,j] += C_priv[t,i,j]


# Estimate covariance of residuals under the proposed model
cpdef void expected(
        f64[:,::1] C, const f64[:,::1] Q, const f64[:,::1] QA, const f64[:,::1] QB,
        const f64[::1] V
    ) noexcept nogil:
    cdef:
        Py_ssize_t N = Q.shape[0]
        Py_ssize_t K = Q.shape[1]
        Py_ssize_t i,j,k
        f64 h = 0.0
        f64 e = 0.0

    with nogil:
        for i in range(N):
            for j in range(i, N):
                h = 0.0
                e = 0.0
                for k in range(K):
                    h += QA[i,k] * Q[j,k]
                    e += QB[i,k] * Q[j,k]
                C[i,j] = e - h * (V[i] + V[j])
                if i == j:
                    C[i,j] += V[i]
                C[j,i] = C[i,j]


# Turn covariances into correlations
cpdef void corr(
        const f64[:,::1] C, f64[:,::1] cor
    ) noexcept nogil:
    cdef:
        Py_ssize_t N_ind = C.shape[0]

    with nogil:
        for i in range(N_ind):
            for j in range(i, N_ind):
                if C[i,i] > 0 and C[j,j] > 0:
                    cor[i,j] = C[i, j] / sqrt(C[i, i] * C[j, j])
                    cor[j,i] = cor[i,j]
                else:
                    cor[i,j] = 0.0
                    cor[j,i] = 0.0
