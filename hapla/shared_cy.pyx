# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
cimport numpy as np
from cython.parallel import prange
from libc.stdint cimport uint8_t, uint32_t

##### hapla - analyses on haplotype cluster assignments #####
### hapla struct
# Extract aggregated haplotype cluster counts
cpdef void haplotypeAggregate(
		const uint8_t[:,::1] Z, uint8_t[:,::1] Z_agg, double[::1] p, const uint8_t[::1] k_vec, 
		const uint32_t[::1] c_vec
	) noexcept nogil:
	cdef:
		size_t W = Z.shape[0]
		size_t N = Z.shape[1]
		size_t c, i, l, s, w
		double d = 1.0/<double>N
	for w in prange(W):
		s = <size_t>c_vec[w]
		for c in range(k_vec[w]):
			l = s + c
			for i in range(N):
				if Z[w,i] == c:
					Z_agg[l,i//2] += 1
					p[l] += 1.0
			p[l] *= d

# Center batch haplotype cluster assignment matrix
cpdef void centerZ(
		const uint8_t[:,::1] Z_agg, float[:,::1] Z_bat, const double[::1] p, const size_t m
	) noexcept nogil:
	cdef:
		float u
		size_t M = Z_bat.shape[0]
		size_t N = Z_bat.shape[1]
		size_t i, j, l
	for j in prange(M):
		l = m+j
		u = 2.0*<float>p[l]
		for i in range(N):
			Z_bat[j,i] = Z_agg[l,i] - u

# Standardize permuted batch haplotype cluster assignment matrix
cpdef void blockZ(
		const uint8_t[:,::1] Z_agg, double[:,::1] Z_bat, const double[::1] p, const double[::1] a, 
		const uint32_t[::1] s, const size_t m
	) noexcept nogil:
	cdef:
		double d, u
		size_t M = Z_bat.shape[0]
		size_t N = Z_bat.shape[1]
		size_t i, j, l
	for j in prange(M):
		l = s[m+j]
		d = a[l]
		u = 2.0*p[l]
		for i in range(N):
			Z_bat[j,i] = (Z_agg[l,i] - u)*d

# Standardize batch haplotype cluster assignment matrix
cpdef void batchZ(
		const uint8_t[:,::1] Z_agg, double[:,::1] Z_bat, const double[::1] p, const double[::1] a, const size_t m
	) noexcept nogil:
	cdef:
		double d, u
		size_t M = Z_bat.shape[0]
		size_t N = Z_bat.shape[1]
		size_t i, j, l
	for j in prange(M):
		l = m+j
		d = a[l]
		u = 2.0*p[l]
		for i in range(N):
			Z_bat[j,i] = (Z_agg[l,i] - u)*d



### hapla predict
# Calculate Hamming distance
cdef inline uint32_t hammingPred(
		const uint8_t* X, const uint8_t* R, const size_t M
	) noexcept nogil:
	cdef:
		size_t j
		uint32_t dist = 0
	for j in range(M):
		if X[j] != R[j] and X[j] != 9: # Ignore missing
			dist += 1
	return dist

# Haplotype cluster assignment based on pre-estimated medians
cpdef void predictCluster(
		uint8_t[:,::1] X, const uint8_t[:,::1] R, uint8_t[:,::1] Z, const uint32_t[::1] n_vec, const size_t K, 
		const size_t w
	) noexcept nogil:
	cdef:
		size_t c, d, i, k, z
		size_t N = X.shape[0]
		size_t M = X.shape[1]
		uint8_t* xi
	for i in prange(N):
		xi = &X[i,0]
		z = 0
		c = hammingPred(xi, &R[0,0], M)
		for k in range(1, K):
			d = hammingPred(xi, &R[k,0], M)
			if d < c or (d == c and n_vec[k] > n_vec[z]):
				z = k
				c = d
		Z[w,i] = <uint8_t>z
