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
		uint32_t W = Z.shape[0]
		uint32_t N = Z.shape[1]
		uint32_t s
		double d = 1.0/<double>N
		size_t c, i, l, w
	for w in prange(W):
		s = c_vec[w]
		for c in range(k_vec[w]):
			l = s+c
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
		uint32_t M = Z_bat.shape[0]
		uint32_t N = Z_bat.shape[1]
		float u
		size_t i, j, l
	for j in prange(M):
		l = m+j
		u = 2.0*<float>p[l]
		for i in range(N):
			Z_bat[j,i] = Z_agg[l,i] - u

# Standardize batch haplotype cluster assignment matrix
cpdef void batchZ(
		const uint8_t[:,::1] Z_agg, double[:,::1] Z_bat, const double[::1] p, const double[::1] a, const size_t m
	) noexcept nogil:
	cdef:
		uint32_t M = Z_bat.shape[0]
		uint32_t N = Z_bat.shape[1]
		double d, u
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
		const uint8_t* X, const uint8_t* R, const uint32_t M
	) noexcept nogil:
	cdef:
		uint32_t dist = 0
		size_t j
	for j in range(M):
		if X[j] != R[j] and X[j] != 9: # Ignore missing
			dist += 1
	return dist

# Haplotype cluster assignment based on pre-estimated medians
cpdef void predictCluster(
		uint8_t[:,::1] X, const uint8_t[:,::1] R, uint8_t[:,::1] Z, const uint32_t K, const uint32_t w
	) noexcept nogil:
	cdef:
		uint8_t* h
		uint32_t N = X.shape[0]
		uint32_t M = X.shape[1]
		size_t c, d, i, k, z
	for i in prange(N):
		h = &X[i,0]
		z = 0
		c = hammingPred(h, &R[0,0], M)
		for k in range(1, K):
			d = hammingPred(h, &R[k,0], M)
			if d <= c:
				z = k
				c = d
		Z[w,i] = <uint8_t>z
