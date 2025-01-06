# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
import numpy as np
cimport numpy as np
from cython.parallel import prange

##### hapla - analyses on haplotype cluster assignments #####
### hapla struct
# Extract aggregated haplotype cluster counts
cpdef void haplotypeAggregate(const unsigned char[:,::1] Z, unsigned char[:,::1] Z_agg, \
		double[::1] p, const unsigned char[::1] k_vec, const unsigned int[::1] c_vec) \
		noexcept nogil:
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
cpdef void centerZ(const unsigned char[:,::1] Z_agg, float[:,::1] Z_bat, \
		const double[::1] p, const size_t M_b) noexcept nogil:
	cdef:
		size_t M = Z_bat.shape[0]
		size_t N = Z_bat.shape[1]
		size_t i, j, l
		float u
	for j in prange(M):
		l = M_b + j
		u = <float>(2.0*p[l])
		for i in range(N):
			Z_bat[j,i] = Z_agg[l,i] - u

# Standardize permuted batch haplotype cluster assignment matrix
cpdef void blockZ(const unsigned char[:,::1] Z_agg, double[:,::1] Z_bat, \
		const double[::1] p, const double[::1] a, const unsigned int[::1] s, \
		const size_t M_b) noexcept nogil:
	cdef:
		size_t M = Z_bat.shape[0]
		size_t N = Z_bat.shape[1]
		size_t i, j, l
		double d, u
	for j in prange(M):
		l = <size_t>s[M_b + j]
		d = a[l]
		u = 2.0*p[l]
		for i in range(N):
			Z_bat[j,i] = (Z_agg[l,i] - u)*d

# Standardize batch haplotype cluster assignment matrix
cpdef void batchZ(const unsigned char[:,::1] Z_agg, double[:,::1] Z_bat, \
		const double[::1] p, const double[::1] a, const size_t M_b) noexcept nogil:
	cdef:
		size_t M = Z_bat.shape[0]
		size_t N = Z_bat.shape[1]
		size_t i, j, l
		double d, u
	for j in prange(M):
		l = M_b + j
		d = a[l]
		u = 2.0*p[l]
		for i in range(N):
			Z_bat[j,i] = (Z_agg[l,i] - u)*d



### hapla predict
# Calculate Hamming distance
cdef inline unsigned int hammingPred(const unsigned char* X, const unsigned char* R, \
		const size_t M) noexcept nogil:
	cdef:
		size_t j
		unsigned int dist = 0
	for j in range(M):
		if X[j] != 9: # Ignore missing
			dist += X[j]^R[j]
	return dist

# Haplotype cluster assignment based on pre-estimated medians
cpdef void predictCluster(const unsigned char[:,::1] X, const unsigned char[:,::1] R, \
		unsigned char[:,::1] Z, const unsigned int[::1] n_vec, const size_t K, \
		const size_t w) noexcept nogil:
	cdef:
		size_t N = X.shape[0]
		size_t M = X.shape[1]
		size_t c, d, i, j, k, z
	for i in prange(N):
		c = M + 1
		for k in range(K):
			d = hammingPred(&X[i,0], &R[k,0], M)
			if d < c:
				z = k
				c = d
			elif d == c: # Assign to largest cluster in ties
				if n_vec[k] > n_vec[z]:
					z = k
					c = d
		Z[w,i] = z
