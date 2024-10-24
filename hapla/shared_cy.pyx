# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
import numpy as np
cimport numpy as np
from cython.parallel import prange

##### hapla - analyses on haplotype cluster assignments #####
### hapla struct
# Extract aggregated haplotype cluster counts
cpdef void haplotypeAggregate(const unsigned char[:,::1] Z, \
		unsigned char[:,::1] Z_agg, float[::1] p, const unsigned char[::1] k_vec) \
		noexcept nogil:
	cdef:
		int W = Z.shape[0]
		int n = Z.shape[1]
		int j = 0
		int i, k, w
		float d = 1.0/<float>n
	for w in range(W):
		for k in range(k_vec[w]):
			for i in range(n):
				if Z[w,i] == k:
					Z_agg[j,i//2] += 1
					p[j] += 1.0
			p[j] *= d
			j += 1

# Standardize the batch haplotype cluster assignment matrix
cpdef void batchZ(const unsigned char[:,::1] Z_agg, float[:,::1] Z_bat, \
		const float[::1] p, const float[::1] a, const int m_b, const int t) \
		noexcept nogil:
	cdef:
		int m = Z_bat.shape[0]
		int n = Z_bat.shape[1]
		int i, j, l
		float s, u
	for j in prange(m, num_threads=t):
		l = m_b + j
		s = a[l]
		u = 2*p[l]
		for i in range(n):
			Z_bat[j,i] = (Z_agg[l,i] - u)*s

# Standardize full matrix
cpdef void standardizeZ(const unsigned char[:,::1] Z_agg, float[:,::1] Z_std, \
		const float[::1] p, const float[::1] a, const int t) noexcept nogil:
	cdef:
		int m = Z_agg.shape[0]
		int n = Z_agg.shape[1]
		int i, j
		float s, u
	for j in prange(m, num_threads=t):
		s = a[j]
		u = 2*p[j]
		for i in range(n):
			Z_std[j,i] = (Z_agg[j,i] - u)*s



### hapla predict
# Calculate Hamming distance
cdef inline int hammingPred(const unsigned char* X, const unsigned char* M, \
		const int m) noexcept nogil:
	cdef:
		int dist = 0
		int j
	for j in range(m):
		if X[j] != 9: # Ignore missing
			dist += <int>(X[j] ^ M[j])
	return dist

# Haplotype cluster assignment based on pre-estimated medians
cpdef void predictCluster(const unsigned char[:,::1] X, const unsigned char[:,::1] M, \
		unsigned char[:,::1] Z, const int[::1] n_vec, const int K, const int w, \
		const int t) noexcept nogil:
	cdef:
		int n = X.shape[0]
		int m = X.shape[1]
		int c, d, i, j, k, z
	for i in prange(n, num_threads=t):
		c = m + 1
		for k in range(K):
			d = hammingPred(&X[i,0], &M[k,0], m)
			if d < c:
				z = k
				c = d
			elif d == c: # Assign to largest cluster in ties
				if n_vec[k] > n_vec[z]:
					z = k
					c = d
		Z[w,i] = z
