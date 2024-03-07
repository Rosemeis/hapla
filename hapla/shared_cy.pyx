# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
import numpy as np
cimport numpy as np
from cython.parallel import prange, parallel

##### hapla - analyses on haplotype cluster assignments #####
### hapla struct
# Find rarest cluster in each window
cpdef void findRare(const unsigned char[:,::1] Z_mat, unsigned char[::1] R_vec, \
		const unsigned char[::1] K_vec, int t) noexcept nogil:
	cdef:
		int W = Z_mat.shape[0]
		int n = Z_mat.shape[1]
		int i, k, w, k_cnt, k_min
	for w in prange(W, num_threads=t):
		k_cnt = n
		for k in range(K_vec[w]):
			k_min = 0
			for i in range(n):
				if Z_mat[w,i] == k:
					k_min = k_min + 1
			if k_min < k_cnt: # Set rarest cluster
				k_cnt = k_min
				R_vec[w] = k

# Extract aggregated haplotype cluster counts
cpdef void haplotypeAggregate(const unsigned char[:,::1] Z_mat, \
		unsigned char[:,::1] Z, float[::1] p, const unsigned char[::1] R_vec, \
		const unsigned char[::1] K_vec) noexcept nogil:
	cdef:
		int W = Z_mat.shape[0]
		int n = Z_mat.shape[1]
		int j = 0
		int i, k, w
		float d = 1.0/<float>n
	for w in range(W):
		for k in range(K_vec[w]):
			if k != R_vec[w]: # Skip rarest cluster
				for i in range(n):
					if Z_mat[w,i] == k:
						Z[j,i//2] += 1
						p[j] += 1.0
				p[j] *= d
				j += 1

# Standardize the batch haplotype cluster assignment matrix
cpdef void batchZ(const unsigned char[:,::1] Z, float[:,::1] Z_b, const float[::1] p, \
		const float[::1] a, const int m_b, const int t) noexcept nogil:
	cdef:
		int m = Z_b.shape[0]
		int n = Z_b.shape[1]
		int i, j, j_b
	for j in prange(m, num_threads=t):
		j_b = m_b+j
		for i in range(n):
			Z_b[j,i] = (Z[j_b,i] - 2*p[j_b])*a[j_b]

# Standardize full matrix
cpdef void standardizeZ(const unsigned char[:,::1] Z, float[:,::1] Z_s, \
		const float[::1] p, const float[::1] a, const int t) noexcept nogil:
	cdef:
		int m = Z.shape[0]
		int n = Z.shape[1]
		int i, j
	for j in prange(m, num_threads=t):
		for i in range(n):
			Z_s[j,i] = (Z[j,i] - 2*p[j])*a[j]



### hapla predict
# Haplotype cluster assignment based on pre-estimated medians
cpdef void predictCluster(const unsigned char[:,::1] X, const signed char[:,::1] M, \
		unsigned char[:,::1] Z, const int K, const int w, const int t) noexcept nogil:
	cdef:
		int n = X.shape[0]
		int m = X.shape[1]
		int i, j, k, dist, m_val
	for i in prange(n, num_threads=t):
		m_val = m 
		for k in range(K):
			dist = 0
			for j in range(m):
				if X[i,j] != 9: # Ignore missing
					if X[i,j] != M[k,j]:
						dist = dist + 1
			# Assignment
			if dist < m_val:
				Z[w,i] = k # Cluster assignment
				m_val = dist
					