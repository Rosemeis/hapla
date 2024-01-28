# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
import numpy as np
cimport numpy as np
from cython.parallel import prange, parallel
from libc.math cimport sqrt

##### hapla - analyses on haplotype cluster assignments #####
### hapla struct
# Estimate haplotype sharing matrix in condensed form
cpdef void hsmCondensed(unsigned char[:,::1] Z, float[::1] G, int t) \
		nogil:
	cdef:
		int n = Z.shape[0]//2
		int W = Z.shape[1]
		int K = G.shape[0]
		int i, j, k, w, g1, g2
		float d = 1.0/<float>(2*W)
	for k in prange(K, num_threads=t):
		i = <int>((sqrt(1 + 8*k) - 1)//2) # Row index in condensed form
		j = k - <int>(i*(i + 3)//2) + i # Column index in condensed form
		if i == j: # Diagonal
			G[k] = 1.0
		else:
			for w in range(W):
				g1 = <int>(Z[2*i+0,w] == Z[2*j+0,w]) + <int>(Z[2*i+1,w] == Z[2*j+1,w])
				g2 = <int>(Z[2*i+1,w] == Z[2*j+0,w]) + <int>(Z[2*i+0,w] == Z[2*j+1,w])
				G[k] += <float>(max(g1, g2))
			G[k] *= d

# Estimate haplotype sharing matrix in full form
cpdef void hsmFull(unsigned char[:,::1] Z, float[:,::1] G, int K, int t) \
		nogil:
	cdef:
		int n = Z.shape[0]//2
		int W = Z.shape[1]
		int i, j, k, w, g1, g2
		float d = 1.0/<float>(2*W)
	for k in prange(K, num_threads=t):
		i = <int>((sqrt(1 + 8*k) - 1)//2) # Row index in condensed form
		j = k - <int>(i*(i + 3)//2) + i # Column index in condensed form
		if i == j: # Diagonal
			G[i,j] = 1.0
		else:
			for w in range(W):
				g1 = <int>(Z[2*i+0,w] == Z[2*j+0,w]) + <int>(Z[2*i+1,w] == Z[2*j+1,w])
				g2 = <int>(Z[2*i+1,w] == Z[2*j+0,w]) + <int>(Z[2*i+0,w] == Z[2*j+1,w])
				G[i,j] += <float>(max(g1, g2))
			G[i,j] *= d
			G[j,i] = G[i,j]

# Extract aggregated haplotype cluster counts
cpdef void haplotypeAggregate(unsigned char[:,::1] Z_mat, unsigned char[:,::1] Z, \
		float[::1] p, unsigned char[::1] K_vec) noexcept nogil:
	cdef:
		int W = Z_mat.shape[0]
		int n = Z_mat.shape[1]
		int j = 0
		int i, k, w
		float d = 1.0/<float>n 
	for w in range(W):
		for k in range(K_vec[w]):
			for i in range(n):
				if Z_mat[w,i] == (k+1):
					Z[j,i//2] += 1
					p[j] += 1
			p[j] *= d
			j += 1

# Array filtering
cpdef void filterZ(unsigned char[:,::1] Z, float[::1] p, unsigned char[::1] mask) \
		noexcept nogil:
	cdef:
		int m = Z.shape[0]
		int n = Z.shape[1]
		int c = 0
		int i, j
	for j in range(m):
		if mask[j] == 1:
			for i in range(n):
				Z[c,i] = Z[j,i]
			p[c] = p[j]
			c += 1

# Standardize the batch haplotype cluster assignment matrix
cpdef void batchZ(unsigned char[:,::1] Z, float[:,::1] Z_b, float[::1] p, \
		float[::1] a, int m_b, int t) noexcept nogil:
	cdef:
		int m = Z_b.shape[0]
		int n = Z_b.shape[1]
		int i, j, j_b
	for j in prange(m, num_threads=t):
		j_b = m_b+j
		for i in range(n):
			Z_b[j,i] = (Z[j_b,i] - 2*p[j_b])*a[j_b]

# Standardize full matrix
cpdef void standardizeZ(unsigned char[:,::1] Z, float[:,::1] Z_s, \
		float[::1] p, float[::1] a, int t) noexcept nogil:
	cdef:
		int m = Z.shape[0]
		int n = Z.shape[1]
		int i, j
	for j in prange(m, num_threads=t):
		for i in range(n):
			Z_s[j,i] = (Z[j,i] - 2*p[j])*a[j]



### hapla predict
# Haplotype cluster assignment based on pre-estimated medians
cpdef void predictCluster(unsigned char[:,::1] X, signed char[:,::1] M, \
		unsigned char[:,::1] Z, int K, int w, int t) noexcept nogil:
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
					