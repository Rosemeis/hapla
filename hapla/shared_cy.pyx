# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
import numpy as np
cimport numpy as np
from cython.parallel import prange, parallel
from libc.math cimport sqrt

##### hapla - analyses on haplotype cluster assignments #####
### hapla pca
# Estimate haplotype sharing matrix in condensed form
cpdef void hsmCondensed(unsigned char[:,::1] Z, float[::1] G, int t) \
		nogil:
	cdef:
		int n = Z.shape[0]//2
		int W = Z.shape[1]
		int K = G.shape[0]
		int i, j, k, w, g1, g2
		unsigned char *i0
		unsigned char *i1
		unsigned char *j0
		unsigned char *j1
	for k in prange(K, num_threads=t):
		i = <int>((sqrt(1 + 8*k) - 1)//2) # Row index in condensed form
		j = k - <int>(i*(i + 3)//2) + i # Column index in condensed form
		if i == j: # Diagonal
			G[k] = 1.0
		else:
			i0 = &Z[2*i+0,0]
			i1 = &Z[2*i+1,0]
			j0 = &Z[2*j+0,0]
			j1 = &Z[2*j+1,0]
			for w in range(W):
				g1 = <int>(i0[w] == j0[w]) + <int>(i1[w] == j1[w])
				g2 = <int>(i1[w] == j0[w]) + <int>(i0[w] == j1[w])
				G[k] += <float>(max(g1, g2))
			G[k] /= <float>(2*W)

# Estimate haplotype sharing matrix in full form
cpdef void hsmFull(unsigned char[:,::1] Z, float[:,::1] G, int K, int t) \
		nogil:
	cdef:
		int n = Z.shape[0]//2
		int W = Z.shape[1]
		int i, j, k, w, g1, g2
		unsigned char *i0
		unsigned char *i1
		unsigned char *j0
		unsigned char *j1
	for k in prange(K, num_threads=t):
		i = <int>((sqrt(1 + 8*k) - 1)//2) # Row index in condensed form
		j = k - <int>(i*(i + 3)//2) + i # Column index in condensed form
		if i == j: # Diagonal
			G[i,j] = 1.0
		else:
			i0 = &Z[2*i+0,0]
			i1 = &Z[2*i+1,0]
			j0 = &Z[2*j+0,0]
			j1 = &Z[2*j+1,0]
			for w in range(W):
				g1 = <int>(i0[w] == j0[w]) + <int>(i1[w] == j1[w])
				g2 = <int>(i1[w] == j0[w]) + <int>(i0[w] == j1[w])
				G[i,j] += <float>(max(g1, g2))
			G[i,j] /= <float>(2*W)
			G[j,i] = G[i,j]

# Extract aggregated haplotype cluster counts
cpdef void haplotypeAggregate(unsigned char[:,::1] Z_mat, unsigned char[:,::1] Z, \
		float[::1] p, unsigned char[::1] K_vec) nogil:
	cdef:
		int W = Z_mat.shape[0]
		int n = Z_mat.shape[1]
		int j = 0
		int i, k, w
	for w in range(W):
		for k in range(K_vec[w]):
			for i in range(n):
				if Z_mat[w,i] == k:
					Z[j,i//2] += 1
					p[j] += 1
			p[j] /= <float>n
			j += 1

# Array filtering
cpdef void filterZ(unsigned char[:,::1] Z, float[::1] p, \
		unsigned char[::1] mask) nogil:
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
		float[::1] s, int m_b, int t) nogil:
	cdef:
		int m = Z_b.shape[0]
		int n = Z_b.shape[1]
		int b, i, j
	for j in prange(m, num_threads=t):
		b = m_b+j
		for i in range(n):
			Z_b[j,i] = (Z[b,i] - 2*p[b])/s[b]

# Standardize full matrix
cpdef void standardizeZ(unsigned char[:,::1] Z, float[:,::1] Z_std, \
		float[::1] p, float[::1] s, int t) nogil:
	cdef:
		int m = Z.shape[0]
		int n = Z.shape[1]
		int i, j
	for j in prange(m, num_threads=t):
		for i in range(n):
			Z_std[j,i] = (Z[j,i] - 2*p[j])/s[j]



### hapla predict
# Haplotype cluster assignment based on pre-estimated medians
cpdef void predictCluster(unsigned char[:,::1] X, signed char[:,::1] M, \
		unsigned char[:,::1] Z, int K, int w, int t) nogil:
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
					