# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
import numpy as np
cimport numpy as np
from cython.parallel import prange, parallel
from libc.math cimport pow, sqrt

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

# Estimate genome-wide relationship matrix in condensed form
cpdef void grmCondensed(unsigned char[:,::1] Z, float[::1] G, float[::1] p, \
		float[::1] s, int t) nogil:
	cdef:
		int n = Z.shape[0]
		int m = Z.shape[1]
		int K = G.shape[0]
		int i, j, k, l
		unsigned char *il
		unsigned char *jl
	for k in prange(K, num_threads=t):
		i = <int>((sqrt(1 + 8*k) - 1)//2) # Row index in condensed form
		j = k - <int>(i*(i + 3)//2) + i # Column index in condensed form
		il = &Z[i,0]
		jl = &Z[j,0]
		for l in range(m):
			G[k] += (<float>il[l] - 2*p[l])*(<float>jl[l] - 2*p[l])/s[l]
		G[k] /= <float>(m)

# Estimate genome-wide relationship matrix in full form
cpdef void grmFull(unsigned char[:,::1] Z, float[:,::1] G, float[::1] p, \
		float[::1] s, int K, int t) nogil:
	cdef:
		int n = Z.shape[0]
		int m = Z.shape[1]
		int i, j, k, l
		unsigned char *il
		unsigned char *jl
	for k in prange(K, num_threads=t):
		i = <int>((sqrt(1 + 8*k) - 1)//2) # Row index in condensed form
		j = k - <int>(i*(i + 3)//2) + i # Column index in condensed form
		il = &Z[i,0]
		jl = &Z[j,0]
		for l in range(m):
			G[i,j] += (<float>il[l] - 2*p[l])*(<float>jl[l] - 2*p[l])/s[l]
		G[i,j] /= <float>(m)
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
			j = j + 1

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
			c = c + 1

# Standardize the batch haplotype cluster assignment matrix
cpdef void batchZ(unsigned char[:,::1] Z, float[:,::1] Z_b, float[::1] p, \
		int m_b, int t) nogil:
	cdef:
		int m = Z_b.shape[0]
		int n = Z_b.shape[1]
		int b, i, j
	for j in prange(m, num_threads=t):
		b = m_b+j
		for i in range(n):
			Z_b[j,i] = (Z[b,i] - 2*p[b])/sqrt(2*p[b]*(1-p[b]))

# Standardize full matrix
cpdef void standardizeZ(unsigned char[:,::1] Z, float[:,::1] Z_std, \
		float[::1] p, int t) nogil:
	cdef:
		int m = Z.shape[0]
		int n = Z.shape[1]
		int i, j
	for j in prange(m, num_threads=t):
		for i in range(n):
			Z_std[j,i] = (Z[j,i] - 2*p[j])/sqrt(2*p[j]*(1-p[j]))



### hapla split
# Estimate squared correlation between variants (r^2) and compute L matrix
cpdef void estimateL(unsigned char[:,::1] Gt, float[::1] F, float[::1] S, \
		float[:,::1] L, float thr, int n, int t):
	cdef:
		int m = Gt.shape[0]
		int B = Gt.shape[1]
		int W = L.shape[1]
		int b, c, i, j, k, bit
		unsigned char mask = 1
		unsigned char b1
		unsigned char b2
		float corr, r2
	with nogil, parallel(num_threads=t):
		# Estimate means and standard deviations
		for j in prange(m):
			k = 0
			for b in range(B):
				b1 = Gt[j,b]
				for bit in range(8):
					F[j] += <float>(b1 & mask)
					b1 = b1 >> 1
					k = k + 1
					if k == n:
						break
			F[j] /= <float>n
			k = 0
			for b in range(B):
				b1 = Gt[j,b]
				for bit in range(8):
					S[j] += (<float>(b1 & mask)-F[j])*(<float>(b1 & mask)-F[j])
					b1 = b1 >> 1
					k = k + 1
					if k == n:
						break
			S[j] = sqrt(S[j]/(<float>n-1))
		# Estimate squared correlations
		for j in prange(m-1):
			if j > (m - W):
				c = m - j - 2
			else:
				c = W - 2
			for i in range(min(j+W, m)-1, j, -1):
				k = 0
				corr = 0.0
				for b in range(B):
					b1 = Gt[j,b]
					b2 = Gt[i,b]
					for bit in range(8):
						corr = corr + \
							(<float>(b1 & mask)-F[j])*(<float>(b2 & mask)-F[i])/(S[j]*S[i])
						b1 = b1 >> 1
						b2 = b2 >> 1
						k = k + 1
						if k == n:
							break
				corr = corr/<float>n
				r2 = corr*corr
				if r2 > thr:
					L[j,c] = <float>r2
				L[j,c] += L[j,c+1]
				c = c - 1

# Estimate E matrix used for cost estimation
cpdef void estimateE(float[:,::1] L, float[:,::1] E) nogil:
	cdef:
		int m = E.shape[0]
		int W = E.shape[1]
		int j, k
	for j in range(m-2, -1, -1):
		for k in range(W-1, -1, -1):
			if k == 0:
				E[j,k] = L[j,k]
			else:
				E[j,k] = L[j,k] + E[j+1,k-1]

# Compute cost for different number of splits
cpdef void estimateC(float[:,::1] E, float[:,::1] C, int[:,::1] I, int w0, int t):
	cdef:
		int m = E.shape[0]
		int W = E.shape[1]
		int K = C.shape[0]
		int c, j, k, w
		float cost
	for c in range(w0, W+1):
		C[0,m-c] = 0.0
		I[0,m-c] = m
	with nogil, parallel(num_threads=t):
		for k in range(1, K):
			for j in prange(m-(k+1)*w0, -1, -1):
				cost = 0.0
				for w in range(w0-1, min(W, m-j-1)):
					cost = E[j,w] + C[k-1,j+w+1]
					if cost < C[k,j]:
						C[k,j] = cost
						I[k,j] = j+w+1

# Reconstruct path of the lowest cost
cpdef void reconstructPath(int[:,::1] I, int[::1] P, int k) nogil:
	cdef:
		int j = 0
		int i
	for i in range(k, -1, -1):
		j = I[i,j]
		P[i] = j



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
				if X[i,j] != M[k,j]:
					dist = dist + 1
			# Assignment
			if dist < m_val:
				Z[w,i] = k # Cluster assignment
				m_val = dist
					