# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
import numpy as np
cimport numpy as np
from cython.parallel import prange, parallel
from libc.math cimport sqrt

##### hapla - analyses on haplotype cluster assignments #####
### hapla pca
# Extract aggregated haplotype cluster counts
cpdef void haplotypeAggregate(unsigned char[:,::1] Z_mat, unsigned char[:,::1] Z, \
		double[::1] pi, double[::1] sd, unsigned char[::1] K_vec):
	cdef:
		int W = Z.shape[0]
		int n = Z.shape[1]
		int i, k, w
		int j = 0
	for w in range(W):
		for k in range(K_vec[w]-1):
			for i in range(n):
				if Z_mat[w,i] == k:
					Z[j,i//2] += 1
					pi[j] += 1
			pi[j] /= <double>n
			for i in range(n//2):
				sd[j] += (<double>Z[j,i]-2*pi[j])*(<double>Z[j,i]-2*pi[j])
			sd[j] = sqrt(sd[j]/(<double>(n//2)-1))
			j += 1

# Array filtering
cpdef void filterZ(unsigned char[:,::1] Z, double[::1] pi, \
		double[::1] sd, unsigned char[::1] mask):
	cdef:
		int m = Z.shape[0]
		int n = Z.shape[1]
		int c = 0
		int i, j
	for j in range(m):
		if mask[j] == 1:
			for i in range(n):
				Z[c,i] = Z[j,i]
			pi[c] = pi[j]
			sd[c] = sd[j]
			c += 1

# Standardize the batch haplotype cluster assignment matrix
cpdef void batchZ(unsigned char[:,::1] Z, double[:,::1] Z_b, double[::1] pi, \
		double[::1] sd, int m_b, int t):
	cdef:
		int m = Z_b.shape[0]
		int n = Z_b.shape[1]
		int i, j
	with nogil:
		for j in prange(m, num_threads=t):
			for i in range(n):
				Z_b[j,i] = (Z[m_b+j,i] - 2*pi[m_b+j])/sd[m_b+j]

# Standardize full matrix
cpdef void standardizeZ(unsigned char[:,::1] Z, double[:,::1] Z_std, \
		double[::1] pi, double[::1] sd, int t):
	cdef:
		int m = Z.shape[0]
		int n = Z.shape[1]
		int i, j
	with nogil:
		for j in prange(m, num_threads=t):
			for i in range(n):
				Z_std[j,i] = (Z[j,i] - 2*pi[j])/sd[j]



### hapla split
# Estimate squared correlation between variants (r^2) and compute L matrix
cpdef void estimateL(unsigned char[:,::1] Gt, double[::1] F, double[::1] S, \
				float[:,::1] L, double thr, int n, int t):
	cdef:
		int m = Gt.shape[0]
		int B = Gt.shape[1]
		int W = L.shape[1]
		int b, c, i, j, k, bit
		unsigned char mask = 1
		unsigned char b1
		unsigned char b2
		double corr, r2
	with nogil, parallel(num_threads=t):
		# Estimate means and standard deviations
		for j in prange(m):
			k = 0
			for b in range(B):
				b1 = Gt[j,b]
				for bit in range(8):
					F[j] += <double>(b1 & mask)
					b1 = b1 >> 1
					k = k + 1
					if k == n:
						break
			F[j] /= <double>n
			k = 0
			for b in range(B):
				b1 = Gt[j,b]
				for bit in range(8):
					S[j] += (<double>(b1 & mask)-F[j])*(<double>(b1 & mask)-F[j])
					b1 = b1 >> 1
					k = k + 1
					if k == n:
						break
			S[j] = sqrt(S[j]/(<double>n-1))
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
							(<double>(b1 & mask)-F[j])*(<double>(b2 & mask)-F[i])/(S[j]*S[i])
						b1 = b1 >> 1
						b2 = b2 >> 1
						k = k + 1
						if k == n:
							break
				corr = corr/<double>n
				r2 = corr*corr
				if r2 > thr:
					L[j,c] = <float>r2
				L[j,c] += L[j,c+1]
				c = c - 1

# Estimate E matrix used for cost estimation
cpdef void estimateE(float[:,::1] L, float[:,::1] E):
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
cpdef void reconstructPath(int[:,::1] I, int[::1] P, int k):
	cdef:
		int j = 0
		int i
	for i in range(k, -1, -1):
		j = I[i,j]
		P[i] = j



### hapla predict
# Haplotype cluster assignment based on pre-estimated medians
cpdef void predictCluster(unsigned char[:,::1] X, signed char[:,::1] M, \
		unsigned char[:,::1] Z, int K, int w, int t):
	cdef:
		int n = X.shape[0]
		int m = X.shape[1]
		int i, j, k, dist, m_val
	with nogil:
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
					