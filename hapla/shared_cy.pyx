# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
import numpy as np
cimport numpy as np
from cython.parallel import prange, parallel
from libc.math cimport sqrt

##### hapla - analyses on haplotype cluster assignments #####
### hapla pca
# Estimate haplotype cluster frequencies
cpdef void haplotypeFreqs(unsigned char[:,::1] Z, double[:,::1] Z_bar, \
		unsigned char[::1] K_vec, double[::1] pi):
	cdef int W = Z.shape[0]
	cdef int n = Z.shape[1]
	cdef int i, k, w
	cdef int j = 0
	for w in range(W):
		for k in range(K_vec[w]):
			for i in range(n):
				if Z[w,i] == k:
					pi[j] += 1.0
					Z_bar[j,i//2] += 1.0
			pi[j] /= <double>n
			j += 1

# Array filtering
cpdef void filterZ(double[:,::1] Z_bar, double[::1] pi, unsigned char[::1] mask):
	cdef int m = Z_bar.shape[0]
	cdef int n = Z_bar.shape[1]
	cdef int c = 0
	cdef int i, j
	for j in range(m):
		if mask[j] == 1:
			for i in range(n):
				Z_bar[c,i] = Z_bar[j,i]
			pi[c] = pi[j]
			c += 1

# Standardize the full haplotype cluster assignment matrix
cpdef void standardizeZ(double[:,::1] Z_bar, double[::1] pi, int t):
	cdef int m = Z_bar.shape[0]
	cdef int n = Z_bar.shape[1]
	cdef int i, j
	cdef double s
	with nogil:
		for j in prange(m, num_threads=t):
			s = 0.0
			for i in range(n):
				s = s + (Z_bar[j,i] - 2*pi[j])*(Z_bar[j,i] - 2*pi[j])
			s = sqrt(s/<double>n)
			for i in range(n):
				Z_bar[j,i] = (Z_bar[j,i] - 2*pi[j])/s


### hapla split
# Estimate squared correlation between variants (r^2) and compute L matrix
cpdef void estimateL(unsigned char[:,::1] Gt, double[::1] F, double[::1] S, \
				float[:,::1] L, double thr, int n, int t):
	cdef int m = Gt.shape[0]
	cdef int B = Gt.shape[1]
	cdef int W = L.shape[1]
	cdef int b, c, i, j, k, bit
	cdef unsigned char mask = 1
	cdef unsigned char b1
	cdef unsigned char b2
	cdef double corr, r2
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
			S[j] = sqrt(S[j]/<double>n)
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
	cdef int m = E.shape[0]
	cdef int W = E.shape[1]
	cdef int j, k
	for j in range(m-2, -1, -1):
		for k in range(W-1, -1, -1):
			if k == 0:
				E[j,k] = L[j,k]
			else:
				E[j,k] = L[j,k] + E[j+1,k-1]

# Compute cost for different number of splits
cpdef void estimateC(float[:,::1] E, float[:,::1] C, int[:,::1] I, int w0, int t):
	cdef int m = E.shape[0]
	cdef int W = E.shape[1]
	cdef int K = C.shape[0]
	cdef int c, j, k, w
	cdef float cost
	for c in range(w0, W+1):
		C[0,m-c] = 0.0
		I[0,m-c] = m
	with nogil, parallel(num_threads=t):
		for k in range(1, K):
			for j in prange(m-(k+1)*w0, -1, -1):
				cost = 0.0
				for w in range(w0-1, min(W, m-j-1)):
					cost = E[j,w] + C[k-1,j+w+1]
					if cost <= C[k,j]:
						C[k,j] = cost
						I[k,j] = j+w+1

# Reconstruct path of the lowest cost
cpdef void reconstructPath(int[:,::1] I, int[::1] P, int k):
	cdef int j = 0
	cdef int i = k
	while i >= 0:
		j = I[i,j]
		P[i] = j
		i -= 1


### hapla predict
# Haplotype cluster assignment based on pre-estimated medians
cpdef void predictCluster(signed char[:,::1] X, signed char[:,::1] M, \
		unsigned char[:,::1] Z, int K, int w, int t):
	cdef int n = X.shape[0]
	cdef int m = X.shape[1]
	cdef int i, j, k, dist, m_val
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
					