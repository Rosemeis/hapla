# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
import numpy as np
cimport numpy as np
from cython.parallel import prange
from libc.math cimport sqrt

##### hapla - optimal window splits #####
# Extract standardized haplotype matrix
cpdef void extractG(const unsigned char[:,::1] G, float[:,::1] X, const int m_b, \
		const int t) noexcept nogil:
	cdef:
		int B = G.shape[1]
		int m = X.shape[0]
		int n = X.shape[1]
		int b, i, j, bit
		float p, s, x
		unsigned char mask = 1
		unsigned char byte
	for j in prange(m, num_threads=t):
		i = 0
		p = 0
		x = 0
		for b in range(B):
			byte = G[m_b+j,b]
			for bit in range(8):
				X[j,i] = <float>(byte & mask)
				p = p + X[j,i]
				x = x + X[j,i]*X[j,i]
				byte = byte >> 1 # Right shift 1 bit
				i = i + 1
				if i == n:
					break
		p = p/<float>n
		x = x/<float>n
		s = 1.0/sqrt(x - p*p)
		for i in range(n):
			X[j,i] -= p
			X[j,i] *= s

# Estimate squared correlation between variants (r^2) and compute L matrix
cpdef void estimateL(const float[:,::1] X, float[:,::1] L, const float thr, \
		const int t) noexcept nogil:
	cdef:
		int m = X.shape[0]
		int n = X.shape[1]
		int W = L.shape[1]
		int i, j, k, c
		float cor, r2
	for i in prange(m-1, num_threads=t):
		if i > (m - W):
			c = m - i - 2
		else:
			c = W - 2
		for j in range(min(i+W, m)-1, i, -1):
			cor = 0.0
			for k in range(n):
				cor = cor + X[i,k]*X[j,k]
			cor = cor/<float>n
			r2 = cor*cor
			if r2 >= thr:
				L[i,c] += r2
			L[i,c] += L[i,c+1]
			c = c - 1

# Estimate E matrix used for cost estimation
cpdef void estimateE(const float[:,::1] L, float[:,::1] E) noexcept nogil:
	cdef:
		int m = E.shape[0]
		int W = E.shape[1]
		int i, j, k
	for i in range(m-2, -1, -1):
		for j in range(W-1, -1, -1):
			if j == 0:
				E[i,j] = L[i,j]
			else:
				E[i,j] = L[i,j] + E[i+1,j-1]

# Compute cost for different number of splits
cpdef int estimateC(const float[:,::1] E, float[:,::1] C, int[:,::1] I, \
		const int minW, const int t) noexcept nogil:
	cdef:
		int m = E.shape[0]
		int W = E.shape[1]
		int K = C.shape[1]
		int c, i, j, k, w, optK
		float cost, optC
	for c in range(minW, W+1):
		C[m-c,0] = 0
		I[m-c,0] = m
	for k in range(1, K):
		for i in prange(m-(k+1)*minW, -1, -1, num_threads=t):
			cost = 0.0
			for w in range(minW-1, min(W, m-i-1)):
				cost = E[i,w] + C[i+w+1,k-1]
				if cost < C[i,k]:
					C[i,k] = cost
					I[i,k] = i+w+1
		if k > 1:
			if C[i,k] <= optC:
				optC = C[i,k]
				optK = k
		else:
			optC = C[i,k]
			optK = k
	return optK

# Reconstruct path of the lowest cost
cpdef void constructP(const int[:,::1] I, int[::1] P, const int m_b, const int k) \
		noexcept nogil:
	cdef:
		int i = 0
		int j = k
	while j >= 0:
		i = I[i,j]
		P[j] = i + m_b
		j -= 1
