# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
import numpy as np
cimport numpy as np
from cpython.mem cimport PyMem_RawMalloc, PyMem_RawFree
from cython.parallel import prange, parallel
from libc.math cimport log

##### hapla - haplotype clustering #####
# Create marginal medians
cpdef void marginalMedians(signed char[:,::1] M, float[:,::1] C, int[::1] N, int K):
	cdef int m = M.shape[1]
	cdef int j, k
	for k in range(K):
		if N[k] > 0:
			for j in range(m):
				C[k,j] = C[k,j]/<float>N[k]
				M[k,j] = <signed char>(C[k,j] > 0.5)
				C[k,j] = 0.0

# Compute distances and cluster assignment
cpdef void clusterAssignment(unsigned char[:,::1] X, signed char[:,::1] M, float[:,::1] C, \
		unsigned char[:,::1] Z, int[::1] c, int[::1] N, int K, int w, int t):
	cdef int n = X.shape[0]
	cdef int m = X.shape[1]
	cdef int i, j, k, l, k2, j2, dist, m_val
	cdef float* tmp
	with nogil, parallel(num_threads=t):
		tmp = <float*>PyMem_RawMalloc(sizeof(float)*K*m)
		for l in range(K*m):
			tmp[l] = 0.0
		
		# Cluster haplotypes
		for i in prange(n):
			Z[w,i] = 0
			m_val = m
			for k in range(K):
				# Distances
				if N[k] > 0: # Safety measure
					dist = 0
					for j in range(m):
						if X[i,j] != M[k,j]:
							dist = dist + 1
				else:
					dist = m
				# Assignment
				if dist < m_val:
					Z[w,i] = k # Cluster assignment
					m_val = dist
			c[i] = m_val # Per individual cost

			# Add individual contributions to thread local array
			for j in range(m):
				tmp[Z[w,i]*m + j] += <float>X[i,j]

		# Construct new centroids (unnormalized)
		with gil:
			for k2 in range(K):
				for j2 in range(m):
					C[k2,j2] += tmp[k2*m + j2]
		
		# Deallocate thread-local arrays
		PyMem_RawFree(tmp)

# Count size of clusters
cpdef void countN(unsigned char[:,::1] Z, int[::1] N, int K, int w):
	cdef int n = Z.shape[1]
	cdef int i, k
	for k in range(K):
		N[k] = 0
	for i in range(n):
		N[Z[w,i]] += 1

# Find non-zero cluster with least assignments
cpdef void findZero(int[::1] N, int n, int thr, int K):
	cdef int k, Nk
	cdef int minI = 0
	cdef int minN = n
	for k in range(K):
		Nk = N[k]
		if Nk > 0:
			if Nk < minN:
				minI = k
				minN = Nk
	if minN <= thr:
		N[minI] = 0

# Fix index of medians
cpdef void medianFix(signed char[:,::1] M, unsigned char[:,::1] Z, \
		int[::1] N, int K, int w):
	cdef int m = M.shape[1]
	cdef int n = Z.shape[1]
	cdef int i, j, k	
	cdef int c = 0
	for k in range(K):
		if N[k] > 0:
			if k != c:
				for j in range(m):
					M[c,j] = M[k,j]
					M[k,j] = -9
				for i in range(n):
					if Z[w,i] == k:
						Z[w,i] = c
				N[c] = N[k]
				N[k] = 0
			c += 1
		else:
			for j in range(m):
				M[k,j] = -9

# Generate haplotype log-likelihoods (Bernoulli)
cpdef void loglikeHaplo(float[:,:,::1] L, unsigned char[:,::1] X, float[:,::1] C, \
		unsigned char[:,::1] Z, int[::1] N, int K, int w, int t):
	cdef int n = X.shape[0]
	cdef int m = X.shape[1]
	cdef int i, j, k
	cdef float p
	for i in range(n):
		for j in range(m):
			C[Z[w,i],j] += <float>X[i,j]/<float>N[Z[w,i]]
	with nogil:
		for i in prange(n, num_threads=t):
			for k in range(K):
				L[w, i, k] = 0.0
				for j in range(m):
					p = min(max(C[k,j], 1e-6), 1-(1e-6))
					L[w, i, k] += X[i,j]*log(p) + (1.0 - X[i,j])*log(1.0 - p)
