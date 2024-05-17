# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
import numpy as np
cimport numpy as np
from cython.parallel import prange
from libc.math cimport log

##### hapla - haplotype clustering #####
# Create marginal medians
cpdef void marginalMedians(unsigned char[:,::1] M, float[:,::1] C, \
		const int[::1] N_vec, const int K) noexcept nogil:
	cdef:
		int m = M.shape[1]
		int j, k
		float Nk
	for k in range(K):
		if N_vec[k] > 0:
			Nk = 1.0/(<float>N_vec[k])
			for j in range(m):
				C[k,j] *= Nk
				M[k,j] = <unsigned char>(C[k,j] > 0.5)

# Compute distances, cluster assignment and prepare for next loop
cpdef void clusterAssignment(const unsigned char[:,::1] X, \
		const unsigned char[:,::1] M, unsigned char[:,::1] Z, int[::1] c_vec, \
		const int[::1] N_vec, const int[:,::1] I_thr, int[:,::1] N_thr, \
		float[:,:,::1] C_thr, const int K, const int w, const int t) noexcept nogil:
	cdef:
		int n = X.shape[0]
		int m = X.shape[1]
		int i, j, k, thr, dist
	for thr in prange(t, num_threads=t, schedule="static", chunksize=1):
		# Reset thread-local arrays
		for k in range(K):
			N_thr[thr,k] = 0
			for j in range(m):
				C_thr[thr,k,j] = 0.0

		# Cluster haplotypes
		for i in range(I_thr[thr,0], I_thr[thr,1]):
			c_vec[i] = m + 1
			for k in range(K):
				# Distances
				if N_vec[k] > 0:
					dist = 0
					for j in range(m):
						dist = dist + (X[i,j] ^ M[k,j])

					# Assignment
					if dist < c_vec[i]:
						Z[w,i] = k
						c_vec[i] = dist

			# Add individual contributions to thread local arrays
			N_thr[thr,Z[w,i]] += 1
			for j in range(m):
				C_thr[thr,Z[w,i],j] += X[i,j]

# Find non-zero cluster with least assignments
cpdef int findZero(int[::1] N_vec, const int n, const int mac, const int K) \
		noexcept nogil:
	cdef:
		int k
		int minI = 0
		int minN = n
	for k in range(K):
		if N_vec[k] > 0:
			if N_vec[k] <= minN:
				minI = k
				minN = N_vec[k]
	if minN < mac:
		N_vec[minI] = 0
	return minN

# Fix index of medians
cpdef void medianFix(unsigned char[:,::1] M, unsigned char[:,::1] Z, \
		int[::1] N_vec, const int K, const int w, const int t) noexcept nogil:
	cdef:
		int m = M.shape[1]
		int n = Z.shape[1]
		int i, j, k
		int c = 0
	for k in range(K):
		if N_vec[k] > 0:
			if k != c:
				for j in range(m):
					M[c,j] = M[k,j]
				for i in prange(n, num_threads=t):
					if Z[w,i] == k:
						Z[w,i] = c
				N_vec[c] = N_vec[k]
				N_vec[k] = 0
			c += 1

# Generate haplotype log-likelihoods (Bernoulli)
cpdef void loglikeHaplo(float[:,::1] L, const unsigned char[:,::1] X, float[:,::1] C, \
		const unsigned char[:,::1] Z, const int[::1] N_vec, const int K, const int w, \
		const int t) noexcept nogil:
	cdef:
		int n = X.shape[0]
		int m = X.shape[1]
		int i, j, k
		float p
	for i in range(n):
		for j in range(m):
			C[Z[w,i],j] += (<float>X[i,j])/(<float>N_vec[Z[w,i]])
	for i in prange(n, num_threads=t):
		for k in range(K):
			L[i,k] = 0.0
			for j in range(m):
				p = min(max(C[k,j], 1e-6), 1-(1e-6))
				L[i,k] += X[i,j]*log(p) + (1.0 - X[i,j])*log(1.0 - p)
