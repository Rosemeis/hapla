# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
import numpy as np
cimport numpy as np
from cython.parallel import prange
from libc.math cimport log

##### hapla - haplotype clustering #####
# Create marginal medians
cpdef void marginalMedians(unsigned char[:,::1] M, float[:,::1] C, \
		const int[::1] n_vec, const int K) noexcept nogil:
	cdef:
		int m = M.shape[1]
		int j, k
		float Nk
	for k in range(K):
		if n_vec[k] > 0:
			Nk = 1.0/(<float>n_vec[k])
			for j in range(m):
				M[k,j] = <unsigned char>(C[k,j]*Nk > 0.5)

# Calculate Hamming distance
cdef int hammingDist(const unsigned char* X, const unsigned char* M, \
		const int m) noexcept nogil:
	cdef:
		int dist = 0
		int j
	for j in range(m):
		dist += <int>(X[j] ^ M[j])
	return dist

# Add haplotype contribution to frequency vector
cdef void addHaplo(const unsigned char* X, float* C, const int m) noexcept nogil:
	cdef int j
	for j in range(m):
		C[j] += <float>X[j]

# Compute distances, cluster assignment and prepare for next loop
cpdef void clusterAssignment(const unsigned char[:,::1] X, \
		const unsigned char[:,::1] M, unsigned char[:,::1] Z, int[::1] c_vec, \
		const int[::1] n_vec, const int[:,::1] I_thr, int[:,::1] N_thr, \
		float[:,:,::1] C_thr, const int K, const int w, const int t) noexcept nogil:
	cdef:
		int n = X.shape[0]
		int m = X.shape[1]
		int c, d, i, j, k, z, thr
	for thr in prange(t, num_threads=t):
		# Reset thread-local arrays
		for k in range(K):
			N_thr[thr,k] = 0
			for j in range(m):
				C_thr[thr,k,j] = 0.0

		# Cluster haplotypes
		for i in range(I_thr[thr,0], I_thr[thr,1]):
			c = m + 1
			for k in range(K):
				if n_vec[k] > 0:
					d = hammingDist(&X[i,0], &M[k,0], m)
					if d < c:
						z = k
						c = d
			Z[w,i] = <unsigned char>z
			c_vec[i] = c

			# Add individual contributions to thread local arrays
			N_thr[thr,z] += 1
			addHaplo(&X[i,0], &C_thr[thr,z,0], m)

# Find non-zero cluster with least assignments
cpdef int findZero(int[::1] n_vec, const int n, const int mac, const int K) \
		noexcept nogil:
	cdef:
		int k
		int minI = 0
		int minN = n
	for k in range(K):
		if n_vec[k] > 0:
			if n_vec[k] <= minN:
				minI = k
				minN = n_vec[k]
	if minN < mac:
		n_vec[minI] = 0
	return minN

# Fix index of medians
cpdef void medianFix(unsigned char[:,::1] M, unsigned char[:,::1] Z, \
		int[::1] n_vec, const int K, const int w, const int t) noexcept nogil:
	cdef:
		int m = M.shape[1]
		int n = Z.shape[1]
		int i, j, k
		int c = 0
	for k in range(K):
		if n_vec[k] > 0:
			if k != c:
				for j in range(m):
					M[c,j] = M[k,j]
				for i in prange(n, num_threads=t):
					if Z[w,i] == k:
						Z[w,i] = c
				n_vec[c] = n_vec[k]
				n_vec[k] = 0
			c += 1

# Generate haplotype log-likelihoods (Bernoulli)
cpdef void loglikeHaplo(float[:,::1] L, const unsigned char[:,::1] X, float[:,::1] C, \
		const unsigned char[:,::1] Z, const int[::1] n_vec, const int K, const int w, \
		const int t) noexcept nogil:
	cdef:
		int n = X.shape[0]
		int m = X.shape[1]
		int i, j, k, z
		float c, p, s
	for i in range(n):
		z = Z[w,i]
		s = 1.0/(<float>n_vec[z])
		for j in range(m):
			C[z,j] += (<float>X[i,j])*z
	for i in prange(n, num_threads=t):
		for k in range(K):
			L[i,k] = 0.0
			for j in range(m):
				L[i,k] += X[i,j]*log(C[k,j]) + (1.0 - X[i,j])*log(1.0 - C[k,j])
