# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
import numpy as np
cimport numpy as np
from cython.parallel import prange
from libc.math cimport log

##### hapla - haplotype clustering #####
### Inline functions
# Calculate Hamming distance
cdef inline int hammingDist(const unsigned char* X, const unsigned char* M, \
		const int m) noexcept nogil:
	cdef:
		int dist = 0
		int j
	for j in range(m):
		dist += <int>(X[j] ^ M[j])
	return dist

# Calculate Hamming distance and change assignments
cdef inline int hammingCheck(const unsigned char* z_vec, unsigned char* z_pre, \
		const int n) noexcept nogil:
	cdef:
		int dist = 0
		int i
	for i in range(n):
		dist += <int>(z_vec[i] ^ z_pre[i])
		z_pre[i] = z_vec[i]
	return dist

# Add haplotype contribution to frequency vector
cdef inline void addHaplo(const unsigned char* X, float* C, const int u, \
		const int m) noexcept nogil:
	cdef int j
	for j in range(m):
		C[j] += <float>(X[j]*u)

# Update cluster information
cdef inline void updateClust(const unsigned char* X, unsigned char* M, float* A, \
		float* B, const int u, const int m) noexcept nogil:
	cdef:
		int j
		unsigned char x
	for j in range(m):
		x = X[j]
		M[j] = x
		A[j] = <double>(x*u)
		B[j] -= A[j]

### Standard functions
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
				C[k,j] = 0.0

# Compute distances, cluster assignment and prepare for next loop
cpdef void clusterAssignment(const unsigned char[:,::1] X, \
		const unsigned char[:,::1] M, unsigned char[::1] z_vec, int[::1] c_vec, \
		const int[::1] n_vec, const int[::1] u_vec, float[:,:,::1] C_thr, \
		int[:,::1] N_thr, const int[:,::1] I_thr, const int K, const int t) \
		noexcept nogil:
	cdef:
		int m = X.shape[1]
		int c, d, i, j, k, u, z, thr
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
					elif d == c: # Assign to largest cluster in ties
						if n_vec[k] > n_vec[z]:
							z = k
							c = d
			z_vec[i] = z
			c_vec[i] = c

			# Add individual contributions to temporary arrays
			u = u_vec[i]
			N_thr[thr,z] += u
			addHaplo(&X[i,0], &C_thr[thr,z,0], u, m)

# Update contributions and counts
cpdef void updateArrays(const float[:,:,::1] C_thr, float[:,::1] C, \
		const int[:,::1] N_thr, int[::1] n_vec, const int K, \
		const int t) noexcept nogil:
	cdef:
		int m = C.shape[1]
		int k, j
	for k in range(K):
		n_vec[k] = 0
	for thr in range(t):
		for k in range(K):
			n_vec[k] += N_thr[thr,k]
			for j in range(m):
				C[k,j] += C_thr[thr,k,j]

# Check and generate new cluster
cpdef int checkCluster(const unsigned char[:,::1] X, unsigned char[:,::1] M, \
		float[:,::1] C, unsigned char[::1] z_vec, const int[::1] c_vec, \
		int[::1] n_vec, const int[::1] u_vec, const float c_lim, const int K) \
		noexcept nogil:
	cdef:
		int n = X.shape[0]
		int m = X.shape[1]
		int L = M.shape[0]
		int c_arg = 0
		int c_max = c_vec[0]
		int i, j, k, u, z
	for i in range(1, n): # Find extreme point
		if c_vec[i] > c_max:
			c_arg = i
			c_max = c_vec[i]
	if (c_max > c_lim) & (K < L):
		u = u_vec[c_arg]
		z = <int>z_vec[c_arg]
		updateClust(&X[c_arg,0], &M[K,0], &C[K,0], &C[z,0], u, m)
		n_vec[K] = u
		n_vec[z] -= u
		z_vec[c_arg] = K
		return 1
	else:
		return 0

# Generate new cluster from no check
cpdef void genCluster(const unsigned char[:,::1] X, unsigned char[:,::1] M, \
		float[:,::1] C, unsigned char[::1] z_vec, const int[::1] c_vec, \
		int[::1] n_vec, const int[::1] u_vec, const int K) noexcept nogil:
	cdef:
		int n = X.shape[0]
		int m = X.shape[1]
		int c_arg = 0
		int c_max = c_vec[0]
		int i, j, k, u, z
	for i in range(1, n): # Find extreme point
		if c_vec[i] > c_max:
			c_arg = i
			c_max = c_vec[i]
	u = u_vec[c_arg]
	z = z_vec[c_arg]
	updateClust(&X[c_arg,0], &M[K,0], &C[K,0], &C[z,0], u, m)
	n_vec[K] = u
	n_vec[z] -= u
	z_vec[c_arg] = K
	
# Convergence check through hamming distance
cpdef int countDist(const unsigned char[::1] z_vec, unsigned char[::1] z_tmp) \
		noexcept nogil:
	cdef int n = z_vec.shape[0]
	return hammingCheck(&z_vec[0], &z_tmp[0], n)

# Find non-zero cluster with least assignments
cpdef int findZero(int[::1] n_vec, const int n, const int mac, const int K) \
		noexcept nogil:
	cdef:
		int k
		int minI = 0
		int minN = n + 1
	for k in range(K):
		if n_vec[k] > 0:
			if n_vec[k] <= minN:
				minI = k
				minN = n_vec[k]
	if minN < mac:
		n_vec[minI] = 0
	return minN

# Fix index of medians
cpdef void medianFix(unsigned char[:,::1] M, unsigned char[::1] z_vec, \
		int[::1] n_vec, const int K, const int U) noexcept nogil:
	cdef:
		int m = M.shape[1]
		int i, j, k
		int c = 0
	for k in range(K):
		if n_vec[k] > 0:
			if k != c:
				for j in range(m):
					M[c,j] = M[k,j]
				for i in range(U):
					if z_vec[i] == k:
						z_vec[i] = c
				n_vec[c] = n_vec[k]
				n_vec[k] = 0
			c += 1

# Fix haplotype cluster assignments
cpdef void assignFix(unsigned char[:,::1] Z, unsigned char[::1] z_vec, \
		int[::1] p_vec, int[::1] d_vec, const int w) noexcept nogil:
	cdef:
		int n = Z.shape[1]
		int i, z
		int u = 0
	for i in range(n):
		if d_vec[i] != 0:
			z = z_vec[u]
			u += 1
		Z[w,p_vec[i]] = z

# Reset arrays for next iteration
cpdef void resetArrays(float[:,:,::1] C_thr, int[:,::1] N_thr, int[::1] c_vec, \
		int[::1] p_vec, int[::1] d_vec, int[::1] u_vec, const int t) noexcept nogil:
	cdef:
		int K = C_thr.shape[1]
		int m = C_thr.shape[2]
		int n = c_vec.shape[0]
		int i, j, k, thr
	for thr in range(t):
		for k in range(K):
			N_thr[thr,k] = 0
			for j in range(m):
				C_thr[thr,k,j] = 0.0
	for i in range(n):
		c_vec[i] = 0
		p_vec[i] = i
		d_vec[i] = 0
		u_vec[i] = 0
