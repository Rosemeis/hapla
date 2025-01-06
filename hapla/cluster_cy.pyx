# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
import numpy as np
cimport numpy as np
from cython.parallel import prange
from libc.math cimport log, log1p

##### hapla - haplotype clustering #####
### Inline functions
# Calculate Hamming distance
cdef inline unsigned int hammingDist(const unsigned char* X, const unsigned char* R, \
		const size_t M) noexcept nogil:
	cdef:
		size_t j
		unsigned int dist = 0
	for j in range(M):
		dist += X[j]^R[j]
	return dist

# Calculate Hamming distance and change assignments
cdef inline unsigned int hammingCheck(const unsigned char* z_vec, unsigned char* z_pre, \
		const size_t N) noexcept nogil:
	cdef:
		size_t i
		unsigned int dist = 0
	for i in range(N):
		dist += z_vec[i]^z_pre[i]
		z_pre[i] = z_vec[i]
	return dist

# Add haplotype contribution to frequency vector
cdef inline void addHaplo(const unsigned char* X, float* C, const size_t u, \
		const size_t M) noexcept nogil:
	cdef size_t j
	for j in range(M):
		C[j] += <float>(X[j]*u)

# Update cluster information
cdef inline void updateClust(const unsigned char* X, unsigned char* R, float* A, \
		float* B, const size_t u, const size_t M) noexcept nogil:
	cdef:
		size_t j
		unsigned char x
	for j in range(M):
		x = X[j]
		R[j] = x
		A[j] = <double>(x*u)
		B[j] -= A[j]

### Standard functions
# Create marginal medians
cpdef void marginalMedians(unsigned char[:,::1] R, float[:,::1] C, \
		const unsigned int[::1] n_vec, const size_t K) noexcept nogil:
	cdef:
		size_t M = R.shape[1]
		size_t j, k
		float Nk
	for k in range(K):
		if n_vec[k] > 0:
			Nk = 1.0/<float>n_vec[k]
			for j in range(M):
				R[k,j] = <unsigned char>(C[k,j]*Nk > 0.5)
				C[k,j] = 0.0

# Compute distances, cluster assignment and prepare for next loop
cpdef void clusterAssignment(const unsigned char[:,::1] X, \
		const unsigned char[:,::1] R, unsigned char[::1] z_vec, \
		unsigned int[::1] c_vec, const unsigned int[::1] n_vec, \
		const unsigned int[::1] u_vec, float[:,:,::1] C_thr, \
		unsigned int[:,::1] N_thr, const unsigned int[:,::1] I_thr, \
		const size_t K, const size_t T) noexcept nogil:
	cdef:
		size_t M = X.shape[1]
		size_t c, d, i, j, k, t, u, z
	for t in prange(T, num_threads=T):
		# Reset thread-local arrays
		for k in range(K):
			N_thr[t,k] = 0
			for j in range(M):
				C_thr[t,k,j] = 0.0

		# Cluster haplotypes
		for i in range(I_thr[t,0], I_thr[t,1]):
			c = M + 1
			for k in range(K):
				if n_vec[k] > 0:
					d = hammingDist(&X[i,0], &R[k,0], M)
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
			N_thr[t,z] += u
			addHaplo(&X[i,0], &C_thr[t,z,0], u, M)

# Update contributions and counts from thread-local arrays
cpdef void updateArrays(const float[:,:,::1] C_thr, float[:,::1] C, 
		const unsigned int[:,::1] N_thr, unsigned int[::1] n_vec, const size_t K, \
		const size_t T) noexcept nogil:
	cdef:
		size_t M = C.shape[1]
		size_t k, j, t
	for k in range(K):
		n_vec[k] = N_thr[0,k]
		for j in range(M):
			C[k,j] = C_thr[0,k,j]
	for t in range(1, T):
		for k in range(K):
			n_vec[k] += N_thr[t,k]
			for j in range(M):
				C[k,j] += C_thr[t,k,j]

# Check and generate new cluster
cpdef unsigned int checkCluster(const unsigned char[:,::1] X, unsigned char[:,::1] R, \
		float[:,::1] C, unsigned char[::1] z_vec, const unsigned int[::1] c_vec, \
		unsigned int[::1] n_vec, const unsigned int[::1] u_vec, const float c_lim, \
		const size_t K) noexcept nogil:
	cdef:
		size_t N = X.shape[0]
		size_t M = X.shape[1]
		size_t L = R.shape[0]
		size_t c_arg = 0
		size_t c_max = c_vec[0]
		size_t i, j, k, u, z
	for i in range(1, N): # Find extreme point
		if c_vec[i] > c_max:
			c_arg = i
			c_max = c_vec[i]
	if (c_max > c_lim) & (K < L):
		u = u_vec[c_arg]
		z = z_vec[c_arg]
		updateClust(&X[c_arg,0], &R[K,0], &C[K,0], &C[z,0], u, M)
		n_vec[K] = u
		n_vec[z] -= u
		z_vec[c_arg] = K
		return 1
	else:
		return 0

# Generate new cluster from no check
cpdef void genCluster(const unsigned char[:,::1] X, unsigned char[:,::1] R, \
		float[:,::1] C, unsigned char[::1] z_vec, const unsigned int[::1] c_vec, \
		unsigned int[::1] n_vec, const unsigned int[::1] u_vec, const size_t K) \
		noexcept nogil:
	cdef:
		size_t N = X.shape[0]
		size_t M = X.shape[1]
		size_t c_arg = 0
		size_t c_max = c_vec[0]
		size_t i, j, k, u, z
	for i in range(1, N): # Find extreme point
		if c_vec[i] > c_max:
			c_arg = i
			c_max = c_vec[i]
	u = u_vec[c_arg]
	z = z_vec[c_arg]
	updateClust(&X[c_arg,0], &R[K,0], &C[K,0], &C[z,0], u, M)
	n_vec[K] = u
	n_vec[z] -= u
	z_vec[c_arg] = K
	
# Convergence check through hamming distance
cpdef unsigned int countDist(const unsigned char[::1] z_vec, unsigned char[::1] z_tmp) \
		noexcept nogil:
	cdef size_t N = z_vec.shape[0]
	return hammingCheck(&z_vec[0], &z_tmp[0], N)

# Find non-zero cluster with least assignments
cpdef unsigned int findZero(unsigned int[::1] n_vec, const size_t N, const size_t mac, \
		const size_t K) noexcept nogil:
	cdef:
		size_t k
		unsigned int minI = 0
		unsigned int minN = N + 1
	for k in range(K):
		if n_vec[k] > 0:
			if n_vec[k] <= minN:
				minI = k
				minN = n_vec[k]
	if minN < mac:
		n_vec[minI] = 0
	return minN

# Fix index of medians
cpdef void medianFix(unsigned char[:,::1] R, unsigned char[::1] z_vec, \
		unsigned int[::1] n_vec, const size_t K, const size_t U) noexcept nogil:
	cdef:
		size_t M = R.shape[1]
		size_t i, j, k
		size_t c = 0
	for k in range(K):
		if n_vec[k] > 0:
			if k != c:
				for j in range(M):
					R[c,j] = R[k,j]
				for i in range(U):
					if z_vec[i] == k:
						z_vec[i] = c
				n_vec[c] = n_vec[k]
				n_vec[k] = 0
			c += 1

# Fix haplotype cluster assignments
cpdef void assignFix(unsigned char[:,::1] Z, unsigned char[::1] z_vec, \
		unsigned int[::1] p_vec, unsigned int[::1] d_vec, const size_t w) noexcept nogil:
	cdef:
		size_t N = Z.shape[1]
		size_t i, z
		size_t u = 0
	for i in range(N):
		if d_vec[i] != 0:
			z = z_vec[u]
			u += 1
		Z[w,p_vec[i]] = z

# Reset arrays for next iteration
cpdef void resetArrays(float[:,:,::1] C_thr, unsigned int[:,::1] N_thr, \
		unsigned int[::1] c_vec, unsigned int[::1] p_vec, unsigned int[::1] d_vec, \
		unsigned int[::1] u_vec, const size_t T) noexcept nogil:
	cdef:
		size_t K = C_thr.shape[1]
		size_t M = C_thr.shape[2]
		size_t N = c_vec.shape[0]
		size_t i, j, k, t
	for t in range(T):
		for k in range(K):
			N_thr[t,k] = 0
			for j in range(M):
				C_thr[t,k,j] = 0.0
	for i in range(N):
		c_vec[i] = 0
		p_vec[i] = i
		d_vec[i] = 0
		u_vec[i] = 0
