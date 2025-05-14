# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
cimport numpy as np
cimport openmp as omp
from cython.parallel import parallel, prange
from libc.math cimport fmaxf, fminf, log, log1p
from libc.stdint cimport uint8_t, uint32_t
from libc.stdlib cimport calloc, free

cdef float PRO_MIN = 1e-5
cdef float PRO_MAX = 1.0-(1e-5)

##### hapla - haplotype clustering #####
### Inline functions
# Inner function for marginal medians and variance
cdef inline void _marginal(
		uint8_t* r, uint32_t* c, const uint32_t n, const uint32_t M
	) noexcept nogil:
	cdef:
		size_t j
	for j in range(M):
		r[j] = c[j] > n
		c[j] = 0

# Calculate Hamming distance
cdef inline uint32_t _hammingDist(
		const uint8_t* h, const uint8_t* r, const uint32_t M
	) noexcept nogil:
	cdef:
		uint32_t dist = 0
		size_t j
	for j in range(M):
		if h[j] != r[j]:
			dist += 1
	return dist

# Calculate Hamming distance and change assignments
cdef inline uint32_t _hammingCheck(
		const uint8_t* z_vec, uint8_t* z_pre, const uint32_t U
	) noexcept nogil:
	cdef:
		uint32_t dist = 0
		size_t i
	for i in range(U):
		if z_vec[i] != z_pre[i]:
			z_pre[i] = z_vec[i]
			dist += 1
	return dist

# Add haplotype contribution to frequency vector
cdef inline void _addHaplo(
		const uint8_t* h, uint32_t* c, const uint32_t u, const uint32_t M
	) noexcept nogil:
	cdef size_t j
	for j in range(M):
		if h[j]:
			c[j] += u

# Update cluster information
cdef inline void _updateClust(
		const uint8_t* X, uint8_t* R, uint32_t* A, uint32_t* B, const uint32_t u, const uint32_t M
	) noexcept nogil:
	cdef:
		size_t j
	for j in range(M):
		R[j] = X[j]
		if X[j]:
			A[j] = u
			B[j] -= u
		else:
			A[j] = 0

# Truncate parameters to domain
cdef inline float _project(
		const float a
	) noexcept nogil:
	return fminf(fmaxf(a, PRO_MIN), PRO_MAX)

# Estimate log-likelihood between cluster medians based on frequencies
cdef inline float _logLike(
		const uint8_t* r, const uint32_t* c, const float n, const uint32_t M
	) noexcept nogil:
	cdef:
		float s = 0.0
		float d, p
		size_t j
	for j in range(M):
		d = <float>r[j]
		p = _project(<float>c[j]*n)
		s += d*log(p) + (1.0 - d)*log1p(-p)
	return s

### Standard functions
# Compute marginal medians and cluster variances
cpdef void marginalMedians(
		uint8_t[:,::1] R, uint32_t[:,::1] C, const uint32_t[::1] n_vec, const uint32_t K
	) noexcept nogil:
	cdef:
		uint32_t M = R.shape[1]
		size_t k
	for k in range(K):
		if n_vec[k] > 0:
			_marginal(&R[k,0], &C[k,0], n_vec[k]/2, M)

# Compute distances and perform cluster assignment
cpdef void assignClust(
		uint8_t[:,::1] X, const uint8_t[:,::1] R, uint32_t[:,::1] C, uint8_t[::1] z_vec, uint32_t[::1] c_vec,
		const uint32_t[::1] n_vec, uint32_t[::1] n_tmp, const uint32_t[::1] u_vec, const uint32_t U, const uint32_t K
	) noexcept nogil:
	cdef:
		uint8_t* h
		uint32_t M = X.shape[1]
		uint32_t c, d, u
		uint32_t* C_thr
		uint32_t* n_thr
		size_t i, k, x, y, z
		omp.omp_lock_t mutex
	omp.omp_init_lock(&mutex)
	with nogil, parallel():
		C_thr = <uint32_t*>calloc(K*M, sizeof(uint32_t))
		n_thr = <uint32_t*>calloc(K, sizeof(uint32_t))
		for i in prange(U):
			h = &X[i,0]
			z = 0
			c = M+1
			for k in range(K):
				if n_vec[k] > 0:
					d = _hammingDist(h, &R[k,0], M)
					if d <= c:
						z = k
						c = d
			z_vec[i] = <uint8_t>z
			c_vec[i] = c

			# Add individual contributions to temporary arrays
			u = u_vec[i]
			n_thr[z] += u
			_addHaplo(h, &C_thr[z*M], u, M)
		
		# omp critical
		omp.omp_set_lock(&mutex)
		for x in range(K):
			if n_vec[x] > 0:
				n_tmp[x] += n_thr[x]
				for y in range(M):
					C[x,y] += C_thr[x*M + y]
		omp.omp_unset_lock(&mutex)
		free(C_thr)
		free(n_thr)
	omp.omp_destroy_lock(&mutex)

# Copy and reset size of clusters
cpdef void updateN(
		uint32_t[::1] n_vec, uint32_t[::1] n_tmp, const uint32_t K
	) noexcept nogil:
	cdef size_t k
	for k in range(K):
		n_vec[k] = n_tmp[k]
		n_tmp[k] = 0

# Check and generate new cluster
cpdef uint32_t checkClust(
		const uint8_t[:,::1] X, uint8_t[:,::1] R, uint32_t[:,::1] C, uint8_t[::1] z_vec, const uint32_t[::1] c_vec,
		uint32_t[::1] n_vec, const uint32_t[::1] u_vec, const uint32_t c_lim, const uint32_t U, const uint32_t K
	) noexcept nogil:
	cdef:
		uint32_t M = X.shape[1]
		uint32_t L = R.shape[0]
		uint32_t c_max = c_vec[0]
		uint32_t u
		size_t c_arg = 0
		size_t i, z
	for i in range(1, U): # Find extreme point
		if c_vec[i] > c_max:
			c_arg = i
			c_max = c_vec[i]
	if c_max >= c_lim and K < L:
		u = u_vec[c_arg]
		z = <size_t>z_vec[c_arg]
		_updateClust(&X[c_arg,0], &R[K,0], &C[K,0], &C[z,0], u, M)
		n_vec[K] = u
		n_vec[z] -= u
		z_vec[c_arg] = K
		return 1
	else:
		return 0

# Generate new cluster from no check
cpdef void genClust(
		const uint8_t[:,::1] X, uint8_t[:,::1] R, uint32_t[:,::1] C, uint8_t[::1] z_vec, const uint32_t[::1] c_vec,
		uint32_t[::1] n_vec, const uint32_t[::1] u_vec, const uint32_t U, const uint32_t K
	) noexcept nogil:
	cdef:
		uint32_t M = X.shape[1]
		uint32_t c_max = c_vec[0]
		uint32_t u
		size_t c_arg = 0
		size_t i, z
	for i in range(1, U): # Find extreme point
		if c_vec[i] > c_max:
			c_arg = i
			c_max = c_vec[i]
	u = u_vec[c_arg]
	z = <size_t>z_vec[c_arg]
	_updateClust(&X[c_arg,0], &R[K,0], &C[K,0], &C[z,0], u, M)
	n_vec[K] = u
	n_vec[z] -= u
	z_vec[c_arg] = K
	
# Convergence check through hamming distance
cpdef uint32_t countDist(
		const uint8_t[::1] z_vec, uint8_t[::1] z_tmp, const uint32_t U
	) noexcept nogil:
	return _hammingCheck(&z_vec[0], &z_tmp[0], U)

# Find non-zero cluster with least assignments
cpdef uint32_t findZero(
		uint32_t[::1] n_vec, const uint32_t N, const uint32_t mac, const uint32_t K
	) noexcept nogil:
	cdef:
		uint32_t minI = K-1
		uint32_t minN = N+1
		size_t k
	for k in range(K, -1, -1):
		if n_vec[k] > 0 and n_vec[k] < minN:
			minI = k
			minN = n_vec[k]
			if minN == 1:
				break
	if minN < mac:
		n_vec[minI] = 0
	return minN

# Fix index of medians
cpdef void medianFix(
		uint8_t[:,::1] R, uint32_t[:,::1] C, uint8_t[::1] z_vec, uint32_t[::1] n_vec, const uint32_t K, 
		const uint32_t U
	) noexcept nogil:
	cdef:
		uint32_t M = R.shape[1]
		size_t c = 0
		size_t i, j, k
	for k in range(K):
		if n_vec[k] > 0:
			if k != c:
				for j in range(M):
					R[c,j] = R[k,j]
					C[c,j] = C[k,j]
				for i in range(U):
					if z_vec[i] == k:
						z_vec[i] = <uint8_t>c
				n_vec[c] = n_vec[k]
				n_vec[k] = 0
			c += 1

# Fix haplotype cluster assignments
cpdef void assignFix(
		uint8_t[:,::1] Z, const uint8_t[::1] z_vec, const uint32_t[::1] p_vec, const uint32_t[::1] d_vec, 
		const uint32_t w
	) noexcept nogil:
	cdef:
		uint8_t z = z_vec[0]
		uint32_t N = Z.shape[1]
		size_t l = <size_t>w
		size_t u = 0
		size_t i
	for i in range(N):
		if d_vec[i] != 0:
			z = z_vec[u]
			u += 1
		Z[l,<size_t>p_vec[i]] = z

# Reset arrays for next iteration
cpdef void resetArrays(
		uint32_t[::1] c_vec, uint32_t[::1] n_vec, uint32_t[::1] p_vec, uint32_t[::1] d_vec, 
		uint32_t[::1] u_vec
	) noexcept nogil:
	cdef:
		uint32_t N = c_vec.shape[0]
		uint32_t K = n_vec.shape[0]
		size_t i, k
	for k in range(K):
		n_vec[k] = 0
	for i in range(N):
		c_vec[i] = 0
		p_vec[i] = <uint32_t>i
		d_vec[i] = 0
		u_vec[i] = 0

# Estimate log-likelihoods between cluster medians
cpdef void estimateLoglike(
		uint8_t[:,::1] R, uint32_t[:,::1] C, float[:,::1] L, uint32_t[::1] n_vec, uint32_t K
	) noexcept nogil:
	cdef:
		uint32_t M = R.shape[1]
		size_t k1, k2
	for k1 in range(K):
		for k2 in range(K):
			L[k1,k2] = _logLike(&R[k1,0], &C[k2,0], 1.0/<float>n_vec[k2], M)
