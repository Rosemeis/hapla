# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
cimport numpy as np
cimport openmp as omp
from cython.parallel import parallel, prange
from libc.math cimport fmaxf, fminf, log
from libc.stdint cimport uint8_t, uint32_t
from libc.stdlib cimport calloc, free

ctypedef uint8_t u8
ctypedef uint32_t u32
ctypedef float f32

cdef f32 PRO_MIN = 1e-5
cdef f32 PRO_MAX = 1.0 - (1e-5)
cdef inline f32 _fmaxf(f32 a, f32 b) noexcept nogil: return a if a > b else b
cdef inline f32 _fminf(f32 a, f32 b) noexcept nogil: return a if a < b else b
cdef inline f32 _clamp3(f32 a) noexcept nogil: return _fmaxf(PRO_MIN, _fminf(a, PRO_MAX))


##### hapla - haplotype clustering #####
### Inline functions
# Inner function for marginal medians and variance
cdef inline void _marginal(
		u8* r, u32* c, const u32 n, const Py_ssize_t M
	) noexcept nogil:
	cdef:
		size_t j
	for j in range(M):
		r[j] = 1 if c[j] > n else 0
		c[j] = 0

# Calculate Hamming distance
cdef inline u32 _hammingDist(
		const u8* h, const u8* r, const Py_ssize_t M
	) noexcept nogil:
	cdef:
		size_t j
		u32 dist = 0
	for j in range(M):
		dist += 1 if h[j] != r[j] else 0
	return dist

# Calculate Hamming distance and change assignments
cdef inline u32 _hammingCheck(
		const u8* z_vec, u8* z_pre, const Py_ssize_t U
	) noexcept nogil:
	cdef:
		size_t i
		u8 z1, z2
		u32 dist = 0
	for i in range(U):
		z1 = z_pre[i]
		z2 = z_vec[i]
		z_pre[i] = z2 if z1 != z2 else z1
		dist += 1 if z1 != z2 else 0
	return dist

# Add haplotype contribution to frequency vector
cdef inline void _addHaplo(
		const u8* h, u32* c, const u32 u, const Py_ssize_t M
	) noexcept nogil:
	cdef:
		size_t j
	for j in range(M):
		c[j] += u if h[j] == 1 else 0

# Update cluster information
cdef inline void _updateClust(
		const u8* X, u8* R, u32* A, u32* B, const u32 u, const Py_ssize_t M
	) noexcept nogil:
	cdef:
		size_t j
		u8 x
	for j in range(M):
		x = X[j]
		R[j] = x
		A[j] = u if x == 1 else 0
		B[j] -= u if x == 1 else 0

# Estimate log-likelihood between cluster medians based on frequencies
cdef inline f32 _logLike(
		const u8* r, const u32* c, const f32 n, const Py_ssize_t M
	) noexcept nogil:
	cdef:
		size_t j
		f32 s = 0.0
		f32 d, f, p
	for j in range(M):
		d = <f32>r[j]
		f = <f32>c[j]*n
		p = _clamp3(f)
		s += d*log(p) + (1.0 - d)*log(1.0 - p)
	return s


### Standard functions
# Compute marginal medians and cluster variances
cpdef void marginalMedians(
		u8[:,::1] R, u32[:,::1] C, const u32[::1] n_vec, const Py_ssize_t K
	) noexcept nogil:
	cdef:
		Py_ssize_t M = R.shape[1]
		size_t k
	for k in range(K):
		if n_vec[k] > 0:
			_marginal(&R[k,0], &C[k,0], n_vec[k]/2, M)

# Compute distances and perform cluster assignment
cpdef void assignClust(
		u8[:,::1] X, const u8[:,::1] R, u32[:,::1] C, u8[::1] z_vec, u32[::1] c_vec, const u32[::1] n_vec, 
		u32[::1] n_tmp, const u32[::1] u_vec, const Py_ssize_t U, const Py_ssize_t K
	) noexcept nogil:
	cdef:
		Py_ssize_t M = X.shape[1]
		size_t i, k, x, y, z
		u8* h
		u32 c, d, u
		u32* n_thr
		u32* C_thr
		omp.omp_lock_t mutex
	omp.omp_init_lock(&mutex)
	with nogil, parallel():
		n_thr = <u32*>calloc(K, sizeof(u32))
		C_thr = <u32*>calloc(K*M, sizeof(u32))
		for i in prange(U, schedule='guided'):
			h = &X[i,0]
			z = 0
			c = M + 1
			for k in range(K):
				if n_vec[k] > 0:
					d = _hammingDist(h, &R[k,0], M)
					if d <= c:
						z = k
						c = d
			z_vec[i] = z
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
		free(n_thr)
		free(C_thr)
	omp.omp_destroy_lock(&mutex)

# Copy and reset size of clusters
cpdef void updateN(
		u32[::1] n_vec, u32[::1] n_tmp, const Py_ssize_t K
	) noexcept nogil:
	cdef:
		size_t k
	for k in range(K):
		n_vec[k] = n_tmp[k]
		n_tmp[k] = 0

# Check and generate new cluster
cpdef u32 checkClust(
		const u8[:,::1] X, u8[:,::1] R, u32[:,::1] C, u8[::1] z_vec, const u32[::1] c_vec, u32[::1] n_vec, 
		const u32[::1] u_vec, const u32 c_lim, const Py_ssize_t U, const Py_ssize_t K
	) noexcept nogil:
	cdef:
		Py_ssize_t M = X.shape[1]
		Py_ssize_t L = R.shape[0]
		size_t c_arg = 0
		size_t i, z
		u32 c_max = c_vec[0]
		u32 u_max = u_vec[0]
		u32 c, u
	for i in range(1, U): # Find extreme point
		c = c_vec[i]
		u = u_vec[i]
		if c > c_max:
			c_arg = i
			c_max = c
			u_max = u
		elif c == c_max: # Choose largest for ties
			if u > u_max:
				c_arg = i
				c_max = c
				u_max = u
	if c_max >= c_lim and K < L:
		u = u_vec[c_arg]
		z = z_vec[c_arg]
		_updateClust(&X[c_arg,0], &R[K,0], &C[K,0], &C[z,0], u, M)
		n_vec[K] = u
		n_vec[z] -= u
		z_vec[c_arg] = K
		return 1
	else:
		return 0

# Generate new cluster from no check
cpdef void genClust(
		const u8[:,::1] X, u8[:,::1] R, u32[:,::1] C, u8[::1] z_vec, const u32[::1] c_vec, u32[::1] n_vec, 
		const u32[::1] u_vec, const Py_ssize_t U, const Py_ssize_t K
	) noexcept nogil:
	cdef:
		Py_ssize_t M = X.shape[1]
		size_t c_arg = 0
		size_t i, z
		u32 c_max = c_vec[0]
		u32 u_max = u_vec[0]
		u32 c, u
	for i in range(1, U): # Find extreme point
		c = c_vec[i]
		u = u_vec[i]
		if c > c_max:
			c_arg = i
			c_max = c
			u_max = u
		elif c == c_max: # Choose largest for ties
			if u > u_max:
				c_arg = i
				c_max = c
				u_max = u
	u = u_vec[c_arg]
	z = z_vec[c_arg]
	_updateClust(&X[c_arg,0], &R[K,0], &C[K,0], &C[z,0], u, M)
	n_vec[K] = u
	n_vec[z] -= u
	z_vec[c_arg] = K
	
# Convergence check through hamming distance
cpdef u32 countDist(
		const u8[::1] z_vec, u8[::1] z_tmp, const Py_ssize_t U
	) noexcept nogil:
	return _hammingCheck(&z_vec[0], &z_tmp[0], U)

# Find non-zero cluster with least assignments
cpdef u32 findZero(
		u32[::1] n_vec, const u32 N, const u32 mac, const Py_ssize_t K
	) noexcept nogil:
	cdef:
		size_t k
		u32 minI = K - 1
		u32 minN = N + 1
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
		u8[:,::1] R, u32[:,::1] C, u8[::1] z_vec, u32[::1] n_vec, const Py_ssize_t K, const Py_ssize_t U
	) noexcept nogil:
	cdef:
		Py_ssize_t M = R.shape[1]
		size_t c = 0
		size_t i, j, k
		u8 z
	for k in range(K):
		if n_vec[k] > 0:
			if k != c:
				for j in range(M):
					R[c,j] = R[k,j]
					C[c,j] = C[k,j]
				for i in range(U):
					z = z_vec[i]
					z_vec[i] = c if z == k else z
				n_vec[c] = n_vec[k]
				n_vec[k] = 0
			c += 1

# Fix haplotype cluster assignments
cpdef void assignFix(
		u8[:,::1] Z, const u8[::1] z_vec, const u32[::1] p_vec, const u32[::1] d_vec, const size_t w
	) noexcept nogil:
	cdef:
		Py_ssize_t N = Z.shape[1]
		size_t u = 0
		size_t i
		u8 z = z_vec[0]
	for i in range(N):
		if d_vec[i] != 0:
			z = z_vec[u]
			u += 1
		Z[w,p_vec[i]] = z

# Reset arrays for next iteration
cpdef void resetArrays(
		u32[::1] c_vec, u32[::1] n_vec, u32[::1] p_vec, u32[::1] d_vec, u32[::1] u_vec
	) noexcept nogil:
	cdef:
		Py_ssize_t N = c_vec.shape[0]
		Py_ssize_t K = n_vec.shape[0]
		size_t i, k
	for k in range(K):
		n_vec[k] = 0
	for i in range(N):
		c_vec[i] = 0
		p_vec[i] = i
		d_vec[i] = 0
		u_vec[i] = 0

# Estimate log-likelihoods between cluster medians
cpdef void estimateLoglike(
		u8[:,::1] R, u32[:,::1] C, f32[:,::1] L, u32[::1] n_vec, Py_ssize_t K
	) noexcept nogil:
	cdef:
		Py_ssize_t M = R.shape[1]
		size_t k1, k2
	for k1 in range(K):
		for k2 in range(K):
			L[k1,k2] = _logLike(&R[k1,0], &C[k2,0], 1.0/<f32>n_vec[k2], M)
