# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
cimport numpy as np
from cython.parallel import parallel, prange
from libc.stdint cimport uint8_t, uint32_t
from libc.stdlib cimport abort, calloc, free

ctypedef uint8_t u8
ctypedef uint32_t u32
ctypedef float f32


##### hapla - analyses on haplotype cluster assignments #####
### Inline functions
# Center function
cdef inline void _center(
		const u8* z, f32* x, const f32 u, const Py_ssize_t N
	) noexcept nogil:
	cdef:
		size_t i
	for i in range(N):
		x[i] = <f32>z[i] - u

# Standardize function
cdef inline void _standardize(
		const u8* z, f32* x, const f32 u, const f32 a, const Py_ssize_t N
	) noexcept nogil:
	cdef:
		size_t i
	for i in range(N):
		x[i] = (<f32>z[i] - u)*a

# Calculate Hamming distance
cdef inline u32 _hammingPred(
		const u8* h, const u8* r, const Py_ssize_t M
	) noexcept nogil:
	cdef:
		size_t j
		u32 dist = 0
	for j in range(M):
		if h[j] != r[j] and h[j] != 9: # Ignore missing
			dist += 1
	return dist

# Homozygous states for unphased genotype clustering
cdef inline void _homStates(
		const u8* x, u8* h1, u8* h2, const Py_ssize_t M
	) noexcept nogil:
	cdef:
		size_t j
	for j in range(M):
		if x[j] == 0:
			h1[j] = 0
			h2[j] = 0
		elif x[j] == 2:
			h1[j] = 1
			h2[j] = 1
		else: # Missing or heterozygous
			h1[j] = 9
			h2[j] = 9

# Heterozygous states for unphased genotype clustering
cdef inline void _hetStates(
		const u8* x, u8* h1, u8* h2, u8* r, const Py_ssize_t M
	) noexcept nogil:
	cdef:
		size_t j
	for j in range(M):
		if x[j] == 1:
			h1[j] = r[j]
			h2[j] = 1 - r[j]


### hapla struct
# Extract aggregated haplotype cluster counts
cpdef void haplotypeAggregate(
		u8[:,::1] Z, u8[:,::1] Z_agg, f32[::1] p, const u32[::1] k_vec, const u32[::1] c_vec
	) noexcept nogil:
	cdef:
		Py_ssize_t W = Z.shape[0]
		Py_ssize_t N = Z.shape[1]
		size_t c, i, l, s, w
		u8* z
		f32 d = 1.0/<f32>N
	for w in prange(W, schedule='static'):
		s = c_vec[w]
		for c in range(k_vec[w]):
			l = s + c
			z = &Z_agg[l,0]
			for i in range(N):
				z[i >> 1] += 1 if Z[w,i] == c else 0
				p[l] += 1.0 if Z[w,i] == c else 0.0
			p[l] *= d

# Estimate haplotype cluster frequencies
cpdef void estimateFreq(
		u8[:,::1] Z, f32[::1] p, const u32[::1] k_vec, const u32[::1] c_vec
	) noexcept nogil:
	cdef:
		Py_ssize_t W = Z.shape[0]
		Py_ssize_t N = Z.shape[1]
		size_t c, i, l, s, w
		u8* z
		f32 d = 1.0/<f32>N
	for w in prange(W, schedule='static'):
		s = c_vec[w]
		z = &Z[w,0]
		for c in range(k_vec[w]):
			l = s + c
			for i in range(N):
				p[l] += 1.0 if z[i] == c else 0.0
			p[l] *= d

# Center expanded batch haplotype cluster assignment matrix
cpdef void centerZ(
		u8[:,::1] Z_agg, f32[:,::1] X, const f32[::1] p, const int S
	) noexcept nogil:
	cdef:
		Py_ssize_t M = X.shape[0]
		Py_ssize_t N = X.shape[1]
		size_t j, l
		f32 u
	for j in prange(M, schedule='static'):
		l = S + j
		u = 2.0*p[l]
		_center(&Z_agg[l,0], &X[j,0], u, N)

# Standardize expanded chunk of haplotype cluster assignment matrix
cpdef void chunkZ(
		u8[:,::1] Z_agg, f32[:,::1] X, const f32[::1] p, const f32[::1] a
	) noexcept nogil:
	cdef:
		Py_ssize_t M = X.shape[0]
		Py_ssize_t N = X.shape[1]
		size_t j
		f32 d, u
	for j in prange(M, schedule='static'):
		u = 2.0*p[j]
		_standardize(&Z_agg[j,0], &X[j,0], u, a[j], N)

# Standardize condense chunk of haplotype cluster assignment matrix
cpdef void memoryC(
		u8[:,::1] Z, f32[:,::1] X, const f32[::1] p, const f32[::1] a, const u32[::1] k_vec, const u32[::1] c_vec
	) noexcept nogil:
	cdef:
		Py_ssize_t W = Z.shape[0]
		Py_ssize_t N = X.shape[1]
		size_t c, i, l, s, w
		u8* z
		f32 u, d
		f32* x
	for w in prange(W, schedule='static'):
		s = c_vec[w] - c_vec[0]
		z = &Z[w,0]
		for c in range(k_vec[w]):
			l = s + c
			u = p[l]
			d = a[l]
			x = &X[l,0]
			for i in range(N):
				x[i] = (1.0 - u)*d if z[2*i] == c else (0.0 - u)*d
				x[i] += (1.0 - u)*d if z[2*i + 1] == c else (0.0 - u)*d


### hapla admix
# Center batch haplotype cluster assignment matrix
cpdef void centerC(
		u8[:,::1] Z, f32[:,::1] X, const f32[::1] p, const u32[::1] k_vec, const u32[::1] c_vec
	) noexcept nogil:
	cdef:
		Py_ssize_t W = Z.shape[0]
		Py_ssize_t N = X.shape[1]
		size_t c, i, l, s, w
		u8* z
		f32 u
		f32* x
	for w in prange(W, schedule='static'):
		s = c_vec[w] - c_vec[0]
		z = &Z[w,0]
		for c in range(k_vec[w]):
			l = s + c
			u = p[l]
			x = &X[l,0]
			for i in range(N):
				x[i] = (1.0 - u) if z[2*i] == c else (0.0 - u)
				x[i] += (1.0 - u) if z[2*i + 1] == c else (0.0 - u)


### hapla predict
# Haplotype cluster assignment based on medians in phased genotypes
cpdef void predictCluster(
		u8[:,::1] X, const u8[:,::1] R, u8[::1] Z
	) noexcept nogil:
	cdef:
		Py_ssize_t N = X.shape[0]
		Py_ssize_t M = X.shape[1]
		Py_ssize_t K = R.shape[0]
		size_t i, k, z
		u8* h
		u32 c, d
	for i in prange(N, schedule='static'):
		h = &X[i,0]
		z = 0
		c = _hammingPred(h, &R[0,0], M)
		for k in range(1, K):
			d = _hammingPred(h, &R[k,0], M)
			if d <= c:
				z = k
				c = d
		Z[i] = z

# Haplotype cluster assignment based on medians in unphased genotypes
cpdef void genoCluster(
		u8[:,::1] X, u8[:,::1] R, u8[::1] Z
	) noexcept nogil:
	cdef:
		Py_ssize_t N = X.shape[0]
		Py_ssize_t M = X.shape[1]
		Py_ssize_t K = R.shape[0]
		size_t i, k1, k2, z1, z2
		u8* x
		u8* h1
		u8* h2
		f32 c, d1, d2
	with nogil, parallel():
		# Thread-local buffer allocation
		h1 = <u8*>calloc(M, sizeof(u8))
		if h1 is NULL:
			abort()
		h2 = <u8*>calloc(M, sizeof(u8))
		if h2 is NULL:
			abort()

		for i in prange(N, schedule='static'):
			z1 = 0
			z2 = 0
			c = M + 1

			# Loop all cluster combinations
			x = &X[i,0]
			_homStates(x, h1, h2, M)
			for k1 in range(K):
				_hetStates(x, h1, h2, &R[k1,0], M)
				d1 = <f32>_hammingPred(h1, &R[k1,0], M)
				for k2 in range(K):
					d2 = <f32>_hammingPred(h2, &R[k2,0], M)
					if (0.66*d1 + 0.33*d2) < c: # Magic weight
						z1 = k1
						z2 = k2
						c = 0.66*d1 + 0.33*d2

			# Cluster assignment
			Z[2*i] = z1
			Z[2*i + 1] = z2
		free(h1)
		free(h2)
