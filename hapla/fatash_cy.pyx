# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
cimport numpy as np
from cython.parallel import prange
from libc.math cimport exp, fmax, fmin, log
from libc.stdint cimport uint8_t, uint32_t

cdef double PRO_MIN = 1e-5
cdef double PRO_MAX = 1.0-(1e-5)

##### hapla - local ancestry inference #####
# Truncate parameters to domain
cdef inline double _project(
		const double a
	) noexcept nogil:
	return fmin(fmax(a, PRO_MIN), PRO_MAX)

# Safe log-sum-exp for array
cdef inline double _logsumexp(
		const double* vec, const size_t K
	) noexcept nogil:
	cdef:
		double max_v = vec[0]
		double sum_v = 0.0
		size_t k
	for k in range(1, K):
		if vec[k] > max_v:
			max_v = vec[k]
	for k in range(K):
		sum_v += exp(vec[k] - max_v)
	return log(sum_v) + max_v

# Inner normalization function for likelihoods
cdef inline void _normLikes(
		float* l, const uint32_t B
	) noexcept nogil:
	cdef:
		float sumC, tmpC
		size_t z, c1, c2
	for c1 in range(B):
		z = c1*B
		sumC = 0.0
		for c2 in range(B):
			tmpC = exp(l[z + c2])
			sumC += tmpC
			l[z + c2] = tmpC
		for c2 in range(B):
			l[z + c2] /= sumC

# Convert log-likes to normalized likes
cpdef void createLikes(
		float[::1] L, const uint8_t[::1] k_vec, const uint32_t[::1] x_vec
	) noexcept nogil:
	cdef:
		uint32_t W = k_vec.shape[0]
		size_t w
	for w in prange(W):
		_normLikes(&L[x_vec[w]], <uint32_t>k_vec[w])

# Calculate emission probabilities using hard calls
cpdef void hardEmissions(
		double[:,:,::1] E, const uint8_t[:,::1] Z, const double[::1] P, const uint8_t[::1] k_vec, 
		const uint32_t[::1] c_vec
	) noexcept nogil:
	cdef:
		uint32_t N = E.shape[0]
		uint32_t W = E.shape[1]
		uint32_t K = E.shape[2]
		size_t B, i, k, s, w, z
	for i in prange(N):
		for w in range(W):
			s = <size_t>c_vec[w]
			B = <size_t>k_vec[w]
			z = <size_t>Z[i,w]
			for k in range(K):
				E[i,w,k] = log(P[s + k*B + z])

# Calculate emission probabilities using cluster probabilities
cpdef void softEmissions(
		double[:,:,::1] E, const uint8_t[:,::1] Z, const double[::1] P, float[::1] L, const uint8_t[::1] k_vec, 
		const uint32_t[::1] c_vec, const uint32_t[::1] x_vec
	) noexcept nogil:
	cdef:
		uint32_t N = E.shape[0]
		uint32_t W = E.shape[1]
		uint32_t K = E.shape[2]
		double p
		float* l
		size_t B, c, i, k, s, w, x, z
	for i in prange(N):
		for w in range(W):
			s = <size_t>c_vec[w]
			x = <size_t>x_vec[w]
			B = <size_t>k_vec[w]
			z = <size_t>Z[i,w]
			l = &L[x + z*B]
			for k in range(K):
				p = P[s + k*B + z]
				for c in range(B):
					E[i,w,k] += <double>l[c]*p
				E[i,w,k] = log(E[i,w,k])

# Calculate distances between windows
cpdef void calcDistances(
		double[::1] d, const double[::1] m_vec
	) noexcept nogil:
	cdef:
		uint32_t W = d.shape[0]
		double d_div, d_max, d_min
		size_t w
	# Initial window distances
	d[1] = <double>(m_vec[1] - m_vec[0])
	d_max = d[1]
	d_min = d_max

	# Loop through windows
	for w in range(2, W):
		d[w] = <double>(m_vec[w] - m_vec[w-1])
		if d[w] > d_max:
			d_max = d[w]
		elif d[w] < d_min:
			d_min = d[w]

	# Min-max normalization
	d_div = d_max - d_min
	for w in range(1, W):
		d[w] = (d[w] - d_min)/d_div

# Calculate transition probabilities - T[w, k2, k1] = P(Z_{w} = k1 | Z_{w-1} = k2)
cpdef void calcTransition(
		double[:,:,::1] T, const double[::1] Q, const double[::1] d, const double a
	) noexcept nogil:
	cdef:
		uint32_t W = T.shape[0]
		uint32_t K = T.shape[1]
		double e, sumK, tmpK
		size_t w, k1, k2
	for w in range(1, W):
		e = exp(-a*d[w])
		for k2 in range(K):
			sumK = 0.0
			for k1 in range(K):
				if k2 == k1:
					tmpK = _project((1.0 - e)*Q[k1] + e)
				else:
					tmpK = _project((1.0 - e)*Q[k1])
				sumK += tmpK
				T[w,k2,k1] = tmpK
			for k1 in range(K):
				T[w,k2,k1] = log(T[w,k2,k1]/sumK)

# Viterbi algorithm
cpdef void viterbi(
		const double[:,::1] E, const double[::1] Q_log, const double[:,:,::1] T, double[:,::1] A, uint8_t[:,::1] I, 
		uint8_t[::1] V
	) noexcept nogil:
	cdef:
		uint32_t W = E.shape[0]
		uint32_t K = E.shape[1]
		double max_v
		size_t k, w, k1, k2
	# Loop through sequence
	for k in range(K):
		A[0,k] = E[0,k] + Q_log[k]
	for w in range(1, W):
		for k1 in range(K):
			A[w,k1] = A[w-1,0] + T[w,0,k1] + E[w,k1]
			I[w,k1] = 0
			for k2 in range(1, K):
				max_v = A[w-1,k2] + T[w,k2,k1] + E[w,k1]
				if max_v > A[w,k1]:
					A[w,k1] = max_v
					I[w,k1] = k2
	
	# Decode path
	V[W-1] = 0
	max_v = A[W-1,0]
	for k in range(1, K):
		if A[W-1,k] > max_v:
			V[W-1] = k
			max_v = A[W-1,k]
	for w in range(W-2, -1, -1):
		V[w] = I[w+1,V[w+1]]

# Forward-backward algorithm
cpdef void calcFwdBwd(
		const double[:,::1] E, double[:,::1] L, const double[::1] Q_log, const double[:,:,::1] T, double[:,::1] A,
		double[:,::1] B, double[::1] v
	) noexcept nogil:
	cdef:
		uint32_t W = E.shape[0]
		uint32_t K = E.shape[1]
		double l_fwd
		size_t k, w, k1, k2
	# Forward calculations
	for k in range(K):
		A[0,k] = E[0,k] + Q_log[k]
	for w in range(1, W):
		for k1 in range(K):
			for k2 in range(K):
				v[k2] = A[w-1,k2] + T[w,k2,k1]
			A[w,k1] = _logsumexp(&v[0], K) + E[w,k1]
	
	# Forward log-likelihood
	for k in range(K):
		v[k] = A[W-1,k]
	l_fwd = _logsumexp(&v[0], K)

	# Backward calculations
	for k in range(K):
		B[W-1,k] = 0.0
	for w in range(W-2, -1, -1):
		for k1 in range(K):
			for k2 in range(K):
				v[k2] = B[w+1,k2] + E[w+1,k2] + T[w+1,k1,k2]
			B[w,k1] = _logsumexp(&v[0], K)

	# Compute posterior probabilities
	for w in range(W):
		for k in range(K):
			L[w,k] = exp(A[w,k] + B[w,k] - l_fwd)
