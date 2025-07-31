# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
cimport numpy as np
from cython.parallel import prange
from libc.math cimport exp, log
from libc.stdint cimport uint8_t, uint32_t

ctypedef uint8_t u8
ctypedef uint32_t u32
ctypedef float f32
ctypedef double f64


##### hapla - local ancestry inference #####
# Safe log-sum-exp for array
cdef inline f64 _logsumexp(
		const f64* vec, const Py_ssize_t K
	) noexcept nogil:
	cdef:
		size_t k
		f64 max_v = vec[0]
		f64 sum_v = 0.0
	for k in range(1, K):
		if vec[k] > max_v:
			max_v = vec[k]
	for k in range(K):
		sum_v += exp(vec[k] - max_v)
	return log(sum_v) + max_v

# Inner normalization function for likelihoods
cdef inline void _normLikes(
		f32* l, const u32 B
	) noexcept nogil:
	cdef:
		size_t z, c1, c2
		f32 sumC, tmpC
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
		f32[::1] L, const u32[::1] k_vec, const u32[::1] x_vec
	) noexcept nogil:
	cdef:
		Py_ssize_t W = k_vec.shape[0]
		size_t w
	for w in prange(W, schedule='guided'):
		_normLikes(&L[x_vec[w]], k_vec[w])

# Calculate emission probabilities using hard calls
cpdef void hardEmissions(
		 const u8[:,::1] Z, f64[:,:,::1] E, const f64[::1] P, const u32[::1] c_vec
	) noexcept nogil:
	cdef:
		Py_ssize_t N = E.shape[0]
		Py_ssize_t W = E.shape[1]
		Py_ssize_t K = E.shape[2]
		size_t i, k, w
	for i in prange(N, schedule='guided'):
		for w in range(W):
			p = &P[c_vec[w] + Z[i,w]*K]
			for k in range(K):
				E[i,w,k] = log(p[k])

# Calculate emission probabilities using cluster probabilities
cpdef void softEmissions(
		const u8[:,::1] Z, f64[:,:,::1] E, f64[::1] P, f32[::1] L, const u32[::1] k_vec, const u32[::1] c_vec, 
		const u32[::1] x_vec
	) noexcept nogil:
	cdef:
		Py_ssize_t N = E.shape[0]
		Py_ssize_t W = E.shape[1]
		Py_ssize_t K = E.shape[2]
		Py_ssize_t B
		size_t c, i, k, s, w, x, z
		f32* l
		f64* p
	for i in prange(N, schedule='guided'):
		for w in range(W):
			s = c_vec[w]
			x = x_vec[w]
			B = k_vec[w]
			z = Z[i,w]
			l = &L[x + z*B]
			p = &P[c_vec[w] + z*K]
			for k in range(K):
				for c in range(B):
					E[i,w,k] += <f64>l[c]*p[k]
				E[i,w,k] = log(E[i,w,k])

# Calculate transition probabilities - T[k2, k1] = P(Z_{w} = k1 | Z_{w - 1} = k2)
cpdef void calcTransition(
		f64[:,::1] T, const f64[::1] Q, const f64 a
	) noexcept nogil:
	cdef:
		Py_ssize_t K = T.shape[0]
		size_t k1, k2
		f64 e = exp(-a)
	for k2 in range(K):
		for k1 in range(K):
			if k2 == k1:
				T[k2,k1] = log((1.0 - e)*Q[k1] + e)
			else:
				T[k2,k1] = log((1.0 - e)*Q[k1])

# Viterbi algorithm
cpdef void viterbi(
		const f64[:,::1] E, const f64[::1] Q_log, const f64[:,::1] T, f64[:,::1] A, u8[:,::1] I, u8[::1] V
	) noexcept nogil:
	cdef:
		Py_ssize_t W = E.shape[0]
		Py_ssize_t K = E.shape[1]
		size_t k, w, k1, k2
		f64 max_v
	# Loop through sequence
	for k in range(K):
		A[0,k] = E[0,k] + Q_log[k]
	for w in range(1, W):
		for k1 in range(K):
			A[w,k1] = A[w - 1,0] + T[0,k1]
			I[w,k1] = 0
			for k2 in range(1, K):
				max_v = A[w - 1,k2] + T[k2,k1]
				if max_v > A[w,k1]:
					A[w,k1] = max_v
					I[w,k1] = k2
			A[w,k1] += E[w,k1]
	
	# Decode path
	V[W - 1] = 0
	max_v = A[W - 1,0]
	for k in range(1, K):
		if A[W - 1,k] > max_v:
			V[W - 1] = k
			max_v = A[W - 1,k]
	for w in range(W - 2, -1, -1):
		V[w] = I[w + 1,V[w + 1]]

# Forward-backward algorithm
cpdef void calcFwdBwd(
		const f64[:,::1] E, f64[:,::1] L, const f64[::1] Q_log, const f64[:,::1] T, f64[:,::1] A, f64[:,::1] B, 
		f64[::1] v
	) noexcept nogil:
	cdef:
		Py_ssize_t W = E.shape[0]
		Py_ssize_t K = E.shape[1]
		size_t k, w, k1, k2
		f64 l_fwd
	# Forward calculations
	for k in range(K):
		A[0,k] = E[0,k] + Q_log[k]
	for w in range(1, W):
		for k1 in range(K):
			for k2 in range(K):
				v[k2] = A[w - 1,k2] + T[k2,k1]
			A[w,k1] = _logsumexp(&v[0], K) + E[w,k1]
	
	# Forward log-likelihood
	for k in range(K):
		v[k] = A[W - 1,k]
	l_fwd = _logsumexp(&v[0], K)

	# Backward calculations
	for k in range(K):
		B[W - 1,k] = 0.0
	for w in range(W - 2, -1, -1):
		for k1 in range(K):
			for k2 in range(K):
				v[k2] = B[w + 1,k2] + E[w + 1,k2] + T[k1,k2]
			B[w,k1] = _logsumexp(&v[0], K)

	# Compute posterior probabilities
	for w in range(W):
		for k in range(K):
			L[w,k] = exp(A[w,k] + B[w,k] - l_fwd)
