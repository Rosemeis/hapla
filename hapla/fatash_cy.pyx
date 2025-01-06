# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
import numpy as np
cimport numpy as np
from cython.parallel import prange
from libc.math cimport log, exp

##### hapla - local ancestry inference #####
# Safe log-sum-exp for array
cdef inline double logsumexp(const double* vec, const size_t K) noexcept nogil:
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

# Calculate emission probabilities
cpdef void calcEmissions(double[:,:,::1] E, const unsigned char[:,::1] Z, \
		const double[::1] P, const unsigned char[::1] k_vec, \
		const unsigned int[::1] c_vec) noexcept nogil:
	cdef:
		size_t N = E.shape[0]
		size_t W = E.shape[1]
		size_t K = E.shape[2]
		size_t i, k, s, w, z, B
	for i in prange(N):
		for w in range(W):
			B = k_vec[w]
			s = c_vec[w]
			z = Z[i,w]
			for k in range(K):
				E[i,w,k] = log(P[s + B*k + z])

# Calculate transition probabilities - T[k1, k2] = P(Z_{w} = k1 | Z_{w-1} = k2)
cpdef void calcTransition(double[:,::1] T, const double[::1] Q, const double a) \
		noexcept nogil:
	cdef:
		size_t K = T.shape[0]
		size_t k1, k2
		double e = exp(-a)
	for k1 in range(K):
		for k2 in range(K):
			if k1 == k2:
				T[k1,k2] = log((1.0 - e)*Q[k1] + e)
			else:
				T[k1,k2] = log((1.0 - e)*Q[k1])

# Viterbi algorithm
cpdef void viterbi(const double[:,::1] E, const double[::1] Q_log, \
		const double[:,::1] T, double[:,::1] A, unsigned char[:,::1] I, \
		unsigned char[::1] V) noexcept nogil:
	cdef:
		size_t W = E.shape[0]
		size_t K = E.shape[1]
		size_t k, w, k1, k2
		double max_v
	# Basis step
	A[0,0] = E[0,0] + Q_log[0]
	for k in range(1, K):
		A[0,k] = E[0,k] + Q_log[k]
	
	# Loop through sequence
	for w in range(1, W):
		for k1 in range(K):
			A[w,k1] = A[w-1,0] + T[k1,0] + E[w,k1]
			I[w,k1] = 0
			for k2 in range(1, K):
				max_v = A[w-1,k2] + T[k1,k2] + E[w,k1]
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
cpdef void calcFwdBwd(const double[:,::1] E, double[:,::1] L, \
		const double[::1] Q_log, const double[:,::1] T, double[:,::1] A, \
		double[:,::1] B, double[::1] v) noexcept nogil:
	cdef:
		size_t W = E.shape[0]
		size_t K = E.shape[1]
		size_t k, w, k1, k2
		double l_fwd
	# Forward calculations
	for k in range(K):
		A[0,k] = E[0,k] + Q_log[k]
	for w in range(1, W):
		for k1 in range(K):
			for k2 in range(K):
				v[k2] = A[w-1,k2] + T[k1,k2]
			A[w,k1] = logsumexp(&v[0], K) + E[w,k1]
	
	# Forward log-likelihood
	for k in range(K):
		v[k] = A[W-1,k]
	l_fwd = logsumexp(&v[0], K)

	# Backward calculations
	for k in range(K):
		B[W-1,k] = 0.0
	for w in range(W-2, -1, -1):
		for k1 in range(K):
			for k2 in range(K):
				v[k2] = B[w+1,k2] + E[w+1,k2] + T[k2,k1]
			B[w,k1] = logsumexp(&v[0], K)

	# Compute posterior probabilities
	for w in range(W):
		for k in range(K):
			L[w,k] = exp(A[w,k] + B[w,k] - l_fwd)
