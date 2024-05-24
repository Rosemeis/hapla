# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
import numpy as np
cimport numpy as np
from cython.parallel import prange
from libc.math cimport log, exp

##### hapla - local ancestry inference #####
# Calculate emission probabilities
cpdef void calcEmissions(const unsigned char[:,::1] Z, const double[:,:,::1] P_chr, \
		double[:,:,::1] E, const int t) noexcept nogil:
	cdef:
		int n = E.shape[0]
		int W = E.shape[1]
		int K = P_chr.shape[2]
		int i, w, k
		unsigned char z
	for i in prange(n, num_threads=t):
		for w in range(W):
			z = Z[i,w]
			for k in range(K):
				E[i,w,k] = log(P_chr[w,z,k])

# Calculate transition probabilities
cpdef void calcTransition(double[:,::1] T, const double[:,::1] Q, const int i, \
		const double a) noexcept nogil:
	cdef:
		int K = T.shape[0]
		int k1, k2
		double e = exp(-a)
	for k1 in range(K):
		for k2 in range(K):
			if k1 == k2:
				T[k1,k2] = log((1.0 - e)*Q[i,k2] + e)
			else:
				T[k1,k2] = log((1.0 - e)*Q[i,k2])

# Safe log-sum-exp for array
cdef double logsumexp(const double[::1] vec, const int K) noexcept nogil:
	cdef:
		double max_v = vec[0]
		double sum_v = 0.0
	for k in range(1, K):
		if vec[k] > max_v:
			max_v = vec[k]
	for k in range(K):
		sum_v += exp(vec[k] - max_v)
	return log(sum_v) + max_v

# Calculate log-likehood in HMM
cpdef double loglikeFatash(double[:,:,::1] E, const double[:,::1] Q, \
		const double[:,::1] T, double[:,::1] A, double[::1] v, const int N, \
		const int i) noexcept nogil:
	cdef:
		int W = E.shape[1]
		int K = E.shape[2]
		int w, k, k1, k2
	for k in range(K):
		A[0,k] = E[i,0,k] + log(Q[i//N,k])
	for w in range(1, W):
		for k1 in range(K):
			for k2 in range(K):
				v[k2] = A[w-1,k2] + T[k2,k1]
			A[w,k1] = logsumexp(v, K) + E[i,w,k1]
	
	# Log-likelihood
	for k in range(K):
		v[k] = A[W-1,k]
	return logsumexp(v, K)

# Forward-backward algorithm
cpdef void calcFwdBwd(const double[:,:,::1] E, double[:,:,::1] L, \
		const double[:,::1] Q, const double[:,::1] T, double[:,::1] A, \
		double[:,::1] B, double[::1] v, const int N, const int i) noexcept nogil:
	cdef:
		int W = E.shape[1]
		int K = E.shape[2]
		int w, k, k1, k2
		double ll_fwd

	# Forward
	for k in range(K):
		A[0,k] = E[i,0,k] + log(Q[i//N,k])
	for w in range(1, W):
		for k1 in range(K):
			for k2 in range(K):
				v[k2] = A[w-1,k2] + T[k2,k1]
			A[w,k1] = logsumexp(v, K) + E[i,w,k1]

	# Log-likelihood forward
	for k in range(K):
		v[k] = A[W-1,k]
	ll_fwd = logsumexp(v, K)

	# Backward
	for k in range(K):
		B[W-1,k] = 0.0
	for w in range(W-2, -1, -1):
		for k1 in range(K):
			for k2 in range(K):
				v[k2] = B[w+1,k2] + E[i,w+1,k2] + T[k1,k2]
			B[w,k1] = logsumexp(v, K)

	# Update posterior
	for w in range(W):
		for k in range(K):
			L[i,w,k] = A[w,k] + B[w,k] - ll_fwd
