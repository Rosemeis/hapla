# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
import numpy as np
cimport numpy as np
from cython.parallel import prange
from libc.math cimport log, exp

##### hapla - local ancestry inference #####
# Safe log-sum-exp for array
cdef inline double logsumexp(const double* vec, const int K) noexcept nogil:
	cdef:
		double max_v = vec[0]
		double sum_v = 0.0
	for k in range(1, K):
		if vec[k] > max_v:
			max_v = vec[k]
	for k in range(K):
		sum_v += exp(vec[k] - max_v)
	return log(sum_v) + max_v

# Calculate emission probabilities
cpdef void calcEmissions(const unsigned char[:,::1] Z, const double[:,:,::1] P, \
		double[:,:,::1] E, const int t) noexcept nogil:
	cdef:
		int n = E.shape[0]
		int W = E.shape[1]
		int K = P.shape[1]
		int i, k, w, z
	for i in prange(n, num_threads=t):
		for w in range(W):
			z = Z[i,w]
			for k in range(K):
				E[i,w,k] = log(P[w,k,z])

# Calculate transition probabilities - T[k1, k2] = P(Z_{w} = k1 | Z_{w-1} = k2)
cpdef void calcTransition(double[:,::1] T, const double[:,::1] Q, const int i, \
		const double a) noexcept nogil:
	cdef:
		int K = T.shape[0]
		int k1, k2
		double e = exp(-a)
	for k1 in range(K):
		for k2 in range(K):
			if k1 == k2:
				T[k1,k2] = log((1.0 - e)*Q[i,k1] + e)
			else:
				T[k1,k2] = log((1.0 - e)*Q[i,k1])

# Viterbi algorithm
cpdef void viterbi(const double[:,:,::1] E, const double[:,::1] Q_log, \
		const double[:,::1] T, double[:,::1] A, unsigned char[:,::1] I, \
		unsigned char[:,::1] V, const int i) noexcept nogil:
	cdef:
		int W = E.shape[1]
		int K = E.shape[2]
		int k, w, k1, k2
		double tmp1, tmp2
	# Basis step
	A[0,0] = E[i,0,0] + Q_log[i,0]
	I[0,0] = 0
	tmp1 = A[0,0]
	for k in range(1, K):
		A[0,k] = E[i,0,k] + Q_log[i,k]
		I[0,k] = 0
		if A[0,k] > tmp1:
			tmp1 = A[0,k]
	for k in range(K):
		A[0,k] -= tmp1
	
	# Loop through sequence
	for w in range(1, W):
		for k1 in range(K):
			A[w,k1] = A[w-1,0] + T[0,k1] + E[i,w,k1]
			I[w,k1] = 0
			for k2 in range(1, K):
				tmp2 = A[w-1,k2] + T[k1,k2] + E[i,w,k1]
				if tmp2 > A[w,k1]:
					A[w,k1] = tmp2
					I[w,k1] = k2
		tmp1 = A[w,0]
		for k in range(1, K):
			if A[w,k] > tmp1:
				tmp1 = A[w,k]
		for k in range(K):
			A[w,k] -= tmp1
	
	# Decode path
	V[i,W-1] = 0
	tmp1 = A[W-1,0]
	for k in range(1, K):
		if A[W-1,k] > tmp1:
			V[i,W-1] = k
			tmp1 = A[W-1,k]
	for w in range(W-2, -1, -1):
		V[i,w] = I[w+1,V[i,w+1]]

# Forward-backward algorithm with scaling
cpdef void calcFwdBwd(const double[:,:,::1] E, double[:,:,::1] L, \
		const double[:,::1] Q_log, const double[:,::1] T, double[:,::1] A, \
		double[:,::1] B, double[::1] c, double[::1] v, const int i) \
		noexcept nogil:
	cdef:
		int W = E.shape[1]
		int K = E.shape[2]
		int k, w, k1, k2
		double l, sumP
	# Forward
	for k in range(K):
		A[0,k] = E[i,0,k] + Q_log[i,k]
	c[0] = logsumexp(&A[0,0], K)
	for k in range(K):
		A[0,k] -= c[0]
	for w in range(1, W):
		for k1 in range(K):
			for k2 in range(K):
				v[k2] = A[w-1,k2] + T[k1,k2]
			A[w,k1] = logsumexp(&v[0], K) + E[i,w,k1]
		c[w] = logsumexp(&A[w,0], K)
		for k1 in range(K):
			A[w,k1] -= c[w]

	# Backward
	for k in range(K):
		B[W-1,k] = 0.0
	for w in range(W-2, -1, -1):
		for k1 in range(K):
			for k2 in range(K):
				v[k2] = B[w+1,k2] + E[i,w+1,k2] + T[k2,k1]
			B[w,k1] = logsumexp(&v[0], K)
			B[w,k1] -= c[w+1]

	# Compute posterior probabilities
	for w in range(W):
		for k in range(K):
			L[i,w,k] = exp(A[w,k] + B[w,k])
