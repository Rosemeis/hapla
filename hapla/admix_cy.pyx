# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
import numpy as np
cimport numpy as np
from cpython.mem cimport PyMem_RawCalloc, PyMem_RawMalloc, PyMem_RawFree
from cython.parallel import parallel, prange
from libc.math cimport log, sqrt

##### hapla - ancestry estimation #####
# Create P matrix from array
cpdef void createP(double[:,:,::1] P, const unsigned char[::1] k_vec) \
		noexcept nogil:
	cdef:
		int W = P.shape[0]
		int K = P.shape[1]
		int C = P.shape[2]
		int w, k, c
		double sumP
	for w in range(W):
		for k in range(K):
			sumP = 0.0
			for c in range(C):
				if c < k_vec[w]:
					sumP += P[w,k,c]
				else:
					P[w,k,c] = 0.0
			for c in range(k_vec[w]):
				P[w,k,c] /= sumP

# Update P and Q temp arrays
cpdef void updateP(const unsigned char[:,::1] Z, double[:,:,::1] P, \
		const double[:,::1] Q, double[:,::1] Q_new, const unsigned char[::1] k_vec, \
		const int N, const int t) noexcept nogil:
	cdef:
		int W = Z.shape[0]
		int n = Z.shape[1]
		int K = P.shape[1]
		int C = P.shape[2]
		int w, i, k, l, c, z, x, y
		double a, h, sumP
		double* P_thr
		double* Q_thr
	with nogil, parallel(num_threads=t):
		P_thr = <double*>PyMem_RawCalloc(K*C, sizeof(double))
		Q_thr = <double*>PyMem_RawCalloc((n//N)*K, sizeof(double))
		for w in prange(W):
			for i in range(n):
				l = i//N
				h = 0.0
				z = <int>Z[w,i]
				for k in range(K):
					h = h + P[w,k,z]*Q[l,k]
				h = 1.0/h
				for k in range(K):
					a = (P[w,k,z]*Q[l,k])*h
					P_thr[k*C+z] = P_thr[k*C+z] + a
					Q_thr[l*K+k] = Q_thr[l*K+k] + a
			for k in range(K):
				sumP = 0.0
				for c in range(k_vec[w]):
					sumP = sumP + P_thr[k*C+c]
				for c in range(k_vec[w]):
					P[w,k,c] = P_thr[k*C+c]/sumP
					P[w,k,c] = min(max(P[w,k,c], 1e-5), 1-(1e-5))
					P_thr[k*C+c] = 0.0
		with gil:
			for x in range(n//N):
				for y in range(K):
					Q_new[x,y] += Q_thr[x*K + y]
		PyMem_RawFree(P_thr)
		PyMem_RawFree(Q_thr)

# Accelerated update P and Q temp arrays
cpdef void accelP(const unsigned char[:,::1] Z, double[:,:,::1] P, \
		const double[:,::1] Q, double[:,::1] Q_new, double[:,:,::1] D, \
		const unsigned char[::1] k_vec, const int N, const int t) noexcept nogil:
	cdef:
		int W = Z.shape[0]
		int n = Z.shape[1]
		int K = P.shape[1]
		int C = P.shape[2]
		int w, i, k, l, c, z, x, y
		double a, h, P0, sumP
		double* P_thr
		double* Q_thr
	with nogil, parallel(num_threads=t):
		P_thr = <double*>PyMem_RawCalloc(K*C, sizeof(double))
		Q_thr = <double*>PyMem_RawCalloc((n//N)*K, sizeof(double))
		for w in prange(W):
			for i in range(n):
				l = i//N
				h = 0.0
				z = <int>Z[w,i]
				for k in range(K):
					h = h + P[w,k,z]*Q[l,k]
				h = 1.0/h
				for k in range(K):
					a = (P[w,k,z]*Q[l,k])*h
					P_thr[k*C+z] = P_thr[k*C+z] + a
					Q_thr[l*K+k] = Q_thr[l*K+k] + a
			for k in range(K):
				sumP = 0.0
				for c in range(k_vec[w]):
					sumP = sumP + P_thr[k*C+c]
				for c in range(k_vec[w]):
					P0 = P[w,k,c]
					P[w,k,c] = P_thr[k*C+c]/sumP
					P[w,k,c] = min(max(P[w,k,c], 1e-5), 1-(1e-5))
					D[w,k,c] = P[w,k,c] - P0
					P_thr[k*C+c] = 0.0
		with gil:
			for x in range(n//N):
				for y in range(K):
					Q_new[x,y] += Q_thr[x*K + y]
		PyMem_RawFree(P_thr)
		PyMem_RawFree(Q_thr)

# Accelerated jump for P (SQUAREM)
cpdef void alphaP(double[:,:,::1] P, const double[:,:,::1] P0, \
		const double[:,:,::1] D1, const double[:,:,::1] D2, double[:,:,::1] D3, \
		const unsigned char[::1] k_vec, const int t) noexcept nogil:
	cdef:
		int W = P.shape[0]
		int K = P.shape[1]
		int w, k, c
		double sum1 = 0.0
		double sum2 = 0.0
		double alpha, sumP
	for w in range(W):
		for k in range(K):
			for c in range(k_vec[w]):
				D3[w,k,c] = D2[w,k,c] - D1[w,k,c]
				sum1 += D1[w,k,c]*D1[w,k,c]
				sum2 += D3[w,k,c]*D3[w,k,c]
	alpha = max(1.0, sqrt(sum1)/sqrt(sum2))
	for w in prange(W, num_threads=t):
		for k in range(K):
			sumP = 0.0
			for c in range(k_vec[w]):
				P[w,k,c] = P0[w,k,c] + 2.0*alpha*D1[w,k,c] + alpha*alpha*D3[w,k,c]
				sumP = sumP + P[w,k,c]
			for c in range(k_vec[w]):
				P[w,k,c] /= sumP
				P[w,k,c] = min(max(P[w,k,c], 1e-5), 1-(1e-5))

# Update Q
cpdef void updateQ(double[:,::1] Q, double[:,::1] Q_new, const double S, \
		const int t) noexcept nogil:
	cdef:
		int n = Q.shape[0]
		int K = Q.shape[1]
		int i, k
		double sumQ
	for i in prange(n, num_threads=t):
		sumQ = 0.0
		for k in range(K):
			Q[i,k] = Q_new[i,k]/S
			Q[i,k] = min(max(Q[i,k], 1e-5), 1-(1e-5))
			Q_new[i,k] = 0.0
			sumQ = sumQ + Q[i,k]
		for k in range(K):
			Q[i,k] /= sumQ

# Accelerated update Q
cpdef void accelQ(double[:,::1] Q, double[:,::1] Q_new, double[:,::1] D, \
		const double S, const int t) noexcept nogil:
	cdef:
		int n = Q.shape[0]
		int K = Q.shape[1]
		int i, k
		double sumQ
		double* q
	with nogil, parallel(num_threads=t):
		q = <double*>PyMem_RawMalloc(sizeof(double)*K)
		for i in prange(n):
			sumQ = 0.0
			for k in range(K):
				q[k] = Q[i,k]
				Q[i,k] = Q_new[i,k]/S
				Q[i,k] = min(max(Q[i,k], 1e-5), 1-(1e-5))
				Q_new[i,k] = 0.0
				sumQ = sumQ + Q[i,k]
			for k in range(K):
				Q[i,k] /= sumQ
				D[i,k] = Q[i,k] - q[k]
		PyMem_RawFree(q)

# Accelerated jump for Q (SQUAREM)
cpdef void alphaQ(double[:,::1] Q, const double[:,::1] Q0, const double[:,::1] D1, \
		const double[:,::1] D2, double[:,::1] D3, const int t) noexcept nogil:
	cdef:
		int n = Q.shape[0]
		int K = Q.shape[1]
		int i, k
		double sum1 = 0.0
		double sum2 = 0.0
		double sumQ
		double alpha
	for i in range(n):
		for k in range(K):
			D3[i,k] = D2[i,k] - D1[i,k]
			sum1 += D1[i,k]*D1[i,k]
			sum2 += D3[i,k]*D3[i,k]
	alpha = max(1.0, sqrt(sum1)/sqrt(sum2))
	for i in prange(n, num_threads=t):
		sumQ = 0.0
		for k in range(K):
			Q[i,k] = Q0[i,k] + 2.0*alpha*D1[i,k] + alpha*alpha*D3[i,k]
			Q[i,k] = min(max(Q[i,k], 1e-5), 1-(1e-5))
			sumQ = sumQ + Q[i,k]
		for k in range(K):
			Q[i,k] /= sumQ

# Log-likelihood
cpdef void loglike(const unsigned char[:,::1] Z, const double[:,:,::1] P, \
		const double[:,::1] Q, double[::1] l_vec, const int N, const int t) \
		noexcept nogil:
	cdef:
		int W = Z.shape[0]
		int n = Z.shape[1]
		int K = P.shape[1]
		int w, i, k, l, z
		double h
	for w in prange(W, num_threads=t):
		l_vec[w] = 0.0
		for i in range(n):
			l = i//N
			z = <int>Z[w,i]
			h = 0.0
			for k in range(K):
				h = h + P[w,k,z]*Q[l,k]
			l_vec[w] += log(h)
