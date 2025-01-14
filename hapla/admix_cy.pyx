# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
import numpy as np
cimport numpy as np
from cython.parallel import parallel, prange
from libc.math cimport log
from libc.stdlib cimport calloc, free

##### hapla - ancestry estimation #####
# Inline functions
cdef inline double project(const double s) noexcept nogil:
	return min(max(s, 1e-5), 1-(1e-5))

cdef inline double computeH(const double* p, const double* q, const size_t z, \
		const size_t K, const size_t B) noexcept nogil:
	cdef:
		size_t k
		double h = 0.0
	for k in range(K):
		h += p[k*B + z]*q[k]
	return 1.0/h

cdef inline double computeL(const double* p, const double* q, const size_t z, \
		const size_t K, const size_t B) noexcept nogil:
	cdef:
		size_t k
		double h = 0.0
	for k in range(K):
		h += p[k*B + z]*q[k]
	return log(h)

cdef inline void innerJ(const double* p, const double* q, double* p_tmp, \
		double* q_thr, const double h, const size_t z, const size_t K, const size_t B) \
		noexcept nogil:
	cdef:
		size_t k
		double a
	for k in range(K):
		a = p[k*B + z]*q[k]*h
		p_tmp[k*B + z] += a
		q_thr[k] += a

cdef inline void outerP(double* p, double* p_tmp, const size_t K, const size_t B) \
		noexcept nogil:
	cdef:
		size_t c, k, s
		double sumA, sumB
	for k in range(K):
		s = k*B
		sumA = 0.0
		sumB = 0.0
		for c in range(B):
			p[s+c] = p_tmp[s+c]
			p_tmp[s+c] = 0.0
			sumA += p[s+c]
		sumA = 1.0/sumA
		for c in range(B):
			p[s+c] = project(p[s+c]*sumA)
			sumB += p[s+c]
		sumB = 1.0/sumB
		for c in range(B):
			p[s+c] *= sumB

cdef inline void outerQ(double* q, double* q_tmp, const double S, const size_t K) \
		noexcept nogil:
	cdef:
		size_t k
		double sumQ = 0.0
	for k in range(K):
		q[k] = project(q_tmp[k]*S)
		q_tmp[k] = 0.0
		sumQ += q[k]
	sumQ = 1.0/sumQ
	for k in range(K):
		q[k] *= sumQ

cdef inline double factorAccel(const double* v0, const double* v1, const double* v2, \
		const size_t I) noexcept nogil:
	cdef:
		size_t i
		double sum1 = 0.0
		double sum2 = 0.0
		double u, v
	for i in prange(I):
		u = v1[i] - v0[i]
		v = v2[i] - v1[i] - u
		sum1 += u*u
		sum2 += u*v
	return min(max(-(sum1/sum2), 1.0), 256.0)

# Create P matrix from array
cpdef void createP(double[::1] P, const unsigned char[::1] k_vec, \
		const unsigned int[::1] c_vec, const size_t K) noexcept nogil:
	cdef:
		size_t W = k_vec.shape[0]
		size_t c, k, l, s, w, B
		double sumP
	for w in prange(W):
		B = k_vec[w]
		l = c_vec[w]
		for k in range(K):
			s = l + k*B
			sumP = 0.0
			for c in range(B):
				sumP = sumP + P[s+c]
			sumP = 1.0/sumP
			for c in range(B):
				P[s+c] *= sumP

# Update Q in supervised mode
cpdef void superQ(double[:,::1] Q, const unsigned char[::1] y) noexcept nogil:
	cdef:
		size_t N = Q.shape[0]
		size_t K = Q.shape[1]
		size_t i, k
		double sumQ
	for i in prange(N):
		if y[i//2] > 0:
			sumQ = 0.0
			for k in range(K):
				if k == (y[i//2]-1):
					Q[i,k] = 1-(1e-5)
				else:
					Q[i,k] = 1e-5
				sumQ = sumQ + Q[i,k]
			sumQ = 1.0/sumQ
			for k in range(K):
				Q[i,k] *= sumQ

# Update P and Q temp arrays
cpdef void updateP(const unsigned char[:,::1] Z, double[::1] P, const double[:,::1] Q, \
		double[::1] P_tmp, double[:,::1] Q_tmp, const unsigned char[::1] k_vec, \
		const unsigned int[::1] c_vec) noexcept nogil:
	cdef:
		size_t W = Z.shape[0]
		size_t N = Z.shape[1]
		size_t K = Q.shape[1]
		size_t i, l, n, w, z, x, y, B
		double h
		double* Q_thr
	with nogil, parallel():
		Q_thr = <double*>calloc((N//2)*K, sizeof(double))
		for w in prange(W):
			B = k_vec[w]
			l = c_vec[w]
			for i in range(N):
				n = i//2
				z = <size_t>Z[w,i]
				h = computeH(&P[l], &Q[n,0], z, K, B)
				innerJ(&P[l], &Q[n,0], &P_tmp[l], &Q_thr[n*K], h, z, K, B)
			outerP(&P[l], &P_tmp[l], K, B)
		with gil:
			for x in range(N//2):
				for y in range(K):
					Q_tmp[x,y] += Q_thr[x*K + y]
		free(Q_thr)

# Accelerated update P and Q temp arrays
cpdef void accelP(const unsigned char[:,::1] Z, const double[::1] P, \
		double[::1] P_new, const double[:,::1] Q, double[::1] P_tmp, \
		double[:,::1] Q_tmp, const unsigned char[::1] k_vec, \
		const unsigned int[::1] c_vec) noexcept nogil:
	cdef:
		size_t W = Z.shape[0]
		size_t N = Z.shape[1]
		size_t K = Q.shape[1]
		size_t i, l, n, w, z, x, y, B
		double h
		double* Q_thr
	with nogil, parallel():
		Q_thr = <double*>calloc((N//2)*K, sizeof(double))
		for w in prange(W):
			B = k_vec[w]
			l = c_vec[w]
			for i in range(N):
				n = i//2
				z = <size_t>Z[w,i]
				h = computeH(&P[l], &Q[n,0], z, K, B)
				innerJ(&P[l], &Q[n,0], &P_tmp[l], &Q_thr[n*K], h, z, K, B)
			outerP(&P_new[l], &P_tmp[l], K, B)
		with gil:
			for x in range(N//2):
				for y in range(K):
					Q_tmp[x,y] += Q_thr[x*K + y]
		free(Q_thr)

# Accelerated jump for P (QN)
cpdef void alphaP(double[::1] P0, const double[::1] P1, const double[::1] P2, \
		const unsigned char[::1] k_vec, const unsigned int[::1] c_vec, const size_t K) \
		noexcept nogil:
	cdef:
		size_t M = P0.shape[0]
		size_t W = k_vec.shape[0]
		size_t c, k, l, s, w, B
		double sum1 = 0.0
		double sum2 = 0.0
		double c1, c2, sumP
	c1 = factorAccel(&P0[0], &P1[0], &P2[0], M)
	c2 = 1.0 - c1
	for w in prange(W):
		B = k_vec[w]
		l = c_vec[w]
		for k in range(K):
			s = l + k*B
			sumP = 0.0
			for c in range(B):
				P0[s+c] = project(c2*P1[s+c] + c1*P2[s+c])
				sumP = sumP + P0[s+c]
			sumP = 1.0/sumP
			for c in range(B):
				P0[s+c] *= sumP

# Update Q
cpdef void updateQ(double[:,::1] Q, double[:,::1] Q_tmp, const size_t W) noexcept nogil:
	cdef:
		size_t N = Q.shape[0]
		size_t K = Q.shape[1]
		size_t i, k
		double S = 1.0/<double>(2*W)
	for i in prange(N):
		outerQ(&Q[i,0], &Q_tmp[i,0], S, K)

# Accelerated jump for Q (QN)
cpdef void alphaQ(double[:,::1] Q0, const double[:,::1] Q1, const double[:,::1] Q2) \
		noexcept nogil:
	cdef:
		size_t N = Q0.shape[0]
		size_t K = Q0.shape[1]
		size_t i, k
		double c1, c2, sumQ
	c1 = factorAccel(&Q0[0,0], &Q1[0,0], &Q2[0,0], N*K)
	c2 = 1.0 - c1
	for i in prange(N):
		sumQ = 0.0
		for k in range(K):
			Q0[i,k] = project(c2*Q1[i,k] + c1*Q2[i,k])
			sumQ = sumQ + Q0[i,k]
		sumQ = 1.0/sumQ
		for k in range(K):
			Q0[i,k] *= sumQ	

# Log-likelihood
cpdef double loglike(const unsigned char[:,::1] Z, const double[::1] P, \
		const double[:,::1] Q, const unsigned char[::1] k_vec, const unsigned int[::1] c_vec) \
		noexcept nogil:
	cdef:
		size_t W = Z.shape[0]
		size_t N = Z.shape[1]
		size_t K = Q.shape[1]
		size_t i, k, l, n, w, z, B
		double res = 0.0
		double h
	for w in prange(W):
		B = k_vec[w]
		l = c_vec[w]
		for i in range(N):
			n = i//2
			z = <size_t>Z[w,i]
			res += computeL(&P[l], &Q[n,0], z, K, B)
	return res
