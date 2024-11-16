# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
import numpy as np
cimport numpy as np
from cython.parallel import parallel, prange
from libc.math cimport log, sqrt
from libc.stdlib cimport calloc, free

##### hapla - ancestry estimation #####
# Inline functions
cdef inline double project(const double s) noexcept nogil:
	return min(max(s, 1e-5), 1-(1e-5))

cdef inline double computeH(const double* p, const double* q, const int z, \
		const int K, const int C) noexcept nogil:
	cdef:
		int k
		double h = 0.0
	for k in range(K):
		h += p[k*C+z]*q[k]
	return 1.0/h

cdef inline void innerJ(const double* p, const double* q, double* p_thr, \
		double* q_thr, const double h, const int z, const int K, const int C) \
		noexcept nogil:
	cdef:
		int k
		double a
	for k in range(K):
		a = (p[k*C+z]*q[k])*h
		p_thr[k*C+z] += a
		q_thr[k] += a

cdef inline void outerP(double* p, double* p_thr, const double S, const int K, const int C, \
		const int B) noexcept nogil:
	cdef:
		int c, k 
		double sumP
	for k in range(K):
		sumP = 0.0
		for c in range(B):
			p[k*C+c] = project(p_thr[k*C+c]*S)
			p_thr[k*C+c] = 0.0
			sumP += p[k*C+c]
		for c in range(B):
			p[k*C+c] /= sumP

cdef inline double factorP(const double* p0, const double* p1, const double* p2, \
		const unsigned char* k_vec, const int W, const int K, const int C) \
		noexcept nogil:
	cdef:
		int c, k, l, w, B
		double sum1 = 0.0
		double sum2 = 0.0
		double u, v
	for w in range(W):
		B = k_vec[w]
		for k in range(K):
			for c in range(B):
				l = w*K*C + k*C + c
				u = p1[l]-p0[l]
				v = (p2[l]-p1[l])-u
				sum1 += u*u
				sum2 += u*v
	return -(sum1/sum2)

cdef inline void outerQ(double* q, double* q_tmp, const double S, const int K) \
		noexcept nogil:
	cdef:
		int k
		double sumQ = 0.0
	for k in range(K):
		q[k] = project(q_tmp[k]*S)
		q_tmp[k] = 0.0
		sumQ += q[k]
	for k in range(K):
		q[k] /= sumQ

cdef inline void outerAccelQ(const double* q, double* q_new, double* q_tmp, \
		const double S, const int K) noexcept nogil:
	cdef:
		int k
		double sumQ = 0.0
	for k in range(K):
		q_new[k] = project(q_tmp[k]*S)
		q_tmp[k] = 0.0
		sumQ += q_new[k]
	for k in range(K):
		q_new[k] /= sumQ

cdef inline double factorQ(const double* q0, const double* q1, const double* q2, \
		const int N, const int K) noexcept nogil:
	cdef:
		int i, k, l
		double sum1 = 0.0
		double sum2 = 0.0
		double u, v
	for i in range(N):
		for k in range(K):
			l = i*K + k
			u = q1[l]-q0[l]
			v = (q2[l]-q1[l])-u
			sum1 += u*u
			sum2 += u*v
	return -(sum1/sum2)

# Create P matrix from array
cpdef void createP(double[:,:,::1] P, const unsigned char[::1] k_vec, const int t) \
		noexcept nogil:
	cdef:
		int W = P.shape[0]
		int K = P.shape[1]
		int C = P.shape[2]
		int c, k, w
		double sumP
	for w in prange(W, num_threads=t):
		for k in range(K):
			sumP = 0.0
			for c in range(C):
				if c < k_vec[w]:
					P[w,k,c] = project(P[w,k,c])
					sumP = sumP + P[w,k,c]
				else:
					P[w,k,c] = 0.0
			for c in range(k_vec[w]):
				P[w,k,c] /= sumP

# Initialize Q in supervised mode
cpdef void initQ(double[:,::1] Q, const unsigned char[::1] y, const int t) \
		noexcept nogil:
	cdef:
		int n = Q.shape[0]
		int K = Q.shape[1]
		int i, k
		double sumQ
	for i in prange(n, num_threads=t):
		if y[i//2] > 0:
			for k in range(K):
				if k == (y[i//2]-1):
					Q[i,k] = 1-(1e-5)
				else:
					Q[i,k] = 1e-5
		sumQ = 0.0
		for k in range(K):
			Q[i,k] = project(Q[i,k])
			sumQ = sumQ + Q[i,k]
		for k in range(K):
			Q[i,k] /= sumQ

# Update Q in supervised mode
cpdef void superQ(double[:,::1] Q, const unsigned char[::1] y, const int t) \
		noexcept nogil:
	cdef:
		int n = Q.shape[0]
		int K = Q.shape[1]
		int i, k
		double sumQ
	for i in prange(n, num_threads=t):
		if y[i//2] > 0:
			sumQ = 0.0
			for k in range(K):
				if k == (y[i//2]-1):
					Q[i,k] = 1-(1e-5)
				else:
					Q[i,k] = 1e-5
				sumQ = sumQ + Q[i,k]
			for k in range(K):
				Q[i,k] /= sumQ

# Update P and Q temp arrays
cpdef void updateP(const unsigned char[:,::1] Z, double[:,:,::1] P, \
		const double[:,::1] Q, double[:,::1] Q_tmp, const unsigned char[::1] k_vec, \
		const int t) noexcept nogil:
	cdef:
		int W = Z.shape[0]
		int n = Z.shape[1]
		int K = P.shape[1]
		int C = P.shape[2]
		int c, i, k, l, w, z, x, y, B
		double S = 1.0/<double>n
		double h, sumP
		double* P_thr
		double* Q_thr
	with nogil, parallel(num_threads=t):
		P_thr = <double*>calloc(K*C, sizeof(double))
		Q_thr = <double*>calloc((n//2)*K, sizeof(double))
		for w in prange(W):
			B = k_vec[w]
			for i in range(n):
				l = i//2
				z = Z[w,i]
				h = computeH(&P[w,0,0], &Q[l,0], z, K, C)
				innerJ(&P[w,0,0], &Q[l,0], &P_thr[0], &Q_thr[l*K], h, z, K, C)
			outerP(&P[w,0,0], &P_thr[0], S, K, C, B)
		with gil:
			for x in range(n//2):
				for y in range(K):
					Q_tmp[x,y] += Q_thr[x*K + y]
		free(P_thr)
		free(Q_thr)

# Accelerated update P and Q temp arrays
cpdef void accelP(const unsigned char[:,::1] Z, const double[:,:,::1] P, \
		double[:,:,::1] P_new, const double[:,::1] Q, double[:,::1] Q_tmp, \
		const unsigned char[::1] k_vec, const int t) noexcept nogil:
	cdef:
		int W = Z.shape[0]
		int n = Z.shape[1]
		int K = P.shape[1]
		int C = P.shape[2]
		int c, i, k, l, w, z, x, y, B
		double S = 1.0/<double>n
		double h, sumP
		double* P_thr
		double* Q_thr
	with nogil, parallel(num_threads=t):
		P_thr = <double*>calloc(K*C, sizeof(double))
		Q_thr = <double*>calloc((n//2)*K, sizeof(double))
		for w in prange(W):
			B = k_vec[w]
			for i in range(n):
				l = i//2
				z = Z[w,i]
				h = computeH(&P[w,0,0], &Q[l,0], z, K, C)
				innerJ(&P[w,0,0], &Q[l,0], &P_thr[0], &Q_thr[l*K], h, z, K, C)
			outerP(&P_new[w,0,0], &P_thr[0], S, K, C, B)
		with gil:
			for x in range(n//2):
				for y in range(K):
					Q_tmp[x,y] += Q_thr[x*K + y]
		free(P_thr)
		free(Q_thr)

# Accelerated jump for P (QN)
cpdef void alphaP(double[:,:,::1] P0, const double[:,:,::1] P1, \
		const double[:,:,::1] P2, const unsigned char[::1] k_vec, const int t) \
		noexcept nogil:
	cdef:
		int W = P0.shape[0]
		int K = P0.shape[1]
		int C = P0.shape[2]
		int c, k, w, B
		double sum1 = 0.0
		double sum2 = 0.0
		double c1, c2, sumP
	c1 = min(max(factorP(&P0[0,0,0], &P1[0,0,0], &P2[0,0,0], &k_vec[0], W, K, C), \
		1.0), 256.0)
	c2 = 1.0 - c1
	for w in prange(W, num_threads=t):
		B = k_vec[w]
		for k in range(K):
			sumP = 0.0
			for c in range(B):
				P0[w,k,c] = project(c2*P1[w,k,c] + c1*P2[w,k,c])
				sumP = sumP + P0[w,k,c]
			for c in range(B):
				P0[w,k,c] /= sumP

# Update Q
cpdef void updateQ(double[:,::1] Q, double[:,::1] Q_tmp, const double S, \
		const int t) noexcept nogil:
	cdef:
		int n = Q.shape[0]
		int K = Q.shape[1]
		int i, k
	for i in prange(n, num_threads=t):
		outerQ(&Q[i,0], &Q_tmp[i,0], S, K)

# Accelerated update Q
cpdef void accelQ(const double[:,::1] Q, double[:,::1] Q_new, double[:,::1] Q_tmp, \
		const double S, const int t) noexcept nogil:
	cdef:
		int n = Q.shape[0]
		int K = Q.shape[1]
		int i, k
	for i in prange(n, num_threads=t):
		outerAccelQ(&Q[i,0], &Q_new[i,0], &Q_tmp[i,0], S, K)

# Accelerated jump for Q (QN)
cpdef void alphaQ(double[:,::1] Q0, const double[:,::1] Q1, const double[:,::1] Q2, \
		const int t) noexcept nogil:
	cdef:
		int n = Q0.shape[0]
		int K = Q0.shape[1]
		int i, k
		double c1, c2, sumQ
	c1 = min(max(factorQ(&Q0[0,0], &Q1[0,0], &Q2[0,0], n, K), 1.0), 256.0)
	c2 = 1.0 - c1
	for i in prange(n, num_threads=t):
		sumQ = 0.0
		for k in range(K):
			Q0[i,k] = project(c2*Q1[i,k] + c1*Q2[i,k])
			sumQ += Q0[i,k]
		for k in range(K):
			Q0[i,k] /= sumQ	

# Log-likelihood
cpdef void loglike(const unsigned char[:,::1] Z, const double[:,:,::1] P, \
		const double[:,::1] Q, double[::1] l_vec, const int t) \
		noexcept nogil:
	cdef:
		int W = Z.shape[0]
		int n = Z.shape[1]
		int K = P.shape[1]
		int i, k, l, w, z
		double h
	for w in prange(W, num_threads=t):
		l_vec[w] = 0.0
		for i in range(n):
			l = i//2
			z = Z[w,i]
			h = 0.0
			for k in range(K):
				h = h + P[w,k,z]*Q[l,k]
			l_vec[w] += log(h)
