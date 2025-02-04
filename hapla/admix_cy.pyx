# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
import numpy as np
cimport numpy as np
cimport openmp as omp
from cython.parallel import parallel, prange
from libc.math cimport fmin, fmax, log
from libc.stdlib cimport calloc, free

##### hapla - ancestry estimation #####
### Inline functions
# Truncate parameters to domain
cdef inline double project(const double s) noexcept nogil:
	return fmin(fmax(s, 1e-5), 1.0-(1e-5))

# Estimate inverse individual allele frequency
cdef inline double computeH(const double* p, const double* q, const size_t z, \
		const size_t K, const size_t B) noexcept nogil:
	cdef:
		size_t k
		double h = 0.0
	for k in range(K):
		h += p[k*B + z]*q[k]
	return 1.0/h

# Estimate individual allele frequency
cdef inline double computeL(const double* p, const double* q, const size_t z, \
		const size_t K, const size_t B) noexcept nogil:
	cdef:
		size_t k
		double h = 0.0
	for k in range(K):
		h += p[k*B + z]*q[k]
	return log(h)

# Inner loop updates for temp P and Q
cdef inline void inner(const double* p, const double* q, double* p_tmp, \
		double* q_thr, const double h, const size_t z, const size_t K, const size_t B) \
		noexcept nogil:
	cdef:
		size_t k
	for k in range(K):
		p_tmp[k*B + z] += q[k]*h
		q_thr[k] += p[k*B + z]*h

# Outer loop update for P
cdef inline void outerP(double* p, double* p_tmp, const size_t K, const size_t B) \
		noexcept nogil:
	cdef:
		size_t c, k, s
		double sumA, sumB, valP
	for k in range(K):
		s = k*B
		sumA = 0.0
		sumB = 0.0
		for c in range(B):
			valP = p_tmp[s+c]*p[s+c]
			sumA += valP
			p[s+c] = valP
		for c in range(B):
			valP = project(p[s+c]/sumA)
			sumB += valP
			p[s+c] = valP
		for c in range(B):
			p[s+c] /= sumB
			p_tmp[s+c] = 0.0

# Outer loop accelerated update for P
cdef inline void outerAccelP(const double* p, double* p_new, double* p_tmp, \
		const size_t K, const size_t B) noexcept nogil:
	cdef:
		size_t c, k, s
		double sumA, sumB, valP
	for k in range(K):
		s = k*B
		sumA = 0.0
		sumB = 0.0
		for c in range(B):
			valP = p_tmp[s+c]*p[s+c]
			sumA += valP
			p_new[s+c] = valP
		for c in range(B):
			valP = project(p_new[s+c]/sumA)
			sumB += valP
			p_new[s+c] = valP
		for c in range(B):
			p_new[s+c] /= sumB
			p_tmp[s+c] = 0.0

# Outer loop update for Q
cdef inline void outerQ(double* q, double* q_tmp, const double S, const size_t K) \
		noexcept nogil:
	cdef:
		size_t k
		double sumQ = 0.0
		double valQ
	for k in range(K):
		valQ = project(q[k]*q_tmp[k]*S)
		sumQ += valQ
		q[k] = valQ
	for k in range(K):
		q[k] /= sumQ
		q_tmp[k] = 0.0

# Outer loop accelerated update for Q
cdef inline void outerAccelQ(const double* q, double* q_new, double* q_tmp, const double S, \
		const size_t K) noexcept nogil:
	cdef:
		size_t k
		double sumQ = 0.0
		double valQ
	for k in range(K):
		valQ = project(q[k]*q_tmp[k]*S)
		sumQ += valQ
		q_new[k] = valQ
	for k in range(K):
		q_new[k] /= sumQ
		q_tmp[k] = 0.0

# Estimate QN factor
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
	return fmin(fmax(-(sum1/sum2), 1.0), 256.0)


### Update functions
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
			for c in range(B):
				P[s+c] /= sumP

# Update Q in supervised mode
cpdef void superQ(double[:,::1] Q, const unsigned char[::1] y) noexcept nogil:
	cdef:
		size_t N = Q.shape[0]
		size_t K = Q.shape[1]
		size_t i, k
		double sumQ, valQ
	for i in prange(N):
		if y[i] > 0:
			sumQ = 0.0
			for k in range(K):
				if k == (y[i]-1):
					valQ = 1-(1e-5)
				else:
					valQ = 1e-5
				sumQ = sumQ + valQ
				Q[i,k] = valQ
			for k in range(K):
				Q[i,k] /= sumQ

# Update P and Q temp arrays
cpdef void updateP(const unsigned char[:,::1] Z, double[::1] P, const double[:,::1] Q, \
		double[::1] P_tmp, double[:,::1] Q_tmp, const unsigned char[::1] k_vec, \
		const unsigned int[::1] c_vec) noexcept nogil:
	cdef:
		size_t W = Z.shape[0]
		size_t N = Z.shape[1]
		size_t K = Q.shape[1]
		size_t i, l, n, w, x, y, z, B
		double h
		double* pl
		double* pt
		double* Q_thr
		omp.omp_lock_t mutex
	omp.omp_init_lock(&mutex)
	with nogil, parallel():
		Q_thr = <double*>calloc((N//2)*K, sizeof(double))
		for w in prange(W):
			B = k_vec[w]
			l = c_vec[w]
			pl = &P[l]
			pt = &P_tmp[l]
			for i in range(N):
				n = i//2
				z = <size_t>Z[w,i]
				h = computeH(pl, &Q[n,0], z, K, B)
				inner(pl, &Q[n,0], pt, &Q_thr[n*K], h, z, K, B)
			outerP(pl, pt, K, B)

		# omp critical
		omp.omp_set_lock(&mutex)
		for x in range(N//2):
			for y in range(K):
				Q_tmp[x,y] += Q_thr[x*K + y]
		omp.omp_unset_lock(&mutex)
		free(Q_thr)
	omp.omp_destroy_lock(&mutex)

# Accelerated update P and Q temp arrays
cpdef void accelP(const unsigned char[:,::1] Z, double[::1] P, double[::1] P_new, \
		const double[:,::1] Q, double[::1] P_tmp, double[:,::1] Q_tmp, \
		const unsigned char[::1] k_vec, const unsigned int[::1] c_vec) noexcept nogil:
	cdef:
		size_t W = Z.shape[0]
		size_t N = Z.shape[1]
		size_t K = Q.shape[1]
		size_t i, l, n, w, x, y, z, B
		double h
		double* pl
		double* pt
		double* Q_thr
		omp.omp_lock_t mutex
	omp.omp_init_lock(&mutex)
	with nogil, parallel():
		Q_thr = <double*>calloc((N//2)*K, sizeof(double))
		for w in prange(W):
			B = k_vec[w]
			l = c_vec[w]
			pl = &P[l]
			pt = &P_tmp[l]
			for i in range(N):
				n = i//2
				z = <size_t>Z[w,i]
				h = computeH(pl, &Q[n,0], z, K, B)
				inner(pl, &Q[n,0], pt, &Q_thr[n*K], h, z, K, B)
			outerAccelP(pl, &P_new[l], pt, K, B)
		
		# omp critical
		omp.omp_set_lock(&mutex)
		for x in range(N//2):
			for y in range(K):
				Q_tmp[x,y] += Q_thr[x*K + y]
		omp.omp_unset_lock(&mutex)
		free(Q_thr)
	omp.omp_destroy_lock(&mutex)

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
		double c1, c2, sumP, valP
	c1 = factorAccel(&P0[0], &P1[0], &P2[0], M)
	c2 = 1.0 - c1
	for w in prange(W):
		B = k_vec[w]
		l = c_vec[w]
		for k in range(K):
			s = l + k*B
			sumP = 0.0
			for c in range(B):
				valP = project(c2*P1[s+c] + c1*P2[s+c])
				sumP = sumP + valP
				P0[s+c] = valP
			for c in range(B):
				P0[s+c] /= sumP

# Update Q
cpdef void updateQ(double[:,::1] Q, double[:,::1] Q_tmp, const size_t W) noexcept nogil:
	cdef:
		size_t N = Q.shape[0]
		size_t K = Q.shape[1]
		size_t i, k
		double S = 1.0/<double>(2*W)
	for i in prange(N):
		outerQ(&Q[i,0], &Q_tmp[i,0], S, K)

# Accelerated update Q
cpdef void accelQ(const double[:,::1] Q, double[:,::1] Q_new, double[:,::1] Q_tmp, \
		const size_t W) noexcept nogil:
	cdef:
		size_t N = Q.shape[0]
		size_t K = Q.shape[1]
		size_t i, k
		double S = 1.0/<double>(2*W)
	for i in prange(N):
		outerAccelQ(&Q[i,0], &Q_new[i,0], &Q_tmp[i,0], S, K)

# Accelerated jump for Q (QN)
cpdef void alphaQ(double[:,::1] Q0, const double[:,::1] Q1, const double[:,::1] Q2) \
		noexcept nogil:
	cdef:
		size_t N = Q0.shape[0]
		size_t K = Q0.shape[1]
		size_t i, k
		double c1, c2, sumQ, valQ
	c1 = factorAccel(&Q0[0,0], &Q1[0,0], &Q2[0,0], N*K)
	c2 = 1.0 - c1
	for i in prange(N):
		sumQ = 0.0
		for k in range(K):
			valQ = project(c2*Q1[i,k] + c1*Q2[i,k])
			sumQ = sumQ + valQ
			Q0[i,k] = valQ
		for k in range(K):
			Q0[i,k] /= sumQ	

# Log-likelihood
cpdef double loglike(const unsigned char[:,::1] Z, double[::1] P, \
		const double[:,::1] Q, const unsigned char[::1] k_vec, \
		const unsigned int[::1] c_vec) noexcept nogil:
	cdef:
		size_t W = Z.shape[0]
		size_t N = Z.shape[1]
		size_t K = Q.shape[1]
		size_t i, l, w, B
		double res = 0.0
		double h
		double* pl
	for w in prange(W):
		B = k_vec[w]
		l = c_vec[w]
		pl = &P[l]
		for i in range(N):
			res += computeL(pl, &Q[i//2,0], <size_t>Z[w,i], K, B)
	return res
