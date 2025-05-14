# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
cimport numpy as np
cimport openmp as omp
from cython.parallel import parallel, prange
from libc.math cimport exp, fmax, fmin, log
from libc.stdint cimport uint8_t, uint32_t
from libc.stdlib cimport calloc, free

cdef double PRO_MIN = 1e-5
cdef double PRO_MAX = 1.0-(1e-5)
cdef double ACC_MIN = 1.0
cdef double ACC_MAX = 256.0

##### hapla - ancestry estimation #####
# Truncate parameters to domain
cdef inline double _project(
		const double a
	) noexcept nogil:
	return fmin(fmax(a, PRO_MIN), PRO_MAX)

# Estimate inverse individual allele frequency
cdef inline double _computeH(
		const double* p, const double* q, const size_t z, const uint32_t K, const uint32_t B
	) noexcept nogil:
	cdef:
		double h = 0.0
		size_t k
	for k in range(K):
		h += p[k*B + z]*q[k]
	return 1.0/h

# Estimate individual allele frequency
cdef inline double _computeL(
		const double* p, const double* q, const size_t z, const uint32_t K, const uint32_t B
	) noexcept nogil:
	cdef:
		double h = 0.0
		size_t k
	for k in range(K):
		h += p[k*B + z]*q[k]
	return log(h)

# Inner loop updates for temp P and Q
cdef inline void _inner(
		const double* p, const double* q, double* p_tmp, double* q_thr, const double h, const size_t z, 
		const uint32_t K, const uint32_t B
	) noexcept nogil:
	cdef size_t k
	for k in range(K):
		p_tmp[k*B + z] += q[k]*h
		q_thr[k] += p[k*B + z]*h

# Outer loop update for P
cdef inline void _outerP(
		double* p, double* p_tmp, const uint32_t K, const uint32_t B
	) noexcept nogil:
	cdef:
		double sumA, sumB, tmpC
		size_t c, k, s
	for k in range(K):
		s = k*B
		sumA = 0.0
		sumB = 0.0
		for c in range(B):
			tmpC = p_tmp[s+c]*p[s+c]
			sumA += tmpC
			p[s+c] = tmpC
			p_tmp[s+c] = 0.0
		for c in range(B):
			tmpC = _project(p[s+c]/sumA)
			sumB += tmpC
			p[s+c] = tmpC
		for c in range(B):
			p[s+c] /= sumB

# Outer loop accelerated update for P
cdef inline void _outerAccelP(
		const double* p, double* p_new, double* p_tmp, const uint32_t K, const uint32_t B
	) noexcept nogil:
	cdef:
		double sumA, sumB, tmpC
		size_t c, k, s
	for k in range(K):
		s = k*B
		sumA = 0.0
		sumB = 0.0
		for c in range(B):
			tmpC = p_tmp[s+c]*p[s+c]
			sumA += tmpC
			p_new[s+c] = tmpC
			p_tmp[s+c] = 0.0
		for c in range(B):
			tmpC = _project(p_new[s+c]/sumA)
			sumB += tmpC
			p_new[s+c] = tmpC
		for c in range(B):
			p_new[s+c] /= sumB

# Outer loop update for Q
cdef inline void _outerQ(
		double* q, double* q_tmp, const double S, const uint32_t K
	) noexcept nogil:
	cdef:
		double sumQ = 0.0
		double tmpQ
		size_t k
	for k in range(K):
		tmpQ = _project(q[k]*q_tmp[k]*S)
		sumQ += tmpQ
		q[k] = tmpQ
		q_tmp[k] = 0.0
	for k in range(K):
		q[k] /= sumQ

# Outer loop accelerated update for Q
cdef inline void _outerAccelQ(
		const double* q, double* q_new, double* q_tmp, const double S, const uint32_t K
	) noexcept nogil:
	cdef:
		double sumQ = 0.0
		double tmpQ
		size_t k
	for k in range(K):
		tmpQ = _project(q[k]*q_tmp[k]*S)
		sumQ += tmpQ
		q_new[k] = tmpQ
		q_tmp[k] = 0.0
	for k in range(K):
		q_new[k] /= sumQ

# Estimate QN factor
cdef inline double _factorAccel(
		const double* v0, const double* v1, const double* v2, const uint32_t I
	) noexcept nogil:
	cdef:
		double sum1 = 0.0
		double sum2 = 0.0
		double u, v
		size_t i
	for i in prange(I):
		u = v1[i] - v0[i]
		v = v2[i] - v1[i] - u
		sum1 += u*u
		sum2 += u*v
	return fmin(fmax(-(sum1/sum2), ACC_MIN), ACC_MAX)

# Estimate batch QN factor
cdef inline double _factorBatchAccel(
		const double* p0, const double* p1, const double* p2, const uint8_t* k_vec, const uint32_t* c_vec, 
		const uint32_t* s_bat, const uint32_t W, const uint32_t K
	) noexcept nogil:
	cdef:
		uint32_t B
		double sum1 = 0.0
		double sum2 = 0.0
		double u, v
		size_t c, k, l, r, s, w
	for w in prange(W):
		r = <size_t>s_bat[w]
		l = <size_t>c_vec[r]
		B = <uint32_t>k_vec[r]
		for k in range(K):
			s = l + k*B
			for c in range(B):
				u = p1[s+c] - p0[s+c]
				v = p2[s+c] - p1[s+c] - u
				sum1 += u*u
				sum2 += u*v
	return fmin(fmax(-(sum1/sum2), ACC_MIN), ACC_MAX)

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

# Average the contribution from haplotypes
cdef inline void _averageQ(
		const double* q1, const double* q2, double* qf, const uint32_t K
	) noexcept nogil:
	cdef:
		double sumQ = 0.0
		double tmpQ
		size_t k
	for k in range(K):
		tmpQ = _project((q1[k] + q2[k])/2.0)
		sumQ += tmpQ
		qf[k] = tmpQ
	for k in range(K):
		qf[k] /= sumQ


### Update functions
# Create P matrix from array
cpdef void createP(
		double[::1] P, const uint8_t[::1] k_vec, const uint32_t[::1] c_vec, const uint32_t K
	) noexcept nogil:
	cdef:
		uint32_t W = k_vec.shape[0]
		uint32_t B
		double sumP
		size_t c, k, l, s, w
	for w in prange(W):
		B = <uint32_t>k_vec[w]
		l = <size_t>c_vec[w]
		for k in range(K):
			s = l + k*B
			sumP = 0.0
			for c in range(B):
				sumP = sumP + P[s+c]
			for c in range(B):
				P[s+c] /= sumP

# Update Q in supervised mode
cpdef void superQ(
		double[:,::1] Q, const uint8_t[::1] y
	) noexcept nogil:
	cdef:
		uint32_t K = Q.shape[1]
		uint32_t N = Q.shape[0]
		double sumQ
		size_t i, k
	for i in prange(N):
		if y[i] > 0:
			sumQ = 0.0
			for k in range(K):
				if k == (y[i] - 1):
					Q[i,k] = PRO_MAX
				else:
					Q[i,k] = PRO_MIN
				sumQ = sumQ + Q[i,k]
			for k in range(K):
				Q[i,k] /= sumQ

# Update P and Q temp arrays
cpdef void updateP(
		const uint8_t[:,::1] Z, double[::1] P, const double[:,::1] Q, double[::1] P_tmp, double[:,::1] Q_tmp, 
		const uint8_t[::1] k_vec, const uint32_t[::1] c_vec
	) noexcept nogil:
	cdef:
		uint32_t W = Z.shape[0]
		uint32_t N = Z.shape[1]
		uint32_t K = Q.shape[1]
		uint32_t B
		double h
		double* pl
		double* pt
		double* Q_thr
		omp.omp_lock_t mutex
		size_t i, l, w, x, y, z
	omp.omp_init_lock(&mutex)
	with nogil, parallel():
		Q_thr = <double*>calloc(N*K, sizeof(double))
		for w in prange(W):
			l = <size_t>c_vec[w]
			B = <uint32_t>k_vec[w]
			pl = &P[l]
			pt = &P_tmp[l]
			for i in range(N):
				z = <size_t>Z[w,i]
				h = _computeH(pl, &Q[i,0], z, K, B)
				_inner(pl, &Q[i,0], pt, &Q_thr[i*K], h, z, K, B)
			_outerP(pl, pt, K, B)

		# omp critical
		omp.omp_set_lock(&mutex)
		for x in range(N):
			for y in range(K):
				Q_tmp[x,y] += Q_thr[x*K + y]
		omp.omp_unset_lock(&mutex)
		free(Q_thr)
	omp.omp_destroy_lock(&mutex)

# Accelerated update P and Q temp arrays
cpdef void accelP(
		const uint8_t[:,::1] Z, double[::1] P, double[::1] P_new, const double[:,::1] Q, double[::1] P_tmp, 
		double[:,::1] Q_tmp, const uint8_t[::1] k_vec, const uint32_t[::1] c_vec
	) noexcept nogil:
	cdef:
		uint32_t W = Z.shape[0]
		uint32_t N = Z.shape[1]
		uint32_t K = Q.shape[1]
		uint32_t B
		double h
		double* pl
		double* pt
		double* Q_thr
		omp.omp_lock_t mutex
		size_t i, l, w, x, y, z
	omp.omp_init_lock(&mutex)
	with nogil, parallel():
		Q_thr = <double*>calloc(N*K, sizeof(double))
		for w in prange(W):
			l = <size_t>c_vec[w]
			B = <uint32_t>k_vec[w]
			pl = &P[l]
			pt = &P_tmp[l]
			for i in range(N):
				z = <size_t>Z[w,i]
				h = _computeH(pl, &Q[i,0], z, K, B)
				_inner(pl, &Q[i,0], pt, &Q_thr[i*K], h, z, K, B)
			_outerAccelP(pl, &P_new[l], pt, K, B)
		
		# omp critical
		omp.omp_set_lock(&mutex)
		for x in range(N):
			for y in range(K):
				Q_tmp[x,y] += Q_thr[x*K + y]
		omp.omp_unset_lock(&mutex)
		free(Q_thr)
	omp.omp_destroy_lock(&mutex)

# Batch accelerated update P and Q temp arrays
cpdef void accelBatchP(
		const uint8_t[:,::1] Z, double[::1] P, double[::1] P_new, const double[:,::1] Q, double[::1] P_tmp, 
		double[:,::1] Q_tmp, const uint8_t[::1] k_vec, const uint32_t[::1] c_vec, const uint32_t[::1] s_bat
	) noexcept nogil:
	cdef:
		uint32_t W = s_bat.shape[0]
		uint32_t N = Z.shape[1]
		uint32_t K = Q.shape[1]
		uint32_t B
		double h
		double* pl
		double* pt
		double* Q_thr
		omp.omp_lock_t mutex
		size_t i, l, r, w, x, y, z
	omp.omp_init_lock(&mutex)
	with nogil, parallel():
		Q_thr = <double*>calloc(N*K, sizeof(double))
		for w in prange(W):
			r = <size_t>s_bat[w]
			l = <size_t>c_vec[r]
			B = <uint32_t>k_vec[r]
			pl = &P[l]
			pt = &P_tmp[l]
			for i in range(N):
				z = <size_t>Z[r,i]
				h = _computeH(pl, &Q[i,0], z, K, B)
				_inner(pl, &Q[i,0], pt, &Q_thr[i*K], h, z, K, B)
			_outerAccelP(pl, &P_new[l], pt, K, B)
		
		# omp critical
		omp.omp_set_lock(&mutex)
		for x in range(N):
			for y in range(K):
				Q_tmp[x,y] += Q_thr[x*K + y]
		omp.omp_unset_lock(&mutex)
		free(Q_thr)
	omp.omp_destroy_lock(&mutex)

# Accelerated jump for P (QN)
cpdef void alphaP(
		double[::1] P0, const double[::1] P1, const double[::1] P2, const uint8_t[::1] k_vec, 
		const uint32_t[::1] c_vec, const uint32_t K
	) noexcept nogil:
	cdef:
		uint32_t M = P0.shape[0]
		uint32_t W = k_vec.shape[0]
		uint32_t B
		double c1, c2, sumP
		size_t c, k, l, s, w
	c1 = _factorAccel(&P0[0], &P1[0], &P2[0], M)
	c2 = 1.0 - c1
	for w in prange(W):
		B = <uint32_t>k_vec[w]
		l = <size_t>c_vec[w]
		for k in range(K):
			s = l + k*B
			sumP = 0.0
			for c in range(B):
				P0[s+c] = _project(c2*P1[s+c] + c1*P2[s+c])
				sumP = sumP + P0[s+c]
			for c in range(B):
				P0[s+c] /= sumP

# Batch accelerated jump for P (QN)
cpdef void alphaBatchP(
		double[::1] P0, const double[::1] P1, const double[::1] P2, const uint8_t[::1] k_vec, 
		const uint32_t[::1] c_vec, const uint32_t[::1] s_bat, const uint32_t K
	) noexcept nogil:
	cdef:
		uint32_t W = s_bat.shape[0]
		uint32_t B
		double c1, c2, sumP
		size_t c, k, l, r, s, w
	c1 = _factorBatchAccel(&P0[0], &P1[0], &P2[0], &k_vec[0], &c_vec[0], &s_bat[0], W, K)
	c2 = 1.0 - c1
	for w in prange(W):
		r = <size_t>s_bat[w]
		l = <size_t>c_vec[r]
		B = <uint32_t>k_vec[r]
		for k in range(K):
			s = l + k*B
			sumP = 0.0
			for c in range(B):
				P0[s+c] = _project(c2*P1[s+c] + c1*P2[s+c])
				sumP = sumP + P0[s+c]
			for c in range(B):
				P0[s+c] /= sumP

# Update Q
cpdef void updateQ(
		double[:,::1] Q, double[:,::1] Q_tmp, const uint32_t W
	) noexcept nogil:
	cdef:
		uint32_t N = Q.shape[0]
		uint32_t K = Q.shape[1]
		double S = 1.0/<double>W
		size_t i, k
	for i in prange(N):
		_outerQ(&Q[i,0], &Q_tmp[i,0], S, K)

# Accelerated update Q
cpdef void accelQ(
		const double[:,::1] Q, double[:,::1] Q_new, double[:,::1] Q_tmp, const uint32_t W
	) noexcept nogil:
	cdef:
		uint32_t N = Q.shape[0]
		uint32_t K = Q.shape[1]
		double S = 1.0/<double>W
		size_t i, k
	for i in prange(N):
		_outerAccelQ(&Q[i,0], &Q_new[i,0], &Q_tmp[i,0], S, K)

# Accelerated jump for Q (QN)
cpdef void alphaQ(
		double[:,::1] Q0, const double[:,::1] Q1, const double[:,::1] Q2
	) noexcept nogil:
	cdef:
		uint32_t N = Q0.shape[0]
		uint32_t K = Q0.shape[1]
		double c1, c2, sumQ
		size_t i, k
	c1 = _factorAccel(&Q0[0,0], &Q1[0,0], &Q2[0,0], N*K)
	c2 = 1.0 - c1
	for i in prange(N):
		sumQ = 0.0
		for k in range(K):
			Q0[i,k] = _project(c2*Q1[i,k] + c1*Q2[i,k])
			sumQ = sumQ + Q0[i,k]
		for k in range(K):
			Q0[i,k] /= sumQ	


### Misc functions
# Log-likelihood
cpdef double loglike(
		const uint8_t[:,::1] Z, double[::1] P, const double[:,::1] Q, const uint8_t[::1] k_vec, 
		const uint32_t[::1] c_vec
	) noexcept nogil:
	cdef:
		uint32_t W = Z.shape[0]
		uint32_t N = Z.shape[1]
		uint32_t K = Q.shape[1]
		double res = 0.0
		double h
		double* p
		size_t B
		size_t i, w
	for w in prange(W):
		B = <size_t>k_vec[w]
		p = &P[c_vec[w]]
		for i in range(N):
			res += _computeL(p, &Q[i,0], <size_t>Z[w,i], K, B)
	return res

# Convert log-likes to normalized likes
cpdef void createLikes(
		float[::1] L, const uint8_t[::1] k_vec, const uint32_t[::1] x_vec
	) noexcept nogil:
	cdef:
		uint32_t W = k_vec.shape[0]
		size_t w
	for w in prange(W):
		_normLikes(&L[x_vec[w]], <uint32_t>k_vec[w])

# Convert ancestry proportions to individual-level from haplotype-level
cpdef void convertQ(
		const double[:,::1] Q, double[:,::1] Q_fin
	) noexcept nogil:
	cdef:
		uint32_t N = Q_fin.shape[0]
		uint32_t K = Q_fin.shape[1]
		size_t i, l
	for i in prange(N):
		l = 2*i
		_averageQ(&Q[l,0], &Q[l+1,0], &Q_fin[i,0], K)
