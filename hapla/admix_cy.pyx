# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
cimport numpy as np
cimport openmp as omp
from cython.parallel import parallel, prange
from libc.math cimport exp, log, sqrtf
from libc.stdint cimport uint8_t, uint32_t
from libc.stdlib cimport calloc, free

ctypedef uint8_t u8
ctypedef uint32_t u32
ctypedef float f32
ctypedef double f64

cdef f64 PRO_MIN = 1e-5
cdef f64 PRO_MAX = 1.0 - (1e-5)
cdef f64 ACC_MIN = 1.0
cdef f64 ACC_MAX = 128.0
cdef f32 FLT_MIN = 1e-5
cdef f32 FLT_MAX = 1.0 - (1e-5)
cdef inline f64 _fmax(f64 a, f64 b) noexcept nogil: return a if a > b else b
cdef inline f64 _fmin(f64 a, f64 b) noexcept nogil: return a if a < b else b
cdef inline f32 _fmaxf(f32 a, f32 b) noexcept nogil: return a if a > b else b
cdef inline f32 _fminf(f32 a, f32 b) noexcept nogil: return a if a < b else b
cdef inline f64 _clamp1(f64 a) noexcept nogil: return _fmax(PRO_MIN, _fmin(a, PRO_MAX))
cdef inline f64 _clamp2(f64 a) noexcept nogil: return _fmax(ACC_MIN, _fmin(a, ACC_MAX))
cdef inline f32 _clamp3(f32 a) noexcept nogil: return _fmaxf(FLT_MIN, _fminf(a, FLT_MAX))


##### hapla - ancestry estimation #####
### Inline functions
# Estimate inverse individual allele frequency
cdef inline f64 _computeH(
		const f64* p, const f64* q, const Py_ssize_t K
	) noexcept nogil:
	cdef:
		size_t k
		f64 h = 0.0
	for k in range(K):
		h += p[k]*q[k]
	return 1.0/h

# Estimate individual allele frequency
cdef inline f64 _computeL(
		const f64* p, const f64* q, const Py_ssize_t K
	) noexcept nogil:
	cdef:
		size_t k
		f64 h = 0.0
	for k in range(K):
		h += p[k]*q[k]
	return log(h)

# Inner loop updates for temp P and Q
cdef inline void _innerJ(
		const f64* p, const f64* q, f64* p_thr, f64* q_thr, const f64 h, const Py_ssize_t K
	) noexcept nogil:
	cdef:
		size_t k
	for k in range(K):
		p_thr[k] += q[k]*h
		q_thr[k] += p[k]*h

# Inner loop updates for temp Q
cdef inline void _innerQ(
		const f64* p, f64* q_thr, const f64 h, const Py_ssize_t K
	) noexcept nogil:
	cdef:
		size_t k
	for k in range(K):
		q_thr[k] += p[k]*h

# Outer loop update for P
cdef inline void _outerP(
		f64* p, f64* p_thr, f64* p_sum, const f64 S, const Py_ssize_t B, const Py_ssize_t K
	) noexcept nogil:
	cdef:
		size_t c, k
		f64 a, b
		f64* p_c
		f64* p_t
	for c in range(B):
		p_c = &p[c*K]
		p_t = &p_thr[c*K]
		for k in range(K):
			a = p_c[k]*p_t[k]*S
			b = _clamp1(a)
			p_sum[k] += b
			p_c[k] = b
			p_t[k] = 0.0
	for c in range(B):
		p_c = &p[c*K]
		for k in range(K):
			p_c[k] /= p_sum[k]
	for k in range(K):
		p_sum[k] = 0.0

# Outer loop accelerated update for P
cdef inline void _outerAccelP(
		f64* p, f64* p_new, f64* p_thr, f64* p_sum, const f64 S, const Py_ssize_t B, const Py_ssize_t K
	) noexcept nogil:
	cdef:
		size_t c, k
		f64 a, b
		f64* p_c
		f64* p_n
		f64* p_t
	for c in range(B):
		p_c = &p[c*K]
		p_n = &p_new[c*K]
		p_t = &p_thr[c*K]
		for k in range(K):
			a = p_c[k]*p_t[k]*S
			b = _clamp1(a)
			p_sum[k] += b
			p_n[k] = b
			p_t[k] = 0.0
	for c in range(B):
		p_n = &p_new[c*K]
		for k in range(K):
			p_n[k] /= p_sum[k]
	for k in range(K):
		p_sum[k] = 0.0

# Outer loop update for Q
cdef inline void _outerQ(
		f64* q, f64* q_tmp, const f64 S, const Py_ssize_t K
	) noexcept nogil:
	cdef:
		size_t k
		f64 sumQ = 0.0
		f64 a, b
	for k in range(K):
		a = q[k]*q_tmp[k]*S
		b = _clamp1(a)
		sumQ += b
		q[k] = b
		q_tmp[k] = 0.0
	for k in range(K):
		q[k] /= sumQ

# Outer loop accelerated update for Q
cdef inline void _outerAccelQ(
		const f64* q, f64* q_new, f64* q_tmp, const f64 S, const Py_ssize_t K
	) noexcept nogil:
	cdef:
		size_t k
		f64 sumQ = 0.0
		f64 a, b
	for k in range(K):
		a = q[k]*q_tmp[k]*S
		b = _clamp1(a)
		sumQ += b
		q_new[k] = b
		q_tmp[k] = 0.0
	for k in range(K):
		q_new[k] /= sumQ

# Estimate QN factor
cdef inline f64 _qnC(
		const f64* v0, const f64* v1, const f64* v2, const Py_ssize_t I
	) noexcept nogil:
	cdef:
		size_t i
		f64 sum1 = 0.0
		f64 sum2 = 0.0
		f64 f, u, v
	for i in prange(I, schedule='guided'): ### TEST nested loops for SIMD in P and keep Q separated?
		u = v1[i] - v0[i]
		v = v2[i] - v1[i] - u
		sum1 += u*u
		sum2 += u*v
	f = -(sum1/sum2)
	return _clamp2(f)

# Estimate batch QN factor for P
cdef inline f64 _qnBatch(
		f64* P0, f64* P1, f64* P2, const u32* k_vec, const u32* c_vec, const u32* s_bat, const Py_ssize_t W, 
		const Py_ssize_t K
	) noexcept nogil:
	cdef:
		size_t c, l, r, w
		f64 sum1 = 0.0
		f64 sum2 = 0.0
		f64 f, u, v
		f64* p0
		f64* p1
		f64* p2
	for w in prange(W, schedule='guided'):
		r = s_bat[w]
		l = c_vec[r]
		p0 = &P0[l]
		p1 = &P1[l]
		p2 = &P2[l]
		for c in range(k_vec[r]*K):
			u = p1[c] - p0[c]
			v = p2[c] - p1[c] - u
			sum1 += u*u
			sum2 += u*v
	f = -(sum1/sum2)
	return _clamp2(f)

# Estimate QN jump in P
cdef inline void _computeP(
		f64* P0, f64* P1, f64* P2, f64* p_sum, const f64 c1, const f64 c2, const Py_ssize_t B, const Py_ssize_t K
	) noexcept nogil:
	cdef:
		size_t c, k
		f64 a, b
		f64* p0
		f64* p1
		f64* p2
	for c in range(B):
		p0 = &P0[c*K]
		p1 = &P1[c*K]
		p2 = &P2[c*K]
		for k in range(K):
			a = c2*p1[k] + c1*p2[k]
			b = _clamp1(a)
			p_sum[k] += b
			p0[k] = b
	for c in range(B):
		p0 = &P0[c*K]
		for k in range(K):
			p0[k] /= p_sum[k]
	for k in range(K):
		p_sum[k] = 0.0

# Estimate QN jump in Q
cdef inline void _computeQ(
		f64* q0, const f64* q1, const f64* q2, const f64 c1, const f64 c2, const Py_ssize_t K
	) noexcept nogil:
	cdef:
		size_t k
		f64 sumQ = 0.0
		f64 a, b
	for k in range(K):
		a = c2*q1[k] + c1*q2[k]
		b = _clamp1(a)
		sumQ += b
		q0[k] = b
	for k in range(K):
		q0[k] /= sumQ

# Average the contribution from haplotypes
cdef inline void _averageQ(
		const f64* q1, const f64* q2, f64* qf, const Py_ssize_t K
	) noexcept nogil:
	cdef:
		size_t k
		f64 sumQ = 0.0
		f64 a, b
	for k in range(K):
		a = (q1[k] + q2[k])/2.0
		b = _clamp1(a)
		sumQ += b
		qf[k] = b
	for k in range(K):
		qf[k] /= sumQ

# Project P to domain
cdef inline void _projectP(
		f32* p, f32* p_sum, const Py_ssize_t B, const Py_ssize_t K
	) noexcept nogil:
	cdef:
		size_t c, k
		f32 a, b
		f32* p_c
	for c in range(B):
		p_c = &p[c*K]
		for k in range(K):
			a = p_c[k]
			b = _clamp3(a)
			p_sum[k] += b
			p_c[k] = b
	for c in range(B):
		p_c = &p[c*K]
		for k in range(K):
			p_c[k] /= p_sum[k]
	for k in range(K):
		p_sum[k] = 0.0

# Project Q to domain
cdef inline void _projectQ(
		f32* q, const Py_ssize_t K
	) noexcept nogil:
	cdef:
		size_t k
		f32 sumQ = 0.0
		f32 a, b
	for k in range(K):
		a = q[k]
		b = _clamp3(a)
		sumQ += b
		q[k] = b
	for k in range(K):
		q[k] /= sumQ

# Compute the squared difference
cdef inline f32 _computeR(
		const f32* a, const f32* b, const Py_ssize_t I
	) noexcept nogil:
	cdef:
		f32 r = 0.0
		f32 c
		size_t i
	for i in range(I):
		c = a[i] - b[i]
		r += c*c
	return r


### Update functions
# Update P and Q temp arrays
cpdef void updateP(
		u8[:,::1] Z, f64[::1] P, const f64[:,::1] Q, f64[:,::1] Q_tmp, const u32[::1] k_vec, const u32[::1] c_vec, 
		const u32 L
	) noexcept nogil:
	cdef:
		Py_ssize_t W = Z.shape[0]
		Py_ssize_t N = Z.shape[1]
		Py_ssize_t K = Q.shape[1]
		Py_ssize_t B
		size_t i, l, s, w, x, y
		f64 S = 1.0/<f64>N
		f64 h
		f64* p
		f64* p_sum
		f64* p_thr
		f64* q_thr
		omp.omp_lock_t mutex
	omp.omp_init_lock(&mutex)
	with nogil, parallel():
		p_sum = <f64*>calloc(K, sizeof(f64))
		p_thr = <f64*>calloc(L*K, sizeof(f64))
		q_thr = <f64*>calloc(N*K, sizeof(f64))
		for w in prange(W, schedule='guided'):
			l = c_vec[w]
			B = k_vec[w]
			for i in range(N):
				s = Z[w,i]*K
				p = &P[l + s]
				h = _computeH(p, &Q[i,0], K)
				_innerJ(p, &Q[i,0], &p_thr[s], &q_thr[i*K], h, K)
			_outerP(&P[l], &p_thr[0], &p_sum[0], S, B, K)

		# omp critical
		omp.omp_set_lock(&mutex)
		for x in range(N):
			for y in range(K):
				Q_tmp[x,y] += q_thr[x*K + y]
		omp.omp_unset_lock(&mutex)
		free(p_sum)
		free(p_thr)
		free(q_thr)
	omp.omp_destroy_lock(&mutex)

# Accelerated update P and Q temp arrays
cpdef void accelP(
		u8[:,::1] Z, f64[::1] P, f64[::1] P_new, const f64[:,::1] Q, f64[:,::1] Q_tmp, const u32[::1] k_vec, 
		const u32[::1] c_vec, const u32 L
	) noexcept nogil:
	cdef:
		Py_ssize_t W = Z.shape[0]
		Py_ssize_t N = Z.shape[1]
		Py_ssize_t K = Q.shape[1]
		Py_ssize_t B
		size_t i, l, s, w, x, y
		f64 S = 1.0/<f64>N
		f64 h
		f64* p
		f64* p_sum
		f64* p_thr
		f64* q_thr
		omp.omp_lock_t mutex
	omp.omp_init_lock(&mutex)
	with nogil, parallel():
		p_sum = <f64*>calloc(K, sizeof(f64))
		p_thr = <f64*>calloc(L*K, sizeof(f64))
		q_thr = <f64*>calloc(N*K, sizeof(f64))
		for w in prange(W, schedule='guided'):
			l = c_vec[w]
			B = k_vec[w]
			for i in range(N):
				s = Z[w,i]*K
				p = &P[l + s]
				h = _computeH(p, &Q[i,0], K)
				_innerJ(p, &Q[i,0], &p_thr[s], &q_thr[i*K], h, K)
			_outerAccelP(&P[l], &P_new[l], &p_thr[0], &p_sum[0], S, B, K)

		# omp critical
		omp.omp_set_lock(&mutex)
		for x in range(N):
			for y in range(K):
				Q_tmp[x,y] += q_thr[x*K + y]
		omp.omp_unset_lock(&mutex)
		free(p_sum)
		free(p_thr)
		free(q_thr)
	omp.omp_destroy_lock(&mutex)

# Batch accelerated update P and Q temp arrays
cpdef void accelBatchP(
		u8[:,::1] Z, f64[::1] P, f64[::1] P_new, const f64[:,::1] Q, f64[:,::1] Q_tmp, const u32[::1] k_vec, 
		const u32[::1] c_vec, const u32[::1] s_bat, const u32 L
	) noexcept nogil:
	cdef:
		Py_ssize_t W = s_bat.shape[0]
		Py_ssize_t N = Z.shape[1]
		Py_ssize_t K = Q.shape[1]
		Py_ssize_t B
		size_t i, l, r, s, w, x, y
		f64 S = 1.0/<f64>N
		f64 h
		f64* p
		f64* p_sum
		f64* p_thr
		f64* q_thr
		omp.omp_lock_t mutex
	omp.omp_init_lock(&mutex)
	with nogil, parallel():
		p_sum = <f64*>calloc(K, sizeof(f64))
		p_thr = <f64*>calloc(L*K, sizeof(f64))
		q_thr = <f64*>calloc(N*K, sizeof(f64))
		for w in prange(W, schedule='guided'):
			r = s_bat[w]
			l = c_vec[r]
			B = k_vec[r]
			for i in range(N):
				s = Z[r,i]*K
				p = &P[l + s]
				h = _computeH(p, &Q[i,0], K)
				_innerJ(p, &Q[i,0], &p_thr[s], &q_thr[i*K], h, K)
			_outerAccelP(&P[l], &P_new[l], &p_thr[0], &p_sum[0], S, B, K)

		# omp critical
		omp.omp_set_lock(&mutex)
		for x in range(N):
			for y in range(K):
				Q_tmp[x,y] += q_thr[x*K + y]
		omp.omp_unset_lock(&mutex)
		free(p_sum)
		free(p_thr)
		free(q_thr)
	omp.omp_destroy_lock(&mutex)

# Accelerated jump for P (QN)
cpdef void jumpP(
		f64[::1] P0, f64[::1] P1, f64[::1] P2, const u32[::1] k_vec, const u32[::1] c_vec, const Py_ssize_t K
	) noexcept nogil:
	cdef:
		Py_ssize_t M = P0.shape[0]
		Py_ssize_t W = k_vec.shape[0]
		Py_ssize_t B
		size_t l, w
		f64 c1, c2
		f64* p_sum
	c1 = _qnC(&P0[0], &P1[0], &P2[0], M)
	c2 = 1.0 - c1
	with nogil, parallel():
		p_sum = <f64*>calloc(K, sizeof(f64))
		for w in prange(W, schedule='guided'):
			l = c_vec[w]
			B = k_vec[w]
			_computeP(&P0[l], &P1[l], &P2[l], &p_sum[0], c1, c2, B, K)
		free(p_sum)

# Batch accelerated jump for P (QN)
cpdef void jumpBatchP(
		f64[::1] P0, const f64[::1] P1, const f64[::1] P2, const u32[::1] k_vec, const u32[::1] c_vec, 
		const u32[::1] s_bat, const Py_ssize_t K
	) noexcept nogil:
	cdef:
		Py_ssize_t W = s_bat.shape[0]
		Py_ssize_t B
		size_t l, r, w
		f64 c1, c2
	c1 = _qnBatch(&P0[0], &P1[0], &P2[0], &k_vec[0], &c_vec[0], &s_bat[0], W, K)
	c2 = 1.0 - c1
	with nogil, parallel():
		p_sum = <f64*>calloc(K, sizeof(f64))
		for w in prange(W, schedule='guided'):
			r = s_bat[w]
			l = c_vec[r]
			B = k_vec[r]
			_computeP(&P0[l], &P1[l], &P2[l], &p_sum[0], c1, c2, B, K)
		free(p_sum)

# Update Q
cpdef void updateQ(
		f64[:,::1] Q, f64[:,::1] Q_tmp, const Py_ssize_t W
	) noexcept nogil:
	cdef:
		Py_ssize_t N = Q.shape[0]
		Py_ssize_t K = Q.shape[1]
		size_t i, k
		f64 S = 1.0/<f64>W
	for i in prange(N, schedule='guided'):
		_outerQ(&Q[i,0], &Q_tmp[i,0], S, K)

# Accelerated update Q
cpdef void accelQ(
		const f64[:,::1] Q, f64[:,::1] Q_new, f64[:,::1] Q_tmp, const Py_ssize_t W
	) noexcept nogil:
	cdef:
		Py_ssize_t N = Q.shape[0]
		Py_ssize_t K = Q.shape[1]
		size_t i, k
		f64 S = 1.0/<f64>W
	for i in prange(N, schedule='guided'):
		_outerAccelQ(&Q[i,0], &Q_new[i,0], &Q_tmp[i,0], S, K)

# Accelerated jump for Q (QN)
cpdef void jumpQ(
		f64[:,::1] Q0, const f64[:,::1] Q1, const f64[:,::1] Q2
	) noexcept nogil:
	cdef:
		Py_ssize_t N = Q0.shape[0]
		Py_ssize_t K = Q0.shape[1]
		size_t i, k
		f64 a, c1, c2, sumQ
	c1 = _qnC(&Q0[0,0], &Q1[0,0], &Q2[0,0], N*K)
	c2 = 1.0 - c1
	for i in prange(N, schedule='guided'):
		_computeQ(&Q0[i,0], &Q1[i,0], &Q2[i,0], c1, c2, K)


### Other functions
# Create P matrix from array
cpdef void createP(
		f64[::1] P, const u32[::1] k_vec, const u32[::1] c_vec, const Py_ssize_t K
	) noexcept nogil:
	cdef:
		Py_ssize_t W = k_vec.shape[0]
		Py_ssize_t B
		size_t c, k, l, w
		f64* p_sum
	with nogil, parallel():
		p_sum = <f64*>calloc(K, sizeof(f64))
		for w in prange(W, schedule='guided'):
			l = c_vec[w]
			B = k_vec[w]
			for c in range(B):
				p = &P[l + c*K]
				for k in range(K):
					p_sum[k] += p[k]
			for c in range(B):
				p = &P[l + c*K]
				for k in range(K):
					p[k] /= p_sum[k]
			for k in range(K):
				p_sum[k] = 0.0
		free(p_sum)

# Log-likelihood
cpdef f64 loglike(
		u8[:,::1] Z, f64[::1] P, const f64[:,::1] Q, const u32[::1] c_vec
	) noexcept nogil:
	cdef:
		Py_ssize_t W = Z.shape[0]
		Py_ssize_t N = Z.shape[1]
		Py_ssize_t K = Q.shape[1]
		Py_ssize_t M = P.shape[0]
		size_t i, l, w
		f64 r = 0.0
		f64 h
		f64* p
	for w in prange(W, schedule='guided'):
		l = c_vec[w]
		p = &P[l]
		for i in range(N):
			r += _computeL(&p[Z[w,i]*K], &Q[i,0], K)
	return r*((<f64>K)/((<f64>M)*(<f64>N)))

# Convert ancestry proportions to individual-level from haplotype-level
cpdef void convertQ(
		const f64[:,::1] Q, f64[:,::1] Q_fin
	) noexcept nogil:
	cdef:
		Py_ssize_t N = Q_fin.shape[0]
		Py_ssize_t K = Q_fin.shape[1]
		size_t i
	for i in prange(N, schedule='guided'):
		_averageQ(&Q[2*i,0], &Q[2*i + 1,0], &Q_fin[i,0], K)

# Projection function for P (f32)
cpdef void projectP(
		f32[:,::1] P, const u32[::1] k_vec, const u32[::1] c_vec
	) noexcept nogil:
	cdef:
		Py_ssize_t W = k_vec.shape[0]
		Py_ssize_t K = P.shape[1]
		Py_ssize_t B
		size_t l, w
		f32* p_sum
	with nogil, parallel():
		p_sum = <f32*>calloc(K, sizeof(f32))
		for w in prange(W, schedule='guided'):
			l = c_vec[w]
			B = k_vec[w]
			_projectP(&P[l,0], &p_sum[0], B, K)
		free(p_sum)

# Projection function for Q (f32)
cpdef void projectQ(
		f32[:,::1] Q
	) noexcept nogil:
	cdef:
		Py_ssize_t N = Q.shape[0]
		Py_ssize_t K = Q.shape[1]
		size_t i
	for i in prange(N, schedule='guided'):
		_projectQ(&Q[i,0], K)

# Root-mean square error between two Q matrices
cpdef f32 rmseQ(
		f32[:,::1] A, f32[:,::1] B
	) noexcept nogil:
	cdef:
		Py_ssize_t N = A.shape[0]
		Py_ssize_t K = A.shape[1]
		f32 r
	r = _computeR(&A[0,0], &B[0,0], N*K)
	return sqrtf(r/(<f32>(N)*<f32>(K)))


### Supervised mode
# Initialize P based on Q and haplotype cluster assignments
cpdef void superP(
		u8[:,::1] Z, f64[:,::1] P, const u32[::1] k_vec, const u32[::1] c_vec, u8[::1] z
	) noexcept nogil:
	cdef:
		Py_ssize_t W = Z.shape[0]
		Py_ssize_t N = Z.shape[1]
		Py_ssize_t K = P.shape[1]
		Py_ssize_t B
		size_t c, k, i, l, w
		u8 y
		f64 a, b
		f64* p
		f64* c_sum
		f64* p_sum
	with nogil, parallel():
		c_sum = <f64*>calloc(K, sizeof(f64))
		p_sum = <f64*>calloc(K, sizeof(f64))
		for w in prange(W, schedule='guided'):
			l = c_vec[w]
			B = k_vec[w]
			for i in range(N):
				y = z[i]
				if y > 0:
					P[l + Z[w,i], y - 1] += 1.0
					c_sum[y - 1] += 1.0
			for c in range(B):
				p = &P[l + c,0]
				for k in range(K):
					a = p[k]
					if c_sum[k] > 0.0:
						a = a/c_sum[k]
					b = _clamp1(a)
					p_sum[k] += b
					p[k] = b
			for c in range(B):
				p = &P[l + c,0]
				for k in range(K):
					p[k] /= p_sum[k]
			for k in range(K):
				c_sum[k] = 0.0
				p_sum[k] = 0.0
		free(c_sum)
		free(p_sum)

# Update Q in supervised mode
cpdef void superQ(
		f64[:,::1] Q, const u8[::1] y
	) noexcept nogil:
	cdef:
		Py_ssize_t K = Q.shape[1]
		Py_ssize_t N = Q.shape[0]
		size_t i, k
		f64 sumQ
	for i in prange(N, schedule='guided'):
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


### Projection mode
# Check haplotype cluster frequencies input
cpdef void checkP(
		f64[::1] P, f64[:,::1] p_sum, const u32[::1] k_vec, const u32[::1] c_vec, const Py_ssize_t K
	) noexcept nogil:
	cdef:
		Py_ssize_t W = k_vec.shape[0]
		Py_ssize_t B
		f64* p
		size_t c, k, l, w
	for w in prange(W, schedule='guided'):
		l = c_vec[w]
		B = k_vec[w]
		for c in range(B):
			p = &P[l + c*K]
			for k in range(K):
				p_sum[w,k] += p[k]

# Update Q temp arrays in projection mode
cpdef void stepQ(
		u8[:,::1] Z, f64[::1] P, const f64[:,::1] Q, f64[:,::1] Q_tmp, const u32[::1] k_vec, const u32[::1] c_vec
	) noexcept nogil:
	cdef:
		Py_ssize_t W = Z.shape[0]
		Py_ssize_t N = Z.shape[1]
		Py_ssize_t K = Q.shape[1]
		size_t i, l, s, w, x, y
		f64 h
		f64* p
		f64* q_thr
		omp.omp_lock_t mutex
	omp.omp_init_lock(&mutex)
	with nogil, parallel():
		q_thr = <f64*>calloc(N*K, sizeof(f64))
		for w in prange(W, schedule='guided'):
			l = c_vec[w]
			for i in range(N):
				s = l + Z[w,i]*K
				p = &P[s]
				h = _computeH(p, &Q[i,0], K)
				_innerQ(p, &q_thr[i*K], h, K)

		# omp critical
		omp.omp_set_lock(&mutex)
		for x in range(N):
			for y in range(K):
				Q_tmp[x,y] += q_thr[x*K + y]
		omp.omp_unset_lock(&mutex)
		free(q_thr)
	omp.omp_destroy_lock(&mutex)

# Batch accelerate update Q temp arrays in projection mode
cpdef void stepBatchQ(
		u8[:,::1] Z, f64[::1] P, const f64[:,::1] Q, f64[:,::1] Q_tmp, const u32[::1] k_vec, const u32[::1] c_vec, 
		const u32[::1] s_bat
	) noexcept nogil:
	cdef:
		Py_ssize_t W = s_bat.shape[0]
		Py_ssize_t N = Z.shape[1]
		Py_ssize_t K = Q.shape[1]
		size_t i, l, r, s, w, x, y
		f64 h
		f64* p
		f64* q_thr
		omp.omp_lock_t mutex
	omp.omp_init_lock(&mutex)
	with nogil, parallel():
		q_thr = <f64*>calloc(N*K, sizeof(f64))
		for w in prange(W, schedule='guided'):
			r = s_bat[w]
			l = c_vec[r]
			for i in range(N):
				s = l + Z[r,i]*K
				p = &P[s]
				h = _computeH(p, &Q[i,0], K)
				_innerQ(p, &q_thr[i*K], h, K)

		# omp critical
		omp.omp_set_lock(&mutex)
		for x in range(N):
			for y in range(K):
				Q_tmp[x,y] += q_thr[x*K + y]
		omp.omp_unset_lock(&mutex)
		free(q_thr)
	omp.omp_destroy_lock(&mutex)
