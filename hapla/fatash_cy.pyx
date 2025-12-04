# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
cimport numpy as np
cimport openmp as omp
from cython.parallel import parallel, prange
from libc.math cimport exp, fmax, fmin, log
from libc.stdint cimport uint8_t, uint32_t
from libc.stdlib cimport abort, calloc, free

ctypedef uint8_t u8
ctypedef uint32_t u32
ctypedef float f32
ctypedef double f64

cdef f64 PRO_MIN = 1e-5
cdef f64 PRO_MAX = 1.0 - (1e-5)
cdef f64 ACC_MIN = 1.0
cdef f64 ACC_MAX = 128.0
cdef inline f64 _clamp1(f64 a) noexcept nogil: return fmax(PRO_MIN, fmin(a, PRO_MAX))
cdef inline f64 _clamp2(f64 a) noexcept nogil: return fmax(ACC_MIN, fmin(a, ACC_MAX))


##### hapla - local ancestry inference #####
### Inline functions
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
		f32* L, const u32 B
	) noexcept nogil:
	cdef:
		size_t c1, c2
		f32 sumC, tmpC
		f32* l
	for c1 in range(B):
		l = &L[c1*B]
		sumC = 0.0
		for c2 in range(B):
			tmpC = exp(l[c2])
			sumC += tmpC
			l[c2] = tmpC
		for c2 in range(B):
			l[c2] /= sumC

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

# Inner loop updates for temp Q
cdef inline void _innerQ(
		const f64* p, f64* q_thr, const f64 h, const Py_ssize_t K
	) noexcept nogil:
	cdef:
		size_t k
	for k in range(K):
		q_thr[k] += p[k]*h

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
	for i in prange(I, schedule='guided'):
		u = v1[i] - v0[i]
		v = v2[i] - v1[i] - u
		sum1 += u*u
		sum2 += u*v
	f = -(sum1/sum2)
	return _clamp2(f)

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

# Compute transition probabilities (T[k1,k2] = P(z_w = k1 | z_{w-1} = k2))
cdef inline void _trans(
		f64* T, const f64* q, const f64 e, const Py_ssize_t K
	) noexcept nogil:
	cdef:
		size_t k1, k2
		f64* t
	for k1 in range(K):
		t = &T[k1*K]
		for k2 in range(K):
			if k1 == k2:
				t[k2] = log((1.0 - e)*q[k1] + e)
			else:
				t[k2] = log((1.0 - e)*q[k1])

# Compute Viterbi scores
cdef inline void _viterbi(
		u8* I, f64* E, f64* A, f64* T, f64* q, const Py_ssize_t W, const Py_ssize_t K
	) noexcept nogil:
	cdef:
		size_t w, k1, k2
		u8* i
		f64 tmp_k
		f64* a
		f64* e
		f64* t
	# First step
	for k1 in range(K):
		A[k1] = E[k1] + q[k1]

	# Loop through sequence
	for w in range(1, W):
		a = &A[w*K]
		e = &E[w*K]
		i = &I[w*K]
		for k1 in range(K):
			t = &T[k1*K]
			a[k1] = A[(w - 1)*K] + t[0]
			i[k1] = 0
			for k2 in range(1, K):
				tmp_k = A[(w - 1)*K + k2] + t[k2]
				if tmp_k > a[k1]:
					a[k1] = tmp_k
					i[k1] = k2
			a[k1] += e[k1]

# Decode Viterbi path
cdef inline void _decode(
		u8* I, u8* d, f64* a, const Py_ssize_t W, const Py_ssize_t K
	) noexcept nogil:
	cdef:
		size_t k, w
		f64 max_k
	# Last step
	d[W - 1] = 0
	max_k = a[0]
	for k in range(1, K):
		if a[k] > max_k:
			d[W - 1] = k
			max_k = a[k]

	# Decode full path
	for w in range(W - 2, -1, -1):
		d[w] = I[(w + 1)*K + d[w + 1]]

# Compute forward scores
cdef inline f64 _forward(
		f64* E, f64* A, f64* T, f64* q, f64* v, const Py_ssize_t W, const Py_ssize_t K
	) noexcept nogil:
	cdef:
		size_t w, k1, k2
		f64* a
		f64* e
		f64* t
	# Forward calculations
	for k1 in range(K):
		A[k1] = E[k1] + q[k1]
	for w in range(1, W):
		a = &A[(w - 1)*K]
		e = &E[w*K]
		for k1 in range(K):
			t = &T[k1*K]
			for k2 in range(K):
				v[k2] = a[k2] + t[k2]
			A[w*K + k1] = _logsumexp(v, K) + e[k1]

	# Forward log-likelihood
	a = &A[(W - 1)*K]
	for k1 in range(K):
		v[k1] = a[k1]
	return _logsumexp(v, K)

# Compute backward scores
cdef inline void _backward(
		f64* E, f64* L, f64* A, f64* B, f64* T, f64* v, const f64 l_fwd, const Py_ssize_t W, const Py_ssize_t K
	) noexcept nogil:
	cdef:
		size_t w, k1, k2
		f64* a
		f64* b
		f64* e
		f64* l
	# Backward calculations
	b = &B[(W - 1)*K]
	for k1 in range(K):
		b[k1] = 0.0
	for w in range(W - 2, -1, -1):
		b = &B[(w + 1)*K]
		e = &E[(w + 1)*K]
		for k1 in range(K):
			for k2 in range(K):
				v[k2] = b[k2] + e[k2] + T[k2*K + k1]
			B[w*K + k1] = _logsumexp(v, K)

	# Compute posterior probabilities
	for w in range(W):
		a = &A[w*K]
		b = &B[w*K]
		l = &L[w*K]
		for k1 in range(K):
			l[k1] = exp(a[k1] + b[k1] - l_fwd)


### Standard functions
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
		 const u8[:,::1] Z, f64[:,:,::1] E, f64[::1] P, const u32[::1] c_vec
	) noexcept nogil:
	cdef:
		Py_ssize_t N = E.shape[0]
		Py_ssize_t W = E.shape[1]
		Py_ssize_t K = E.shape[2]
		size_t i, k, w
		f64* e
		f64* p
	for i in prange(N, schedule='guided'):
		for w in range(W):
			e = &E[i,w,0]
			p = &P[c_vec[w] + Z[i,w]*K]
			for k in range(K):
				e[k] = log(p[k])

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
			z = Z[i,w]
			s = c_vec[w]
			x = x_vec[w]
			B = k_vec[w]
			l = &L[x + z*B]
			p = &P[c_vec[w] + z*K]
			for k in range(K):
				for c in range(B):
					E[i,w,k] += <f64>l[c]*p[k]
				E[i,w,k] = log(E[i,w,k])


## Multithreaded functions
# Viterbi algorithm
cpdef void viterbi(
		f64[:,:,::1] E, u8[:,::1] D, f64[:,::1] Q, f64[:,::1] Q_log, const f64 alpha
	) noexcept nogil:
	cdef:
		Py_ssize_t N = E.shape[0]
		Py_ssize_t W = E.shape[1]
		Py_ssize_t K = E.shape[2]
		size_t i
		u8* i_thr
		f64 e = exp(-(alpha))
		f64* a_thr
		f64* t_thr
	with nogil, parallel():
		# Thread-local buffer allocation
		i_thr = <u8*>calloc(W*K, sizeof(u8))
		if i_thr is NULL:
			abort()
		a_thr = <f64*>calloc(W*K, sizeof(f64))
		if a_thr is NULL:
			abort()
		t_thr = <f64*>calloc(K*K, sizeof(f64))
		if t_thr is NULL:
			abort()

		for i in prange(N, schedule='guided'):
			_trans(t_thr, &Q[i,0], e, K)
			_viterbi(i_thr, &E[i,0,0], a_thr, t_thr, &Q_log[i,0], W, K)
			_decode(i_thr, &D[i,0], &a_thr[(W - 1)*K], W, K)
		free(i_thr)
		free(a_thr)
		free(t_thr)

# Forward-backward algorithm
cpdef void fwdbwd(
		f64[:,:,::1] E, f64[:,:,::1] L, f64[:,::1] Q, f64[:,::1] Q_log, const f64 alpha
	) noexcept nogil:
	cdef:
		Py_ssize_t N = E.shape[0]
		Py_ssize_t W = E.shape[1]
		Py_ssize_t K = E.shape[2]
		size_t i
		f64 e = exp(-(alpha))
		f64 l_fwd
		f64* a_thr
		f64* b_thr
		f64* t_thr
		f64* v_thr
	with nogil, parallel():
		# Thread-local buffer allocation
		a_thr = <f64*>calloc(W*K, sizeof(f64))
		if a_thr is NULL:
			abort()
		b_thr = <f64*>calloc(W*K, sizeof(f64))
		if b_thr is NULL:
			abort()
		t_thr = <f64*>calloc(K*K, sizeof(f64))
		if t_thr is NULL:
			abort()
		v_thr = <f64*>calloc(K, sizeof(f64))
		if v_thr is NULL:
			abort()

		for i in prange(N, schedule='guided'):
			_trans(t_thr, &Q[i,0], e, K)
			l_fwd = _forward(&E[i,0,0], a_thr, t_thr, &Q_log[i,0], v_thr, W, K)
			_backward(&E[i,0,0], &L[i,0,0], a_thr, b_thr, t_thr, v_thr, l_fwd, W, K)
		free(a_thr)
		free(b_thr)
		free(t_thr)
		free(v_thr)

# Majority voting across multiple alpha values
cpdef void voting(
		const u8[:,:,::1] B, u8[:,::1] D, const Py_ssize_t K
	) noexcept nogil:
	cdef:
		Py_ssize_t A = B.shape[0]
		Py_ssize_t N = B.shape[1]
		Py_ssize_t W = B.shape[2]
		size_t a, i, k, w
		u8 k_cnt, k_idx
		u8* k_thr
	with nogil, parallel():
		# Thread-local buffer allocation
		k_thr = <u8*>calloc(K, sizeof(u8))
		if k_thr is NULL:
			abort()

		for i in prange(N):
			for w in range(W):
				for a in range(A):
					k_thr[B[a,i,w]] += 1
				k_cnt = k_thr[0]
				k_idx = 0
				k_thr[0] = 0
				for k in range(1, K):
					if k_thr[k] > k_cnt:
						k_cnt = k_thr[k]
						k_idx = k
					k_thr[k] = 0
				D[i,w] = k_idx
		free(k_thr)


## Estimate file-specific ancestry proportions
# Log-likelihood
cpdef f64 loglike(
		u8[:,::1] Z, f64[::1] P, const f64[:,::1] Q, const u32[::1] c_vec, const Py_ssize_t M
	) noexcept nogil:
	cdef:
		Py_ssize_t W = Z.shape[0]
		Py_ssize_t N = Z.shape[1]
		Py_ssize_t K = Q.shape[1]
		size_t i, l, w
		f64 r = 0.0
		f64 h
		f64* p
	for w in prange(W, schedule='guided'):
		l = c_vec[w]
		p = &P[l]
		for i in range(N):
			r += _computeL(&p[Z[w,i]*K], &Q[i,0], K)
	return r/((<f64>M)*(<f64>N))

# Update Q temp arrays
cpdef void stepQ(
		u8[:,::1] Z, f64[::1] P, const f64[:,::1] Q, f64[:,::1] Q_tmp, const u32[::1] c_vec
	) noexcept nogil:
	cdef:
		Py_ssize_t W = Z.shape[0]
		Py_ssize_t N = Z.shape[1]
		Py_ssize_t K = Q.shape[1]
		size_t i, l, w, x, y
		f64 h
		f64* p
		f64* q_thr
		omp.omp_lock_t mutex
	omp.omp_init_lock(&mutex)
	with nogil, parallel():
		# Thread-local buffer allocation
		q_thr = <f64*>calloc(N*K, sizeof(f64))
		if q_thr is NULL:
			abort()

		for w in prange(W, schedule='guided'):
			l = c_vec[w]
			for i in range(N):
				p = &P[l + Z[w,i]*K]
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

# Batch accelerate update Q temp arrays
cpdef void stepBatchQ(
		u8[:,::1] Z, f64[::1] P, const f64[:,::1] Q, f64[:,::1] Q_tmp, const u32[::1] c_vec, const u32[::1] s_bat
	) noexcept nogil:
	cdef:
		Py_ssize_t W = s_bat.shape[0]
		Py_ssize_t N = Z.shape[1]
		Py_ssize_t K = Q.shape[1]
		size_t i, l, r, w, x, y
		f64 h
		f64* p
		f64* q_thr
		omp.omp_lock_t mutex
	omp.omp_init_lock(&mutex)
	with nogil, parallel():
		# Thread-local buffer allocation
		q_thr = <f64*>calloc(N*K, sizeof(f64))
		if q_thr is NULL:
			abort()

		for w in prange(W, schedule='guided'):
			r = s_bat[w]
			l = c_vec[r]
			for i in range(N):
				p = &P[l + Z[r,i]*K]
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
		f64 a, c1, c2
	c1 = _qnC(&Q0[0,0], &Q1[0,0], &Q2[0,0], N*K)
	c2 = 1.0 - c1
	for i in prange(N, schedule='guided'):
		_computeQ(&Q0[i,0], &Q1[i,0], &Q2[i,0], c1, c2, K)