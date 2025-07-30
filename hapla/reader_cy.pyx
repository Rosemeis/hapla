# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
cimport numpy as np
from cython.parallel import parallel, prange
from libc.stdint cimport uint8_t, int16_t, uint32_t
from libc.stdlib cimport malloc, free

ctypedef uint8_t u8
ctypedef uint32_t u32
ctypedef int16_t i16
ctypedef double f64


##### Cython functions for reading genotype files #####
# Read variant from VCF/BCF into 8-bit integer format
cpdef void readVar(
		u8[::1] G, const i16[:,::1] V, const Py_ssize_t N
	) noexcept nogil:
	cdef:
		size_t i
	for i in range(N):
		G[2*i] = V[i,0] # Allele 1
		G[2*i + 1] = V[i,1] # Allele 2

# Read variant from VCF/BCF into 8-bit integer format with missing option
cpdef void predVar(
		u8[::1] G, const i16[:,::1] V, const Py_ssize_t N
	) noexcept nogil:
	cdef:
		size_t i
	for i in range(N):
		G[2*i] = 9 if V[i,0] == -1 else V[i,0] # Allele 1
		G[2*i + 1] = 9 if V[i,1] == -1 else V[i,1] # Allele 2

# Initialize cluster mean and suffix arrays
cpdef void convertHap(
		const u8[:,::1] G, u32[:,::1] C, u32[::1] p_vec, u32[::1] d_vec, u32[::1] a_tmp, u32[::1] b_tmp, 
		u32[::1] d_tmp, u32[::1] e_tmp, const u32 S
	) noexcept nogil:
	cdef:
		Py_ssize_t M = C.shape[1]
		Py_ssize_t N = G.shape[1]
		size_t m = S
		size_t i, j, k, s, u, v
		u32 f, l, p, q
	for j in range(M):
		s = m + j
		u = v = 0
		p = q = j + 1
		C[0,j] = 0
		for i in range(N):
			# Add to cluster mean
			C[0,j] += G[s,i]

			# Suffix array updates
			f = p_vec[i]
			l = d_vec[i]
			if l > p:
				p = l
			if l > q:
				q = l
			if G[s,f] == 0:
				a_tmp[u] = f
				d_tmp[u] = p
				u += 1
				p = 0
			else:
				b_tmp[v] = f
				e_tmp[v] = q
				v += 1
				q = 0
		for k in range(u):
			p_vec[k] = a_tmp[k]
			d_vec[k] = d_tmp[k]
		for k in range(v):
			p_vec[u + k] = b_tmp[k]
			d_vec[u + k] = e_tmp[k]

# Extract unique haplotypes from suffix arrays
cpdef u32 uniqueHap(
		const u8[:,::1] G, u8[:,::1] X, const u32[::1] p_vec, const u32[::1] d_vec, u32[::1] u_vec, const u32 S
	) noexcept nogil:
	cdef:
		Py_ssize_t N = X.shape[0]
		Py_ssize_t M = X.shape[1]
		size_t u = 0
		size_t h, i, j
	for i in range(N):
		if d_vec[i] != 0:
			h = p_vec[i]
			for j in range(M):
				X[u,j] = G[S + j,h]
			u += 1
		u_vec[u - 1] += 1
	return u
			
# Convert transposed window for predicting target clusters
cpdef void predictHap(
		const u8[:,::1] G, u8[:,::1] X, const u32 S
	) noexcept nogil:
	cdef:
		Py_ssize_t N = X.shape[0]
		Py_ssize_t M = X.shape[1]
		size_t i, j
	for i in range(N):
		for j in range(M):
			X[i,j] = G[S + j,i]

# Convert haplotype cluster alleles to 2-bit PLINK format
cpdef void convertPlink(
		const u8[:,::1] Z, u8[:,::1] Z_bin, u32[:,::1] P_mat, const u32[::1] k_vec, const u32[::1] c_vec, 
		const u32[::1] b_vec
	) noexcept nogil:
	cdef:
		Py_ssize_t W = Z.shape[0]
		Py_ssize_t N = Z.shape[1]//2
		Py_ssize_t B = Z_bin.shape[1]
		size_t b, c, i, j, l, n, s, w, bit
		u8* z_vec
	with nogil, parallel():
		z_vec = <u8*>malloc(sizeof(u8)*N)
		for w in prange(W, schedule='guided'):
			s = c_vec[w]
			for c in range(k_vec[w]):
				# Create haplotype cluster alleles
				for i in range(0, 2*N, 2):
					n = i//2
					z_vec[n] = 0
					if Z[w,i] == c:
						z_vec[n] += 1
					if Z[w,i+1] == c:
						z_vec[n] += 1

				# Save in 2-bit form with bit-wise operations
				j = 0
				l = s + c
				for b in range(B):
					for bit in range(0, 8, 2):
						if z_vec[j] == 0:
							Z_bin[l,b] |= (1<<bit)
							Z_bin[l,b] |= (1<<(bit+1))
						if z_vec[j] == 1:
							Z_bin[l,b] |= (1<<(bit+1))

						# Increase counter and check for break
						j = j + 1
						if j == N:
							break
				
				# Save window and cluster information
				P_mat[l,0] = w + 1
				P_mat[l,1] = c + 1
				P_mat[l,2] = b_vec[w]
		free(z_vec)

# Convert 2-bit into standardized genotype array for phenotypes
cpdef void phenoPlink(
		const u8[:,::1] G_mat, f64[:,::1] G, const u32[::1] c
	) noexcept nogil:
	cdef:
		Py_ssize_t M = G.shape[0]
		Py_ssize_t N = G.shape[1]
		Py_ssize_t B = G_mat.shape[1]
		size_t b, i, j, bytepart
		u8[4] recode = [2, 9, 1, 0]
		u8 mask = 3
		u8 byte
	for j in range(M):
		i = 0
		for b in range(B):
			byte = G_mat[c[j],b]
			for bytepart in range(4):
				G[j,i] = <f64>recode[byte & mask]
				byte = byte >> 2
				i += 1
				if i == N:
					break
