# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
cimport numpy as np
from cython.parallel import parallel, prange
from libc.stdint cimport uint8_t, int16_t, uint32_t
from libc.stdlib cimport malloc, free

##### Cython functions for reading genotype files #####
# Read variant from VCF/BCF into 8-bit integer format
cpdef void readVar(
		uint8_t[::1] G, const int16_t[:,::1] V, const uint32_t N
	) noexcept nogil:
	cdef size_t i
	for i in range(N):
		G[2*i] = <uint8_t>V[i,0] # Allele 1
		G[2*i+1] = <uint8_t>V[i,1] # Allele 2

# Read variant from VCF/BCF into 8-bit integer format with missing option
cpdef void predVar(
		uint8_t[::1] G, const int16_t[:,::1] V, const uint32_t N
	) noexcept nogil:
	cdef size_t i
	for i in range(N):
		G[2*i] = 9 if V[i,0] == -1 else <uint8_t>V[i,0] # Allele 1
		G[2*i+1] = 9 if V[i,1] == -1 else <uint8_t>V[i,1] # Allele 2

# Initialize cluster mean and suffix arrays
cpdef void convertHap(
		const uint8_t[:,::1] G, uint32_t[:,::1] C, uint32_t[::1] p_vec, uint32_t[::1] d_vec, uint32_t[::1] a_tmp, 
		uint32_t[::1] b_tmp, uint32_t[::1] d_tmp, uint32_t[::1] e_tmp, const uint32_t S
	) noexcept nogil:
	cdef:
		uint32_t M = C.shape[1]
		uint32_t N = G.shape[1]
		uint32_t f, l, p, q
		size_t i, j, k, s, u, v
	for j in range(M):
		s = <size_t>S + j
		u = v = 0
		p = q = <uint32_t>j + 1
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
			p_vec[u+k] = b_tmp[k]
			d_vec[u+k] = e_tmp[k]

# Extract unique haplotypes from suffix arrays
cpdef uint32_t uniqueHap(
		const uint8_t[:,::1] G, uint8_t[:,::1] X, const uint32_t[::1] p_vec, const uint32_t[::1] d_vec, 
		uint32_t[::1] u_vec, const size_t S
	) noexcept nogil:
	cdef:
		uint32_t N = X.shape[0]
		uint32_t M = X.shape[1]
		size_t u = 0
		size_t h, i, j
	for i in range(N):
		if d_vec[i] != 0:
			h = <size_t>p_vec[i]
			for j in range(M):
				X[u,j] = G[S+j,h]
			u += 1
		u_vec[u - 1] += 1
	return u
			
# Convert transposed window for predicting target clusters
cpdef void predictHap(
		const uint8_t[:,::1] G, uint8_t[:,::1] X, const size_t S
	) noexcept nogil:
	cdef:
		uint32_t N = X.shape[0]
		uint32_t M = X.shape[1]
		size_t i, j
	for i in range(N):
		for j in range(M):
			X[i,j] = G[S+j,i]

# Convert haplotype cluster alleles to 2-bit PLINK format
cpdef void convertPlink(
		const uint8_t[:,::1] Z, uint8_t[:,::1] Z_bin, uint32_t[:,::1] P_mat, const uint8_t[::1] k_vec, 
		const uint32_t[::1] c_vec, const uint32_t[::1] b_vec
	) noexcept nogil:
	cdef:
		uint8_t* z_vec
		uint32_t W = Z.shape[0]
		uint32_t N = Z.shape[1]//2
		uint32_t B = Z_bin.shape[1]
		size_t b, c, i, j, l, n, s, w, bit
	with nogil, parallel():
		z_vec = <uint8_t*>malloc(sizeof(uint8_t)*N)
		for w in prange(W):
			s = <size_t>c_vec[w]
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
				l = s+c
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
				P_mat[l,0] = <uint32_t>(w + 1)
				P_mat[l,1] = <uint32_t>(c + 1)
				P_mat[l,2] = b_vec[w]
		free(z_vec)

# Convert 2-bit into standardized genotype array for phenotypes
cpdef void phenoPlink(
		const uint8_t[:,::1] G_mat, double[:,::1] G, const uint32_t[::1] c
	) noexcept nogil:
	cdef:
		uint8_t[4] recode = [2, 9, 1, 0]
		uint8_t mask = 3
		uint8_t byte
		uint32_t M = G.shape[0]
		uint32_t N = G.shape[1]
		uint32_t B = G_mat.shape[1]
		size_t b, i, j, bytepart
	for j in range(M):
		i = 0
		for b in range(B):
			byte = G_mat[c[j],b]
			for bytepart in range(4):
				G[j,i] = <double>recode[byte & mask]
				byte = byte >> 2
				i += 1
				if i == N:
					break
