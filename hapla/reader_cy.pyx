# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
import numpy as np
cimport numpy as np
from cython.parallel import parallel, prange
from libc.stdlib cimport malloc, free

##### Cython functions for reading genotype files #####
# Read variant from VCF/BCF into 8-bit integer format
cpdef void readVar(unsigned char[::1] G, const short[:,::1] V, const size_t N) \
		noexcept nogil:
	cdef:
		size_t i
	for i in range(N):
		G[2*i] = <unsigned char>V[i,0] # Allele 1
		G[2*i+1] = <unsigned char>V[i,1] # Allele 2

# Read variant from VCF/BCF into 8-bit integer format with missing option
cpdef void predVar(unsigned char[::1] G, const short[:,::1] V, const size_t N) \
		noexcept nogil:
	cdef:
		size_t i
	for i in range(N):
		# Allele 1
		if V[i,0] == -1:
			G[2*i] = 9
		else:
			G[2*i] = <unsigned char>V[i,0]

		# Allele 2
		if V[i,0] == -1:
			G[2*i+1] = 9
		else:
			G[2*i+1] = <unsigned char>V[i,1]

# Initialize cluster mean and suffix arrays
cpdef void convertHap(const unsigned char[:,::1] G, unsigned int[:,::1] C, \
		unsigned int[::1] p_vec, unsigned int[::1] d_vec, unsigned int[::1] a_tmp, \
		unsigned int[::1] b_tmp, unsigned int[::1] d_tmp, unsigned int[::1] e_tmp, \
		const size_t S) noexcept nogil:
	cdef:
		size_t M = C.shape[1]
		size_t N = G.shape[1]
		size_t b, i, j, k, u, v
		unsigned int f, l, p, q
	for j in range(M):
		b = S+j
		u = v = 0
		p = q = j+1
		C[0,j] = 0
		for i in range(N):
			# Add to cluster mean
			C[0,j] += G[b,i]

			# Suffix array updates
			f = p_vec[i]
			l = d_vec[i]
			if l > p:
				p = l
			if l > q:
				q = l
			if G[b,f] == 0:
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
cpdef unsigned int uniqueHap(const unsigned char[:,::1] G, unsigned char[:,::1] X, \
		const unsigned int[::1] p_vec, const unsigned int[::1] d_vec, \
		unsigned int[::1] u_vec, const size_t S) noexcept nogil:
	cdef:
		size_t N = X.shape[0]
		size_t M = X.shape[1]
		size_t h, i, j
		unsigned int u = 0
	for i in range(N):
		if d_vec[i] != 0:
			h = p_vec[i]
			for j in range(M):
				X[u,j] = G[S+j,h]
			u += 1
		u_vec[u-1] += 1
	return u
			
# Convert transposed window for predicting target clusters
cpdef void predictHap(const unsigned char[:,::1] G, unsigned char[:,::1] X, \
		const size_t S) noexcept nogil:
	cdef:
		size_t N = X.shape[0]
		size_t M = X.shape[1]
		size_t i, j
	for i in range(N):
		for j in range(M):
			X[i,j] = G[S+j,i]

# Convert haplotype cluster alleles to 2-bit PLINK format
cpdef void convertPlink(const unsigned char[:,::1] Z, unsigned char[:,::1] Z_bin, \
		unsigned int[:,::1] P_mat, const unsigned char[::1] k_vec, \
		const unsigned int[::1] c_vec, const unsigned int[::1] b_vec) noexcept nogil:
	cdef:
		size_t W = Z.shape[0]
		size_t N = Z.shape[1]//2
		size_t B = Z_bin.shape[1]
		size_t b, i, c, l, n, s, w, bit
		unsigned char* z_vec
	with nogil, parallel():
		z_vec = <unsigned char*>malloc(sizeof(unsigned char)*N)
		for w in prange(W):
			s = <size_t>c_vec[w]
			for c in range(k_vec[w]):
				# Create haplotype cluster alleles
				l = s + c
				for i in range(0, 2*N, 2):
					n = i//2
					z_vec[n] = 0
					if Z[w,i] == c:
						z_vec[n] += 1
					if Z[w,i+1] == c:
						z_vec[n] += 1

				# Save in 2-bit form with bit-wise operations
				i = 0
				for b in range(B):
					for bit in range(0, 8, 2):
						if z_vec[i] == 0:
							Z_bin[l,b] |= (1<<bit)
							Z_bin[l,b] |= (1<<(bit+1))
						if z_vec[i] == 1:
							Z_bin[l,b] |= (1<<(bit+1))

						# Increase counter and check for break
						i = i + 1
						if i == N:
							break
				
				# Save window and cluster information
				P_mat[l,0] = w + 1
				P_mat[l,1] = c + 1
				P_mat[l,2] = b_vec[w]
		free(z_vec)

# Convert 2-bit into standardized genotype array for phenotypes
cpdef void phenoPlink(const unsigned char[:,::1] G_mat, double[:,::1] G, \
		const unsigned int[::1] c) noexcept nogil:
	cdef:
		size_t M = G.shape[0]
		size_t N = G.shape[1]
		size_t B = G_mat.shape[1]
		size_t b, i, j, bytepart
		unsigned char[4] recode = [2, 9, 1, 0]
		unsigned char mask = 3
		unsigned char byte
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
