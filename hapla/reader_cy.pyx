# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free

##### Cython functions for reading genotype files #####
# Read variant from VCF/BCF into 8-bit integer format
cpdef void readVar(unsigned char[:,::1] G, const short[:,::1] V, const int j, \
		const int n) noexcept nogil:
	cdef:
		int i
	for i in range(n):
		G[j,2*i] = <unsigned char>V[i,0]
		G[j,2*i+1] = <unsigned char>V[i,1]

# Read variant from VCF/BCF into 8-bit integer format with missing option
cpdef void predVar(unsigned char[:,::1] G, const short[:,::1] V, const int j, \
		const int n) noexcept nogil:
	cdef:
		int i
	for i in range(n):
		# Allele 1
		if V[i,0] == -1:
			G[j,2*i] = 9
		else:
			G[j,2*i] = <unsigned char>V[i,0]

		# Allele 2
		if V[i,0] == -1:
			G[j,2*i+1] = 9
		else:
			G[j,2*i+1] = <unsigned char>V[i,1]

# Initialize cluster mean and suffix arrays
cpdef void convertHap(const unsigned char[:,::1] G, float[:,::1] C, int[::1] p_vec, \
		int[::1] d_vec, int[::1] a_tmp, int[::1] b_tmp, int[::1] d_tmp, \
		int[::1] e_tmp, const int s) noexcept nogil:
	cdef:
		int m = C.shape[1]
		int n = G.shape[1]
		int b, f, i, j, k, l, p, q, u, v
	for j in range(m):
		b = s + j
		u = v = 0
		p = q = j + 1
		C[0,j] = 0.0
		for i in range(n):
			# Add to cluster mean
			C[0,j] += <float>G[b,i]

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
cpdef int uniqueHap(const unsigned char[:,::1] G, unsigned char[:,::1] X, \
		const int[::1] p_vec, const int[::1] d_vec, int[::1] u_vec, const int s) \
		noexcept nogil:
	cdef:
		int n = X.shape[0]
		int m = X.shape[1]
		int h, i, j
		int u = 0
	for i in range(n):
		if d_vec[i] != 0:
			h = p_vec[i]
			for j in range(m):
				X[u,j] = G[s+j,h]
			u += 1
		u_vec[u-1] += 1
	return u

# Create interval for individual-based threading
cpdef void intervalThr(int[:,::1] I_thr, const int U, const int B) noexcept nogil:
	cdef:
		int t = I_thr.shape[0]
		int thr
	for thr in range(t-1):
		I_thr[thr,0] = thr*B
		I_thr[thr,1] = (thr+1)*B
	I_thr[t-1,0] = (t-1)*B
	I_thr[t-1,1] = U
			
# Convert transposed window for predicting target clusters
cpdef void predictHap(const unsigned char[:,::1] G, unsigned char[:,::1] X, \
		const int s) noexcept nogil:
	cdef:
		int n = X.shape[0]
		int m = X.shape[1]
		int i, j
	for i in range(n):
		for j in range(m):
			X[i,j] = G[s+j,i]

# Convert haplotype cluster alleles to 2-bit PLINK format
cpdef void convertPlink(const unsigned char[:,::1] Z_mat, unsigned char[:,::1] Z_bin, \
		int[:,::1] P_mat, const unsigned char[::1] k_vec, const int[::1] b_vec) \
		noexcept nogil:
	cdef:
		int W = Z_mat.shape[0]
		int n = Z_mat.shape[1]//2
		int B = Z_bin.shape[1]
		int b, i, k, l, w, bit
		int j = 0
		unsigned char* z_vec = <unsigned char*>malloc(sizeof(unsigned char)*n)
	for w in range(W):
		for k in range(k_vec[w]):
			# Create haplotype cluster alleles
			for i in range(0, 2*n, 2):
				l = i//2
				z_vec[l] = 0
				if Z_mat[w,i] == k:
					z_vec[l] += 1
				if Z_mat[w,i+1] == k:
					z_vec[l] += 1

			# Save in 2-bit form with bit-wise operations
			i = 0
			for b in range(B):
				for bit in range(0, 8, 2):
					if z_vec[i] == 0:
						Z_bin[j,b] |= (1<<bit)
						Z_bin[j,b] |= (1<<(bit+1))
					if z_vec[i] == 1:
						Z_bin[j,b] |= (1<<(bit+1))

					# Increase counter and check for break
					i += 1
					if i == n:
						break
			
			# Save window and cluster information
			P_mat[j,0] = w + 1
			P_mat[j,1] = k + 1
			P_mat[j,2] = b_vec[w]
			j += 1
	free(z_vec)

# Convert 2-bit into standardized genotype array for phenotypes
cpdef void phenoPlink(const unsigned char[:,::1] G_mat, double[:,::1] G, \
		const long[::1] c) noexcept nogil:
	cdef:
		int m = G.shape[0]
		int n = G.shape[1]
		int B = G_mat.shape[1]
		int b, i, j, bytepart
		unsigned char[4] recode = [2, 9, 1, 0]
		unsigned char mask = 3
		unsigned char byte
	for j in range(m):
		i = 0
		for b in range(B):
			byte = G_mat[c[j],b]
			for bytepart in range(4):
				G[j,i] = <double>recode[byte & mask]
				byte = byte >> 2
				i += 1
				if i == n:
					break
