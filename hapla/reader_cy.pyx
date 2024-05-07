# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
import numpy as np
cimport numpy as np

##### Cython function for reading VCF/BCF files #####
### Read variant from VCF/BCF into 1-bit integer format
cpdef void readVar(unsigned char[:,::1] G, const short[:,::1] V, const int j, \
		const int n) noexcept nogil:
	cdef:
		int B = G.shape[1]
		int i = 0
		int b, bit
	for b in range(B):
		for bit in range(0, 8, 2):
			if V[i,0] == 1:
				G[j,b] |= (1<<bit)
			if V[i,1] == 1:
				G[j,b] |= (1<<(bit+1))
			# Increase counter and check for break
			i += 1
			if i == n:
				break

### Read variant from VCF/BCF into 2-bit integer format
cpdef void readPred(unsigned char[:,::1] G, const short[:,::1] V, const int j, \
		const int n) noexcept nogil:
	cdef:
		int B = G.shape[1]
		int i = 0
		int b, bit
	for b in range(B):
		for bit in range(0, 8, 4):
			if V[i,0] == 1: # Allele 1 (1,1)
				G[j,b] |= (1<<bit)
				G[j,b] |= (1<<(bit+1))
			elif V[i,0] == -1: # Missing (1,0)
				G[j,b] |= (1<<(bit))
			if V[i,1] == 1: # Allele 2 (1,1)
				G[j,b] |= (1<<(bit+2))
				G[j,b] |= (1<<(bit+3))
			elif V[i,1] == -1: # Missing (1,0)
				G[j,b] |= (1<<(bit+2))
			# Increase counter and check for break
			i += 1
			if i == n:
				break

### Convert 1-bit into full array and initialize cluster mean
cpdef void convertBit(const unsigned char[:,::1] G, unsigned char[:,::1] X, \
		float[:,::1] C, const int w0) noexcept nogil:
	cdef:
		int B = G.shape[1]
		int n = X.shape[0]
		int m = X.shape[1]
		int b, i, j, bit
		unsigned char mask = 1
		unsigned char byte
	for j in range(m):
		i = 0
		C[0,j] = 0.0
		for b in range(B):
			byte = G[w0+j,b]
			for bit in range(8):
				X[i,j] = byte & mask
				C[0,j] += <float>(byte & mask)
				byte = byte >> 1 # Right shift 1 bit
				i += 1
				if i == n:
					break

### Convert 1-bit into full array for predicting target clusters
cpdef void predictBit(const unsigned char[:,::1] G, unsigned char[:,::1] X, \
		const int w0) noexcept nogil:
	cdef:
		int B = G.shape[1]
		int n = X.shape[0]
		int m = X.shape[1]
		int b, i, j, bit
		unsigned char[4] recode = [0, 9, 9, 1]
		unsigned char mask = 3
		unsigned char byte
	for j in range(m):
		i = 0
		for b in range(B):
			byte = G[w0+j,b]
			for bit in range(4):
				X[i,j] = recode[byte & mask]
				byte = byte >> 2 # Right shift 2 bit
				i += 1
				if i == n:
					break

### Convert haplotype cluster alleles to 2-bit PLINK format
cpdef void convertPlink(const unsigned char[:,::1] Z_mat, unsigned char[:,::1] Z_bin, \
		int[:,::1] P_mat, unsigned char[::1] Z_vec, const unsigned char[::1] K_vec) \
		noexcept nogil:
	cdef:
		int W = Z_mat.shape[0]
		int n = Z_mat.shape[1]//2
		int B = Z_bin.shape[1]
		int b, i, k, l, w, bit
		int j = 0
	for w in range(W):
		for k in range(K_vec[w]):
			# Create haplotype cluster alleles
			for i in range(0, 2*n, 2):
				l = <int>(i/2.0)
				Z_vec[l] = 0
				if Z_mat[w,i] == k:
					Z_vec[l] += 1
				if Z_mat[w,i+1] == k:
					Z_vec[l] += 1

			# Save in 2-bit form with bit-wise operations
			i = 0
			for b in range(B):
				for bit in range(0, 8, 2):
					if Z_vec[i] == 0:
						Z_bin[j,b] |= (1<<bit)
						Z_bin[j,b] |= (1<<(bit+1))
					if Z_vec[i] == 1:
						Z_bin[j,b] |= (1<<(bit+1))

					# Increase counter and check for break
					i += 1
					if i == n:
						break
			
			# Save window and cluster information
			P_mat[j,0] = w + 1
			P_mat[j,1] = k + 1
			j += 1

### Convert 2-bit into standardized genotype array for phenotypes
cpdef void phenoPlink(const unsigned char[:,::1] G_mat, double[:,::1] G, \
		const long[::1] c) noexcept nogil:
	cdef:
		int m = G.shape[0]
		int n = G.shape[1]
		int B = G_mat.shape[1]
		int b, i, j, bytepart
		unsigned char[4] recode = [0, 9, 1, 2]
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
