# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
import numpy as np
cimport numpy as np
from cpython.mem cimport PyMem_RawMalloc, PyMem_RawFree
from cython.parallel import prange
from libcpp.vector cimport vector
from libc.math cimport sqrt

DTYPE = np.uint8
ctypedef np.uint8_t DTYPE_t
DTYPE2 = np.int16
ctypedef np.int16_t DTYPE2_t
ctypedef vector[unsigned char] char_vec

##### Cython function for reading VCF/BCF files #####
### Read VCF/BCF into 1-bit integer matrix
cpdef np.ndarray[DTYPE_t, ndim=2] readVCF(v_file, int n, int B):
	cdef:
		int b, i, j, m, bit
		np.ndarray[DTYPE_t, ndim=2] G_np
		np.ndarray[DTYPE2_t, ndim=2] geno
		char_vec G_var = char_vec(B)
		vector[char_vec] G
		unsigned char *G_ptr
	for var in v_file: # Loop through VCF file
		i = 0
		geno = var.genotype.array()
		for b in range(B):
			G_var[b] = 0
			for bit in range(0, 8, 2):
				if geno[i,0] == 1:
					G_var[b] |= (1<<bit)
				if geno[i,1] == 1:
					G_var[b] |= (1<<(bit+1))
				# Increase counter and check for break
				i += 1
				if i == n:
					break
		G.push_back(G_var)
	m = G.size()
	
	# Fill up and return NumPy array
	G_np = np.empty((m, B), dtype=DTYPE)
	for j in range(m):
		G_ptr = &G[j][0]
		G_np[j] = np.asarray(<unsigned char[:B]>G_ptr)
	return G_np

### Read VCF/BCF into 2-bit integer matrix for cluster prediction
cpdef np.ndarray[DTYPE_t, ndim=2] readPredict(v_file, int n, int B):
	cdef:
		int a, b, i, j, m, bit
		np.ndarray[DTYPE_t, ndim=2] G_np
		np.ndarray[DTYPE2_t, ndim=2] geno
		char_vec G_var = char_vec(B)
		vector[char_vec] G
		unsigned char *G_ptr
	for var in v_file: # Loop through VCF file
		i = 0
		geno = var.genotype.array()
		for b in range(B):
			G_var[b] = 0
			for bit in range(0, 8, 4):
				if geno[i,0] == 1: # Allele 1 (1,1)
					G_var[b] |= (1<<bit)
					G_var[b] |= (1<<(bit+1))
				if geno[i,0] == -1: # Missing (1,0)
					G_var[b] |= (1<<bit)
				if geno[i,1] == 1: # Allele 1 (1,1)
					G_var[b] |= (1<<(bit+2))
					G_var[b] |= (1<<(bit+3))
				if geno[i,1] == -1: # Missing (1,0)
					G_var[b] |= (1<<(bit+2))
				# Increase counter and check for break
				i += 1
				if i == n:
					break
		G.push_back(G_var)
	m = G.size()
	
	# Fill up and return NumPy array
	G_np = np.empty((m, B), dtype=DTYPE)
	for j in range(m):
		G_ptr = &G[j][0]
		G_np[j] = np.asarray(<unsigned char[:B]>G_ptr)
	return G_np

### Convert 1-bit into full array and initialize cluster mean
cpdef void convertBit(unsigned char[:,::1] G, unsigned char[:,::1] H, \
		float[:,::1] C, int w0) nogil:
	cdef:
		int B = G.shape[1]
		int m = H.shape[0]
		int n = H.shape[1]
		int b, i, j, bit
		unsigned char mask = 1
		unsigned char byte
	for j in range(m):
		i = 0
		C[0,j] = 0.0
		for b in range(B):
			byte = G[w0+j,b]
			for bit in range(8):
				H[j,i] = byte & mask
				C[0,j] = C[0,j] + H[j,i]
				byte = byte >> 1 # Right shift 1 bit
				i += 1
				if i == n:
					break

### Convert 1-bit into full array for predicting target clusters
cpdef void predictBit(unsigned char[:,::1] G, unsigned char[:,::1] H, int w0) nogil:
	cdef:
		int B = G.shape[1]
		int m = H.shape[0]
		int n = H.shape[1]
		int b, i, j, bit
		unsigned char[4] recode = [0, 9, 9, 1]
		unsigned char mask = 3
		unsigned char byte
	for j in range(m):
		i = 0
		for b in range(B):
			byte = G[w0+j,b]
			for bit in range(4):
				H[j,i] = recode[byte & mask]
				byte = byte >> 2 # Right shift 2 bit
				i += 1
				if i == n:
					break

### Convert haplotype cluster alleles to 2-bit PLINK format
cpdef void convertPlink(unsigned char[:,::1] Z_mat, unsigned char[:,::1] Z_bin, \
		int[:,::1] P_mat, unsigned char[::1] Z_vec, unsigned char[::1] K_vec) nogil:
	cdef:
		int W = Z_mat.shape[0]
		int n = Z_mat.shape[1]//2
		int B = Z_bin.shape[1]
		int b, i, k, w, bit
		int j = 0
	for w in range(W):
		for k in range(K_vec[w]):
			# Create haplotype cluster alleles
			for i in range(0, 2*n, 2):
				Z_vec[i//2] = 0
				if Z_mat[w,i] == k:
					Z_vec[i//2] += 1
				if Z_mat[w,i+1] == k:
					Z_vec[i//2] += 1

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

### Convert 1-bit into standardized genotype array for phenotypes
cpdef void phenoVCF(unsigned char[:,::1] G_mat, double[:,::1] G, long[::1] p) nogil:
	cdef:
		int m = G.shape[0]
		int n = G.shape[1]
		int B = G_mat.shape[1]
		int b, i, j, bit
		unsigned char mask = 1
		unsigned char byte
	for j in range(m):
		i = 0
		for b in range(B):
			byte = G_mat[p[j],b]
			for bit in range(0, 8, 2):
				G[j,i] = <double>(byte & mask)
				byte = byte >> 1 # Right shift 1 bit
				G[j,i] += <double>(byte & mask)
				byte = byte >> 1 # Right shift 1 bit
				i += 1
				if i == n:
					break

### Convert 2-bit into standardized genotype array for phenotypes
cpdef void phenoPlink(unsigned char[:,::1] G_mat, double[:,::1] G, long[::1] p) nogil:
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
			byte = G_mat[p[j],b]
			for bytepart in range(4):
				G[j,i] = <double>recode[byte & mask]
				byte = byte >> 2
				i += 1
				if i == n:
					break

### Convert haplotype cluster alleles to standardized array for phenotypes
cpdef void phenoHaplo(unsigned char[:,::1] Z, double[:,::1] G, \
		unsigned char[::1] K_vec, long[::1] C) nogil:
	cdef:
		int W = Z.shape[0]
		int m = G.shape[0]
		int n = G.shape[1]
		int i, k, w
		int b = 0
		int j = 0
	for w in range(W):
		for k in range(K_vec[w]):
			if b == C[j]:
				for i in range(2*n):
					if Z[w,i] == k:
						G[j,i//2] += 1
				j += 1
			b += 1

### Filter out variants from haplotype clustering and fix window sizes
cpdef void filterSNPs(unsigned char[:,::1] Gt, long[::1] W, unsigned char[::1] mask) \
		nogil:
	cdef:
		int m = Gt.shape[0]
		int B = Gt.shape[1]
		int s = W.shape[0]
		int c = 0
		int b, j, k
		int* count
	count = <int*>PyMem_RawMalloc(sizeof(int)*s)
	for k in range(s):
		count[k] = 0
	for j in range(m):
		if mask[j] == 1:
			for b in range(B):
				Gt[c,b] = Gt[j,b]
			c += 1
		else:
			for k in range(1, s):
				if (W[k] + count[k]) >= j:
					W[k] -= 1
					count[k] += 1
	PyMem_RawFree(count)

### Read VCF/BCF position information
cpdef void readPOS(v_file, double[:,::1] P):
	cdef:
		int j = 0
	for var in v_file: # Loop through VCF file
		P[j,1] = <double>var.POS
		j += 1
