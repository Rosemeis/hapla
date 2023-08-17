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
DTYPE3 = np.int64
ctypedef np.int64_t DTYPE3_t
ctypedef vector[unsigned char] char_vec

##### Cython function for reading VCF/BCF files #####
### Read VCF/BCF into 1-bit integer matrix
cpdef np.ndarray[DTYPE_t, ndim=2] readVCF(v_file, const int n, const int B):
	cdef:
		int b, i, j, m, bit
		np.ndarray[DTYPE2_t, ndim=2] geno
		char_vec G_var
		vector[char_vec] G
	G_var = char_vec(B)
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
				i = i + 1
				if i == n:
					break
		G.push_back(G_var)
	m = G.size()
	# Fill up and return NumPy array
	cdef:
		np.ndarray[DTYPE_t, ndim=2] G_np = np.empty((m, B), dtype=DTYPE)
		unsigned char *G_ptr
	for j in range(m):
		G_ptr = &G[j][0]
		G_np[j] = np.asarray(<unsigned char[:B]>G_ptr)
	return G_np

### Convert 1-bit into full array and initialize cluster mean
cpdef void convertBit(unsigned char[:,::1] Gt, unsigned char[:,::1] Xt, float[:,::1] C, \
		int w0, int t) nogil:
	cdef:
		int B = Gt.shape[1]
		int m = Xt.shape[0]
		int n = Xt.shape[1]
		int b, i, j, bit
		unsigned char mask = 1
		unsigned char byte
	for j in prange(m, num_threads=t):
		i = 0
		C[0,j] = 0.0
		for b in range(B):
			byte = Gt[w0+j,b]
			for bit in range(8):
				Xt[j,i] = byte & mask
				C[0,j] = C[0,j] + Xt[j,i]
				byte = byte >> 1 # Right shift 1 bit
				i = i + 1
				if i == n:
					break

### Convert 1-bit into full array for predicting target clusters
cpdef void predictBit(unsigned char[:,::1] Gt, unsigned char[:,::1] Xt, int w0, \
		int t) nogil:
	cdef:
		int B = Gt.shape[1]
		int m = Xt.shape[0]
		int n = Xt.shape[1]
		int b, i, j, bit
		unsigned char mask = 1
		unsigned char byte
	for j in prange(m, num_threads=t):
		i = 0
		for b in range(B):
			byte = Gt[w0+j,b]
			for bit in range(8):
				Xt[j,i] = byte & mask
				byte = byte >> 1 # Right shift 1 bit
				i = i + 1
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
					i = i + 1
					if i == n:
						break
			
			# Save window and cluster information
			P_mat[j,0] = w + 1
			P_mat[j,1] = k + 1
			j = j + 1

### Convert 1-bit into genotype array for phenotypes
cpdef void genotypeBit(unsigned char[:,::1] G_mat, double[:,::1] G, long[::1] p) nogil:
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
				i = i + 1
				if i == n:
					break

### Convert haplotype cluster alleles to standardized array for phenotypes
cpdef void convertHaplo(unsigned char[:,::1] Z, double[:,::1] G, \
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
				j = j + 1
			b = b + 1

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
			c = c + 1
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
		j = j + 1
