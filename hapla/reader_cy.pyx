# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
import numpy as np
cimport numpy as np
from cython.parallel import prange
from libcpp.vector cimport vector
from libc.stdlib cimport malloc, free

DTYPE = np.uint8
ctypedef np.uint8_t DTYPE_t
DTYPE2 = np.int16
ctypedef np.int16_t DTYPE2_t
ctypedef vector[unsigned char] char_vec

##### Cython function for reading VCF/BCF files #####
### Read VCF/BCF into 1-bit integer matrix
cpdef np.ndarray[DTYPE_t, ndim=2] readVCF(v_file, const int n, const int B):
	cdef int b, i, j, m, bit
	cdef np.ndarray[DTYPE2_t, ndim=2] geno
	cdef char_vec G_var
	cdef vector[char_vec] G
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
	cdef np.ndarray[DTYPE_t, ndim=2] G_np = np.empty((m, B), dtype=DTYPE)
	cdef unsigned char *G_ptr
	for j in range(m):
		G_ptr = &G[j][0]
		G_np[j] = np.asarray(<unsigned char[:B]>G_ptr)
	return G_np

### Convert 1-bit into full array and initialize cluster mean
cpdef void convertBit(unsigned char[:,::1] Gt, unsigned char[:,::1] Xt, float[:,::1] C, \
		int w0, int t):
	cdef int B = Gt.shape[1]
	cdef int m = Xt.shape[0]
	cdef int n = Xt.shape[1]
	cdef int b, i, j, bit
	cdef unsigned char mask = 1
	cdef unsigned char byte
	with nogil:
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
cpdef void predictBit(unsigned char[:,::1] Gt, unsigned char[:,::1] Xt, int w0, int t):
	cdef int B = Gt.shape[1]
	cdef int m = Xt.shape[0]
	cdef int n = Xt.shape[1]
	cdef int b, i, j, bit
	cdef unsigned char mask = 1
	cdef unsigned char byte
	with nogil:
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

### Convert 1-bit into genotype array for generating phenotypes
cpdef void genotypeBit(unsigned char[:,::1] Gt, unsigned char[:,::1] G, long[::1] p):
	cdef int m = G.shape[0]
	cdef int n = G.shape[1]
	cdef int B = Gt.shape[1]
	cdef int b, i, j, bit
	cdef unsigned char mask = 1
	cdef unsigned char byte
	for j in range(m):
		i = 0
		for b in range(B):
			byte = Gt[p[j],b]
			for bit in range(0, 8, 2):
				G[j,i] = byte & mask
				byte = byte >> 1 # Right shift 1 bit
				G[j,i] += byte & mask
				byte = byte >> 1 # Right shift 1 bit
				i = i + 1
				if i == n:
					break

### Convert haplotype cluster assignments to proper array for generating phenotypes
cpdef void convertHaplo(unsigned char[:,::1] Z, unsigned char[:,::1] G, \
		unsigned char[::1] K_vec, long[::1] C):
	cdef int W = Z.shape[0]
	cdef int n = Z.shape[1]
	cdef int i, k, w
	cdef int b = 0
	cdef int j = 0
	for w in range(W):
		for k in range(K_vec[w]):
			if b == C[j]:
				for i in range(n):
					if Z[w,i] == k:
						G[j,i//2] += 1
				j += 1
			b += 1

# Estimate haplotype cluster frequencies
cpdef void estimateFreqs(unsigned char[:,::1] Z, unsigned char[::1] K_vec, \
		double[::1] pi):
	cdef int W = Z.shape[0]
	cdef int n = Z.shape[1]
	cdef int i, k, w
	cdef int j = 0
	for w in range(W):
		for k in range(K_vec[w]):
			for i in range(n):
				if Z[w,i] == k:
					pi[j] += 1.0
			pi[j] /= <double>n
			j += 1

### Filter out variants from haplotype clustering and fix window sizes
cpdef void filterSNPs(unsigned char[:,::1] Gt, long[::1] W, unsigned char[::1] mask):
	cdef int m = Gt.shape[0]
	cdef int B = Gt.shape[1]
	cdef int s = W.shape[0]
	cdef int c = 0
	cdef int b, j, k
	cdef int* count
	count = <int*>malloc(sizeof(int)*s)
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
	free(count)
