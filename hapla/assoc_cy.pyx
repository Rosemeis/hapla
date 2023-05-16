# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
import numpy as np
cimport numpy as np
from cpython.mem cimport PyMem_Malloc, PyMem_Free
from cython.parallel import prange
from libc.math cimport sqrt

##### hapla - association testing #####
### hapla regress
# Setup haplotype clusters for a given window
cpdef void haplotypeExtract(unsigned char[:,::1] Z, double[:,::1] Z_tilde, \
		long[::1] B_arr, unsigned char[::1] K_vec):
	cdef int n = Z.shape[1]
	cdef int W = B_arr.shape[0]
	cdef int i, k, w
	cdef int b = 0
	for w in range(W):
		for k in range(K_vec[B_arr[w]]):
			for i in range(n):
				if Z[B_arr[w],i] == k:
					Z_tilde[b,i//2] += 1.0
			b += 1

# Setup haplotype clusters for a given window and estimate frequencies
cpdef void haplotypeAssoc(unsigned char[:,::1] Z, double[:,::1] Z_tilde, \
		double[:,::1] P, long[::1] B_arr, unsigned char[::1] K_vec, int B_idx):
	cdef int n = Z.shape[1]
	cdef int W = B_arr.shape[0]
	cdef int i, k, w
	cdef int b = 0
	for w in range(W):
		for k in range(K_vec[B_arr[w]]):
			for i in range(n):
				if Z[B_arr[w],i] == k:
					Z_tilde[b,i//2] += 1.0
					P[B_idx+b,3] += 1.0
			P[B_idx+b,3] /= <double>n
			b += 1

# Setup LOCO prediction if multiple chromosomes are present
cpdef void haplotypeLoco(float[:,::1] B_mat, float[:,::1] E_hat, double[:,::1] y_chr, \
		double[::1] y, double[::1] y_hat, unsigned char[::1] N_ind, long[::1] B_list, \
		int R):
	cdef int C = y_chr.shape[0]
	cdef int n = y_chr.shape[1]
	cdef int b, c, i, r
	cdef int B_blk = 0
	for c in range(C):
		for i in range(n):
			y_chr[c,i] = y[i] - y_hat[i]
			for b in range(B_blk*R, B_blk*R + B_list[c]*R):
				y_chr[c,i] += B_mat[b,i]*E_hat[N_ind[i],b]
		B_blk += B_list[c]

# Remove residualized effect of haplotype clusters in same block
cpdef void residualY(float[:,::1] B_mat, float[:,::1] E_hat, double[::1] y, \
		double[::1] y_hat, double[::1] y_res, unsigned char[::1] N_ind, \
		int b, int R):
	cdef int n = y.shape[0]
	cdef int i, r
	for i in range(n):
		y_res[i] = y[i] - y_hat[i]
		for r in range(R):
			y_res[i] += B_mat[b*R+r,i]*E_hat[N_ind[i],b*R+r]

# Association testing of haplotype clusters
cpdef void haplotypeTest(double[:,::1] Z_tilde, double[:,::1] P, double[::1] y_res, \
		long[::1] B_arr, unsigned char[::1] K_vec, double s_env, int B):
	cdef int n = Z_tilde.shape[1]
	cdef int W = B_arr.shape[0]
	cdef int i, k, w
	cdef int b = 0
	cdef double gTg, gTy
	for w in range(W):
		for k in range(K_vec[B_arr[w]]):
			gTg = 0.0
			gTy = 0.0
			for i in range(n):
				gTg = gTg + Z_tilde[b,i]*Z_tilde[b,i]
				gTy = gTy + Z_tilde[b,i]*y_res[i]
			P[B+b,1] = B_arr[w]+1 # Window
			P[B+b,2] = k+1 # Cluster
			P[B+b,4] = gTy/gTg # Beta
			P[B+b,6] = gTy/(s_env*sqrt(gTg)) # Wald's
			P[B+b,5] = P[B+b,4]/P[B+b,6] # SE(Beta)
			P[B+b,6] *= P[B+b,6]
			b += 1


### hapla prs
# Expand the haplotype cluster matrix
cpdef void updateZ(unsigned char[:,::1] Z, unsigned char[:,::1] Z_tilde, \
		unsigned char[::1] K_vec):
	cdef int W = Z.shape[0]
	cdef int n = Z.shape[1]
	cdef int i, k, w
	cdef int j = 0
	for w in range(W):
		for k in range(K_vec[w]):
			for i in range(n):
				if Z[w,i] == k:
					Z_tilde[j,i//2] += 1
			j += 1

# Clumping adjacent windows based on most significant haplotype cluster
cpdef void clumpWindows(unsigned char[::1] mask, unsigned char[::1] maskW,
		unsigned char[::1] K_vec, double[::1] p, int B):
	cdef int W = maskW.shape[0]
	cdef int b, i, k, w
	cdef int j = 0
	cdef double p_val
	cdef double* p_min
	p_min = <double*>PyMem_Malloc(sizeof(double)*W)
	# Find minimum per window
	for w in range(W):
		p_val = 1.0
		p_min[w] = 1.0
		for k in range(K_vec[w]):
			if p[j] < p_val:
				p_min[w] = p[j]
			j += 1
	
	# Clump B windows in running window
	for w in range(W-1):
		if maskW[w] == 0:
			for b in range(w+1, min(w+B, W)):
				if p_min[w] < p_min[b]:
					maskW[b] = 1
				else:
					maskW[w] = 1
					break
	
	# Mask haplotype clusters in masked windows
	j = 0
	for w in range(W):
		if maskW[w] == 1:
			for k in range(K_vec[w]):
				mask[j] = 1
				j += 1
		else:
			j += K_vec[w]
	
	# Deallocate
	PyMem_Free(p_min)
