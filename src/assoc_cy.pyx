# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
import numpy as np
cimport numpy as np
from scipy.linalg.cython_blas cimport dgemv, ddot
from libc.math cimport sqrt

##### hapla - association testing #####
### hapla regress
# Setup standardized haplotype clusters for a block
cpdef void haplotypeStandard(unsigned char[:,::1] Z_mat, double[:,::1] Z, \
		long[::1] B_arr, unsigned char[::1] K_vec):
	cdef:
		int n = Z.shape[1]
		int W = B_arr.shape[0]
		int b = 0
		int i, k, w
		double pi, sd
	for w in range(W):
		for k in range(K_vec[B_arr[w]]):
			pi = 0.0
			sd = 0.0
			for i in range(2*n):
				if Z_mat[B_arr[w],i] == k:
					Z[b,i//2] += 1.0
					pi += 1.0
			pi /= <double>n
			for i in range(n):
				sd += (Z[b,i]-pi)*(Z[b,i]-pi)
			sd = sqrt(sd/(<double>(n-1)))
			for i in range(n):
				Z[b,i] = (Z[b,i] - pi)/sd
			b += 1

# Setup haplotype clusters for a block and estimate frequencies
cpdef void haplotypeAssoc(unsigned char[:,::1] Z_mat, double[:,::1] Z, \
		double[:,::1] P, long[::1] B_arr, unsigned char[::1] K_vec, int B_idx):
	cdef:
		int n = Z.shape[1]
		int W = B_arr.shape[0]
		int b = 0
		int i, k, w
	for w in range(W):
		for k in range(K_vec[B_arr[w]]):
			for i in range(n):
				if Z_mat[B_arr[w],i] == k:
					Z[b,i//2] += 1.0
					P[B_idx+b,3] += 1.0
			P[B_idx+b,3] /= <double>n
			b += 1

# Fast Level 0 LOOCV using SciPy BLAS routines
cpdef void loocvLevel0(double[:,::1] Z, double[:,::1] L, double[:,::1] H, \
		double[::1] p, double[::1] y, double[::1] x, int B) nogil:
	cdef:
		char *trans = "T"
		int n = Z.shape[0]
		int b = Z.shape[1]
		int i1 = 1
		int i2 = 1
		int i
		double alpha = 1.0
		double beta = 0.0
		double *H0 = &H[0,0]
		double *x0 = &x[0]
		double *Z0
		double h
	for i in range(n):
		Z0 = &Z[i,0]
		dgemv(trans, &b, &b, &alpha, H0, &b, Z0, &i1, &beta, x0, &i2)
		h = ddot(&b, Z0, &i1, x0, &i2)
		L[B,i] = p[i] - h*(y[i] - p[i])/(1.0 - h)

# Fast Level 1 LOOCV using SciPy BLAS routines
cpdef void loocvLevel1(double[:,::1] L, double[:,::1] y_prs, double[::1] y_mse, \
		double[:,::1] H, double[::1] p, double[::1] y, double[::1] x, int r) nogil:
	cdef:
		char *trans = "T"
		int n = L.shape[0]
		int b = L.shape[1]
		int i1 = 1
		int i2 = 1
		int i
		double alpha = 1.0
		double beta = 0.0
		double *H0 = &H[0,0]
		double *x0 = &x[0]
		double *L0
		double h
	for i in range(n):
		L0 = &L[i,0]
		dgemv(trans, &b, &b, &alpha, H0, &b, L0, &i1, &beta, x0, &i2)
		h = ddot(&b, L0, &i1, x0, &i2)
		y_prs[r,i] = p[i] - h*(y[i] - p[i])/(1.0 - h)
		y_mse[r] += (y[i] - y_prs[r,i])*(y[i] - y_prs[r,i])

# LOCO prediction for LOOCV using SciPy BLAS routines
cpdef void loocvLOCO(double[:,::1] L, double[:,::1] y_chr, double[::1] y_hat, \
		double[:,::1] H, double[::1] p, double[::1] y, double[::1] a, double[::1] x, \
		long[::1] B_list, int R) nogil:
	cdef:
		char *trans = "T"
		int C = y_chr.shape[1]
		int n = L.shape[0]
		int b = L.shape[1]
		int i1 = 1
		int i2 = 1
		int B_blk, blk, c, i
		double alpha = 1.0
		double beta = 0.0
		double *H0 = &H[0,0]
		double *x0 = &x[0]
		double *L0
		double e, h
	for i in range(n):
		L0 = &L[i,0]
		dgemv(trans, &b, &b, &alpha, H0, &b, L0, &i1, &beta, x0, &i2)
		h = ddot(&b, L0, &i1, x0, &i2)
		e = y[i] - p[i]
		B_blk = 0
		for c in range(C):
			y_chr[i,c] = y_hat[i]
			for blk in range(B_blk*R, B_blk*R + B_list[c]*R):
				y_chr[i,c] -= L[i,blk]*(a[blk] - x[blk]*e/(1.0 - h))
			B_blk += B_list[c]

# LOCO prediction for K-fold CV
cpdef void haplotypeLOCO(double[:,::1] L, double[:,::1] E_hat, double[:,::1] y_chr, \
		double[::1] y_hat, unsigned char[::1] N_ind, long[::1] B_list, int R):
	cdef:
		int n = y_chr.shape[0]
		int C = y_chr.shape[1]
		int B_blk, blk, c, i
	for i in range(n):
		B_blk = 0
		for c in range(C):
			y_chr[i,c] = y_hat[i]
			for blk in range(B_blk*R, B_blk*R + B_list[c]*R):
				y_chr[i,c] -= L[i,blk]*E_hat[N_ind[i],blk]
			B_blk += B_list[c]

# Association testing of haplotype clusters
cpdef void haplotypeTest(double[:,::1] Z, double[:,::1] P, double[::1] y_res, \
		long[::1] B_arr, unsigned char[::1] K_vec, double s_env, int W_chr, int B):
	cdef:
		int n = Z.shape[1]
		int W = B_arr.shape[0]
		int b = 0
		int i, k, w
		double gTg, gTy
	for w in range(W):
		for k in range(K_vec[B_arr[w]]):
			gTg = 0.0
			gTy = 0.0
			for i in range(n):
				gTg += Z[b,i]*Z[b,i]
				gTy += Z[b,i]*y_res[i]
			P[B+b,1] = W_chr+w+1 # Window
			P[B+b,2] = k+1 # Cluster
			P[B+b,4] = gTy/gTg # Beta
			P[B+b,6] = gTy/(s_env*sqrt(gTg)) # Wald's
			P[B+b,5] = P[B+b,4]/P[B+b,6] # SE(Beta)
			P[B+b,6] *= P[B+b,6]
			b += 1


### hapla asso
# Convert 1-bit into genotype block
cpdef void genotypeAssoc(unsigned char[:,::1] G_mat, double[:,::1] G, \
		double[:,::1] P, int B_idx):
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
			byte = G_mat[B_idx+j,b]
			for bit in range(0, 8, 2):
				G[j,i] = byte & mask
				byte = byte >> 1 # Right shift 1 bit
				G[j,i] += byte & mask
				byte = byte >> 1 # Right shift 1 bit
				P[B_idx+j,2] += G[j,i]
				i = i + 1
				if i == n:
					break
		P[B_idx+j,2] /= 2.0*(<double>n)

# Association testing of SNPs
cpdef void genotypeTest(double[:,::1] G, double[:,::1] P, double[::1] y_res, \
		double s_env, int B_idx):
	cdef:
		int m = G.shape[0]
		int n = G.shape[1]
		int i, j
		double gTg, gTy
	for j in range(m):
		gTg = 0.0
		gTy = 0.0
		for i in range(n):
			gTg += G[j,i]*G[j,i]
			gTy += G[j,i]*y_res[i]
		P[B_idx+j,3] = gTy/gTg # Beta
		P[B_idx+j,5] = gTy/(s_env*sqrt(gTg)) # Wald's
		P[B_idx+j,4] = P[B_idx+j,3]/P[B_idx+j,5] # SE(Beta)
		P[B_idx+j,5] *= P[B_idx+j,5]
