"""
hapla.
Perform association tests using haplotype cluster assignments.
Using whole-genome regression approach from REGENIE.
"""

__author__ = "Jonas Meisner"

##### hapla regress #####
def main(args):
	print("hapla regress by Jonas Meisner (v0.1)")
	print(f"Using {args.threads} thread(s).")

	# Check input
	assert (args.filelist is not None) or (args.clusters is not None), \
		"No input data (--filelist or --clusters)"
	assert args.pheno is not None, "No phenotype file provided!"
	assert args.ridge >= 3, "Need at least 3 ridge regressors!"
	if args.eigen is None:
		print("WARNING: Eigenvectors (PCs) have not been provided!")
	if args.covar is None:
		print("WARNING: Covariates have not been provided!")

	# Control threads of external numerical libraries
	import os
	os.environ["MKL_NUM_THREADS"] = str(args.threads)
	os.environ["OMP_NUM_THREADS"] = str(args.threads)
	os.environ["NUMEXPR_NUM_THREADS"] = str(args.threads)
	os.environ["OPENBLAS_NUM_THREADS"] = str(args.threads)

	# Import numerical libraries and cython functions
	import numpy as np
	from math import sqrt
	from scipy.stats import chi2
	from src import functions
	from src import assoc_cy

	### Load data
	# Load haplotype cluster assignments (and concatentate across windows)
	if args.filelist is not None:
		Z_list = []
		C_list = []
		with open(args.filelist) as f:
			N_chr = 0
			for chr in f:
				Z_list.append(np.load(chr.strip("\n")))
				C_list.append(Z_list[-1].shape[0])
				print(f"\rParsed file #{N_chr+1}", end="")
				N_chr += 1
		C_list = np.array(C_list, dtype=int)
		B_list = np.zeros(N_chr, dtype=int)
		Z_mat = np.concatenate(Z_list, axis=0)
		del Z_list
	else:
		Z_mat = np.load(args.clusters)
	print("\rLoaded haplotype cluster assignments of " + \
		f"{Z_mat.shape[1]} haplotypes in {Z_mat.shape[0]} windows.")
	W = Z_mat.shape[0]
	n = Z_mat.shape[1]//2

	# Load phenotype file (outcome)
	y = np.loadtxt(args.pheno, dtype=float)
	assert y.shape[0] == n, "Number of samples differ between files!"
	print("Loaded phenotype file.")

	# Load covariates and add bias term
	if args.covar is not None:
		C = np.loadtxt(args.covar, dtype=np.float32)
		assert C.shape[0] == n, "Number of samples differ between files!"
		C = np.concatenate((np.ones((n, 1), dtype=np.float32), C), axis=1)
		print("Loaded covariates file.")
	else:
		C = np.ones((n, 1), dtype=np.float32)

	# Load eigenvectors
	if args.eigen is not None:
		E = np.loadtxt(args.eigen, dtype=np.float32)
		assert E.shape[0] == n, "Number of samples differ between files!"
		print("Loaded eigenvectors file.")
		C = np.concatenate((C, E), axis=1)
		del E

	# Setup parameters and containers for estimation
	np.random.seed(args.seed) # Set random seed
	h2 = np.clip(np.linspace(0.0, 1.0, args.ridge), 0.01, 0.99) # Heritability
	K_vec = np.max(Z_mat, axis=1) + 1 # Number of haplotype clusters in windows
	N_split = np.array_split(np.random.permutation(n), args.folds) # K-fold splits

	### Residualize and scale phenotypes by covariates
	U_c, _, _ = functions.truncatedSVD(C)
	R_c = U_c.shape[1]
	y -= np.dot(U_c, np.dot(U_c.T, y))
	y /= np.linalg.norm(y)/sqrt(n - R_c)

	### Level 0 - Ridge regression
	# Setup haplotype cluster window blocks
	if args.block > 1:
		if args.filelist is not None: # Setup blocks per chromosome
			B_arr = []
			C_sum = 0
			for c in range(N_chr):
				B_chr = np.split(np.arange(args.block*(C_list[c]//args.block)) + \
					C_sum, C_list[c]//args.block)
				B_chr[-1] = np.concatenate((B_chr[-1], np.arange(\
					args.block*(C_list[c]//args.block), C_list[c]) + C_sum))
				B_arr += B_chr
				C_sum += C_list[c]
				B_list[c] = len(B_chr)
			del B_chr
		else: # Only one chromosome
			B_arr = np.split(np.arange(args.block*(W//args.block)), W//args.block)
			B_arr[-1] = np.concatenate((B_arr[-1], np.arange(\
				args.block*(W//args.block), W)))
	else: # One window per block
		B_arr = np.split(np.arange(W), W)
	B = len(B_arr) # Number of window blocks
	L_mat = np.zeros((B*args.ridge, n), dtype=np.float32) # Local predictors

	# Ridge regression in blocks (local predictors)
	lmbda = W*(1.0 - h2)/h2 # Lambda scaling in ridge regression 1
	for b in np.arange(B):
		print(f"\rLevel 0 - Block {b+1}/{B}", end="")
		B_num = np.sum(K_vec[B_arr[b]], dtype=int)

		# Extract haplotype clusters, residualize and scale by covariates
		Z_tilde = np.zeros((B_num, n), dtype=float)
		assoc_cy.haplotypeExtract(Z_mat, Z_tilde, B_arr[b], K_vec)
		Z_tilde -= np.dot(np.dot(Z_tilde, U_c), U_c.T)
		Z_tilde /= np.linalg.norm(Z_tilde, axis=1, keepdims=True)/sqrt(n - R_c)

		# K-fold cross-validation scheme
		for k in np.arange(args.folds):
			# Define folds
			N_test = np.sort(N_split[k])
			N_train = np.setdiff1d(np.arange(n), N_test)

			# Ridge regressors
			U, S, V = functions.truncatedSVD(Z_tilde[:, N_train], transpose=True)
			UtY = np.dot(U.T, y[N_train])
			for r in np.arange(args.ridge):
				L_mat[b*args.ridge + r, N_test] = np.dot(Z_tilde[:, N_test].T, \
					np.dot(V*(S/(S*S + lmbda[r])), UtY))
			
			# Free memory
			del U, S, V, UtY
		del Z_tilde
	print("")
	
	# Center and scale local predictors
	L_mat -= np.mean(L_mat, axis=1, keepdims=True)
	L_mat /= (np.linalg.norm(L_mat, axis=1, keepdims=True)/sqrt(n - 1))

	### Level 1 - Ridge regression - K-fold validation scheme
	E_mat = np.zeros((args.folds, args.ridge, L_mat.shape[0]), dtype=np.float32)
	y_mse = np.zeros(args.ridge, dtype=float) # Cross validation phenotype MSE
	y_prs = np.zeros((args.ridge, n), dtype=float) # Phenotype prediction
	N_ind = np.zeros(n, dtype=np.uint8) # Index vector for K-fold information
	lmbda = L_mat.shape[0]*(1.0 - h2)/h2 # Lambda scaling in ridge regression 2
	for k in np.arange(args.folds):
		print(f"\rLevel 1 - Fold {k+1}/{args.folds}", end="")
		N_test = np.sort(N_split[k])
		N_train = np.setdiff1d(np.arange(n), N_test)

		# Ridge regressors
		U, S, V = functions.truncatedSVD(L_mat[:, N_train], transpose=True)
		UtY = np.dot(U.T, y[N_train])
		for r in np.arange(args.ridge):
			E_mat[k,r,:] = np.dot(V*(S/(S*S + lmbda[r])), UtY)
			y_prs[r, N_test] = np.dot(L_mat[:, N_test].T, E_mat[k,r,:])
			y_mse[r] += (np.linalg.norm(y[N_test] - y_prs[r, N_test]))**2
		N_ind[N_test] = k
		
		# Free memory
		del U, S, V, UtY
	y_mse /= float(n)
	h2_opt = np.argmin(y_mse)
	print(f"\nOptimal h2 from CV: {h2[h2_opt]}, MSE={y_mse[h2_opt]}")
	y_hat = np.copy(y_prs[h2_opt,:])
	E_hat = np.copy(E_mat[:,h2_opt,:])
	del y_mse, y_prs, E_mat

	# Optional save of whole-genome prediction
	if args.save_pred:
		r2 = 1.0 - np.sum((y_hat - y)**2)/np.sum((y - np.mean(y))**2)
		np.savetxt(f"{args.out}.pred", y_hat, fmt="%.7f", header=f"R2={round(r2, 7)}")
		print(f"Saved whole-genome prediction as {args.out}.pred")

	### Association testing
	P = np.zeros((np.sum(K_vec), 8), dtype=float) # Output matrix

	# Create LOCO predictions if multiple chromosomes provided
	if args.filelist is not None:
		y_chr = np.zeros((N_chr, n), dtype=float)
		assoc_cy.haplotypeLOCO(L_mat, E_hat, y_chr, y, y_hat, N_ind, B_list, args.ridge)
		if args.save_loco:
			np.savetxt(f"{args.out}.loco", y_chr.T, fmt="%.7f")
			print(f"Saved {N_chr} LOCO predictions as {args.out}.loco")
	else: # Residualized phenotype if one chromosome provided
		y_res = np.zeros(n, dtype=float)

	# Perform testing
	B_idx = 0 # Start index for haplotype clusters
	for b in np.arange(B):
		print(f"\rAssociation testing - Block {b+1}/{B}", end="")
		B_num = np.sum(K_vec[B_arr[b]], dtype=int)
		
		# Extract haplotype clusters, residualize and scale by covariates
		Z_tilde = np.zeros((B_num, n), dtype=float)
		assoc_cy.haplotypeAssoc(Z_mat, Z_tilde, P, B_arr[b], K_vec, B_idx)
		Z_tilde -= np.dot(np.dot(Z_tilde, U_c), U_c.T)
		Z_scale = np.linalg.norm(Z_tilde, axis=1)/sqrt(n - R_c)
		Z_tilde /= Z_scale.reshape(-1,1)

		# Create residualized phenotypes
		if args.filelist is not None: # Extract LOCO prediction for block
			if b == 0: # First chromosome
				C_idx = 0
				W_idx = 0
				B_nxt = B_list[C_idx]
				y_res = y_chr[0,:]
				s_env = np.linalg.norm(y_res)/sqrt(n - R_c)
			if b == B_nxt: # Next chromosome
				C_idx += 1
				W_idx = 0
				B_nxt += B_list[C_idx]
				y_res = y_chr[C_idx,:]
				s_env = np.linalg.norm(y_res)/sqrt(n - R_c)
			P[B_idx:(B_idx+B_num),0] = C_idx + 1 # Chromosome info
		else: # Residualized phenotype without block effect
			assoc_cy.residualY(L_mat, E_hat, y, y_hat, y_res, N_ind, b, args.ridge)
			s_env = np.linalg.norm(y_res)/sqrt(n - R_c)

		# Test haplotype clusters
		assoc_cy.haplotypeTest(Z_tilde, P, y_res, B_arr[b], K_vec, s_env, W_idx, B_idx)
		P[B_idx:(B_idx+B_num),4] *= Z_scale # Rescale beta
		P[B_idx:(B_idx+B_num),5] *= Z_scale # Rescale se(beta)
		B_idx += B_num
		W_idx += B_arr[b].shape[0]

		# Free memory
		del Z_tilde, Z_scale
	print("")

	### Save association results
	P[:,7] = chi2.sf(P[:,6], df=1) # P-values (1 - cdf) - Wald's
	np.savetxt(f"{args.out}.assoc", P, fmt=["%i", "%i", "%i", "%.7f", "%.7f", \
		"%.7f", "%.7f", "%.7e"], header="chrom window cluster freq beta se chisq p",
		comments="")
	print(f"Saved association test statistics as {args.out}.assoc")



##### Main exception #####
assert __name__ != "__main__", "Please use the 'hapla regress' command!"
