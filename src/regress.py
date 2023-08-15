"""
hapla.
Perform whole-genome regression using haplotype cluster alleles.
"""

__author__ = "Jonas Meisner"

##### hapla regress #####
def main(args):
	print("hapla by Jonas Meisner (v0.2)")
	print(f"hapla regress using {args.threads} thread(s).")

	# Check input
	assert args.filelist is not None, "No input data provided!"
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
	from src import functions
	from src import asso_cy

	### Load data
	# Load haplotype cluster alleles (and concatentate across windows)
	Z_list = []
	K_list = []
	with open(args.filelist) as f:
		N_chr = 0
		K_tot = 0
		for c_idx in f:
			Z_list.append(np.load(c_idx.strip("\n")))
			K_list.append(np.max(Z_list[-1], axis=1)) # Dummy encoding
			print(f"\rParsed file #{N_chr+1}", end="")
			N_chr += 1
			K_tot += np.sum(K_list[-1], dtype=int)
	n = Z_list[0].shape[1]//2
	print(f"\rLoaded {K_tot} haplotype cluster alleles of " + \
		f"{n} individuals in {len(Z_list)} chromosomes.")

	# Load phenotype file (outcome)
	y = np.loadtxt(args.pheno, dtype=float)
	assert y.shape[0] == n, "Number of samples differ between files!"
	print("Loaded phenotype file.")

	# Load covariates and add intercept
	if args.covar is not None:
		C = np.loadtxt(args.covar, dtype=float)
		assert C.shape[0] == n, "Number of samples differ between files!"
		C = np.concatenate((np.ones((n, 1), dtype=float), C), axis=1)
		print("Loaded covariates file.")
	else:
		C = np.ones((n, 1), dtype=float)

	# Load eigenvectors
	if args.eigen is not None:
		E = np.loadtxt(args.eigen, dtype=float)
		assert E.shape[0] == n, "Number of samples differ between files!"
		print("Loaded eigenvectors file.")
		C = np.concatenate((C, E), axis=1)
		del E
	assert C.shape[1] < n, "Number of covariates exceed indviduals!"

	# Setup parameters and containers for estimation
	if args.folds > 0: # K-fold splits
		np.random.seed(args.seed) # Set random seed
		N_split = np.array_split(np.random.permutation(n), args.folds)
	h2 = np.clip(np.linspace(0.0, 1.0, args.ridge), 0.01, 0.99) # h^2_g

	### Residualize and scale phenotypes by covariates
	U_cov, _, _ = functions.fastSVD(C)
	R_cov = U_cov.shape[1]
	del C
	y_res = y - np.dot(U_cov, np.dot(U_cov.T, y))
	y_res /= (np.linalg.norm(y_res)/sqrt(n - R_cov))


	##### Step 1 - Whole-genome regression #####
	y_mse = np.zeros(args.ridge, dtype=float) # Cross validation phenotype MSE
	y_prs = np.zeros((args.ridge, n), dtype=float) # Ridge predictors

	### Level 0 - Per-chromosome ridge regression
	L = np.zeros((N_chr, n), dtype=float) # Chromosome predictors
	lmbda = K_tot*(1.0 - h2)/h2 # Lambda scaling

	# Loop through chromosomes
	for c_idx in np.arange(N_chr):
		print(f"\rLevel 0 - Chromosome {c_idx+1}/{N_chr}", end="")
		K_vec = np.max(Z_list[c_idx], axis=1) # Dummy encoding
		K_num = np.sum(K_list[c_idx], dtype=int) # Number of haplotype clusters

		# Standardize haplotype clusters, residualize and scale by covariates
		Z = np.zeros((K_num, n), dtype=float)
		asso_cy.haplotypeCenter(Z_list[c_idx], Z, K_list[c_idx])
		Z -= np.dot(np.dot(Z, U_cov), U_cov.T)
		Z /= (np.linalg.norm(Z, axis=1, keepdims=True)/sqrt(n - R_cov))
		Z = np.ascontiguousarray(Z.T)

		# Cross-validation scheme
		if args.folds > 0: # K-fold CV
			for k in np.arange(args.folds):
				# Define folds
				N_test = np.sort(N_split[k])
				N_train = np.setdiff1d(np.arange(n), N_test)

				# Regression
				U, S, V = functions.fastSVD(Z[N_train,:])
				UtY = np.dot(U.T, y_res[N_train])
				for r in np.arange(args.ridge):
					y_prs[r, N_test] = np.dot(Z[N_test,:], \
						np.dot(V*(S/(S*S + lmbda[r])), UtY))
					y_mse[r] += (np.linalg.norm(y_res[N_test] - y_prs[r, N_test]))**2
				del N_test, N_train, U, S, V, UtY
		else: # N-fold CV (LOOCV)
			U, S, V = functions.fastSVD(Z)
			UtY = np.dot(U.T, y_res)
			x = np.zeros(K_num, dtype=float)
			for r in np.arange(args.ridge):
				H = np.dot(V*(1.0/(S*S + lmbda[r])), V.T)
				p = np.dot(U*((S*S)/(S*S + lmbda[r])), UtY)
				asso_cy.loocv(Z, y_prs, y_mse, H, p, y_res, x, r)
			del U, S, V, UtY, H, p, x
		del Z
		L[c_idx,:] = y_prs[np.argmin(y_mse),:]
		y_prs.fill(0.0)
		y_mse.fill(0.0)
	del Z_list, K_list
	print("")

	### Level 1 - Combined ridge regression
	L = np.ascontiguousarray(L.T)
	lmbda = N_chr*(1.0 - h2)/h2 # Lambda scaling

	# Cross-validation scheme
	if args.folds > 0: # K-fold CV
		E_mat = np.zeros((args.folds, args.ridge, N_chr), dtype=float)
		N_ind = np.zeros(n, dtype=np.uint8) # Index vector for K-fold info
		for k in np.arange(args.folds):
			print(f"\rLevel 1 - Fold {k+1}/{args.folds}", end="")
			N_test = np.sort(N_split[k])
			N_train = np.setdiff1d(np.arange(n), N_test)

			# Regression
			U, S, V = functions.fastSVD(L[N_train,:])
			UtY = np.dot(U.T, y_res[N_train])
			for r in np.arange(args.ridge):
				E_mat[k,r,:] = np.dot(V*(S/(S*S + lmbda[r])), UtY)
				y_prs[r, N_test] = np.dot(L[N_test,:], E_mat[k,r,:])
				y_mse[r] += (np.linalg.norm(y_res[N_test] - y_prs[r, N_test]))**2
			N_ind[N_test] = k
			
			# Free memory
			del N_test, N_train, U, S, V, UtY
		print("")
	else: # N-fold CV (LOOCV)
		print("Level 1 - LOOCV")
		U, S, V = functions.fastSVD(L)
		UtY = np.dot(U.T, y_res)
		x = np.zeros(L.shape[1], dtype=float)
		for r in np.arange(args.ridge):
			H = np.dot(V*(1.0/(S*S + lmbda[r])), V.T)
			p = np.dot(U*(S*S/(S*S + lmbda[r])), UtY)
			asso_cy.loocv(L, y_prs, y_mse, H, p, y_res, x, r)
		
		# Free memory
		del H, p
	
	# Find optimal hyperparameter from CV
	y_opt = np.argmin(y_mse)
	y_hat = np.copy(y_prs[y_opt,:])
	if args.folds > 0:
		E_hat = np.copy(E_mat[:,y_opt,:])
		del E_mat
	del y_mse, y_prs

	# Optional save of whole-genome prediction
	np.savetxt(f"{args.out}.pred", y_hat, fmt="%.7f")
	print(f"Saved whole-genome prediction as {args.out}.pred")

	### Create LOCO predictions
	print("Obtaining LOCO predictions.")
	y_chr = np.zeros((n, N_chr), dtype=float)
	if args.folds > 0: # LOCO predictions from K-fold CV
		asso_cy.haplotypeLOCO(L, E_hat, y_chr, y_hat, N_ind)
	else: # LOCO predictions from LOOCV
		H = np.dot(V*(1.0/(S*S + lmbda[y_opt])), V.T)
		p = np.dot(U*(S*S/(S*S + lmbda[y_opt])), UtY)
		a = np.dot(V*(S/(S*S + lmbda[y_opt])), UtY)
		asso_cy.loocvLOCO(L, y_chr, y_hat, H, p, y_res, a, x)
		del U, S, V, UtY, H, p, a, x
	np.savetxt(f"{args.out}.loco", y_chr, fmt="%.7f")
	print(f"Saved {N_chr} LOCO predictions as {args.out}.loco")



##### Main exception #####
assert __name__ != "__main__", "Please use the 'hapla regress' command!"
