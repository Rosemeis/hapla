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
	assert (args.filelist is not None) or (args.clusters is not None), \
		"No input data provided!"
	assert args.pheno is not None, "No phenotype file provided!"
	assert args.block is not None, "Need to provide number of windows in a block!"
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
	from src import assoc_cy

	### Load data
	# Load haplotype cluster alleles (and concatentate across windows)
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
	print("\rLoaded haplotype cluster alleles of " + \
		f"{Z_mat.shape[1]} haplotypes in {Z_mat.shape[0]} windows.")
	W = Z_mat.shape[0]
	n = Z_mat.shape[1]//2

	# Load phenotype file (outcome)
	y = np.loadtxt(args.pheno, dtype=float)
	assert y.shape[0] == n, "Number of samples differ between files!"
	print("Loaded phenotype file.")

	# Load covariates and add bias term
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
	K_vec = np.max(Z_mat, axis=1) + 1 # Number of haplotype clusters in windows

	### Residualize and scale phenotypes by covariates
	U_c, _, _ = functions.fastSVD(C)
	R_c = U_c.shape[1]
	y -= np.dot(U_c, np.dot(U_c.T, y))
	y /= np.linalg.norm(y)/sqrt(n - R_c)


	##### Step 1 - Whole-genome regression #####
	### Level 0 - Ridge regression
	# Setup haplotype cluster window blocks
	if args.block > 1:
		if args.filelist is not None: # Setup blocks per chromosome
			B_arr = []
			C_sum = 0
			for c in range(N_chr):
				B_chr = np.split(np.arange(args.block*(C_list[c]//args.block)) + \
					C_sum, C_list[c]//args.block)
				B_chr[-1] = np.concatenate((B_chr[-1], \
					np.arange(args.block*(C_list[c]//args.block), C_list[c]) + C_sum))
				B_arr += B_chr
				C_sum += C_list[c]
				B_list[c] = len(B_chr)
			del B_chr
		else: # Only one chromosome
			B_arr = np.split(np.arange(args.block*(W//args.block)), W//args.block)
			B_arr[-1] = np.concatenate((B_arr[-1], \
				np.arange(args.block*(W//args.block), W)))
	else: # One window per block
		B_arr = np.split(np.arange(W), W)
	B = len(B_arr) # Number of window blocks

	# Regression in blocks (local predictors)
	if args.linreg:
		r0 = 1
	else:
		r0 = args.ridge
		lmbda = np.sum(K_vec)*(1.0 - h2)/h2 # Lambda scaling (level 0)
	L = np.zeros((B*r0, n), dtype=float) # Local predictors
	for b in np.arange(B):
		print(f"\rLevel 0 - Block {b+1}/{B}", end="")
		B_num = np.sum(K_vec[B_arr[b]], dtype=int) - B_arr[b].shape[0]

		# Extract haplotype clusters, residualize and scale by covariates
		Z = np.zeros((B_num, n), dtype=float)
		assoc_cy.haplotypeStandard(Z_mat, Z, B_arr[b], K_vec)
		Z -= np.dot(np.dot(Z, U_c), U_c.T)
		Z /= np.linalg.norm(Z, axis=1, keepdims=True)/sqrt(n - R_c)
		Z = np.ascontiguousarray(Z.T)
		if args.linreg:
			assert B_num < n, "Number of clusters exceeds individuals!"

		# Cross-validation scheme
		if args.folds > 0: # K-fold CV
			for k in np.arange(args.folds):
				# Define folds
				N_test = np.sort(N_split[k])
				N_train = np.setdiff1d(np.arange(n), N_test)

				# Regression
				U, S, V = functions.fastSVD(Z[N_train,:])
				UtY = np.dot(U.T, y[N_train])
				if args.linreg:
					L[b, N_test] = np.dot(Z[N_test,:], np.dot(V*(1.0/S), UtY))
				else:
					for r in np.arange(args.ridge):
						L[b*args.ridge + r, N_test] = np.dot(Z[N_test,:], \
							np.dot(V*(S/(S*S + lmbda[r])), UtY))
				del N_test, N_train, U, S, V, UtY
		else: # N-fold CV (LOOCV)
			U, S, V = functions.fastSVD(Z)
			UtY = np.dot(U.T, y)
			x = np.zeros(B_num, dtype=float)
			if args.linreg:
				H = np.dot(V*(1.0/(S*S)), V.T)
				p = np.dot(U, UtY)
				assoc_cy.loocvLevel0(Z, L, H, p, y, x, b)
			else:
				for r in np.arange(args.ridge):
					H = np.dot(V*(1.0/(S*S + lmbda[r])), V.T)
					p = np.dot(U*((S*S)/(S*S + lmbda[r])), UtY)
					assoc_cy.loocvLevel0(Z, L, H, p, y, x, b*args.ridge + r)
			del U, S, V, UtY, H, p, x
		del Z
	print("")
	
	# Center and scale local predictors
	L -= np.mean(L, axis=1, keepdims=True)
	L /= np.linalg.norm(L, axis=1, keepdims=True)/sqrt(n - 1)
	L = np.ascontiguousarray(L.T)

	### Level 1 - Ridge regression
	y_mse = np.zeros(args.ridge, dtype=float) # Cross validation phenotype MSE
	y_prs = np.zeros((args.ridge, n), dtype=float) # Phenotype prediction
	lmbda = L.shape[1]*(1.0 - h2)/h2 # Lambda scaling (level 1)

	# Cross-validation scheme
	if args.folds > 0: # K-fold CV
		E_mat = np.zeros((args.folds, args.ridge, L.shape[1]), dtype=float)
		N_ind = np.zeros(n, dtype=np.uint8) # Index vector for K-fold info
		for k in np.arange(args.folds):
			print(f"\rLevel 1 - Fold {k+1}/{args.folds}", end="")
			N_test = np.sort(N_split[k])
			N_train = np.setdiff1d(np.arange(n), N_test)

			# Regression
			U, S, V = functions.fastSVD(L[N_train,:])
			UtY = np.dot(U.T, y[N_train])
			for r in np.arange(args.ridge):
				E_mat[k,r,:] = np.dot(V*(S/(S*S + lmbda[r])), UtY)
				y_prs[r, N_test] = np.dot(L[N_test,:], E_mat[k,r,:])
				y_mse[r] += (np.linalg.norm(y[N_test] - y_prs[r, N_test]))**2
			N_ind[N_test] = k
			
			# Free memory
			del N_test, N_train, U, S, V, UtY
		print("")
	else: # N-fold CV (LOOCV)
		print("Level 1 - LOOCV")
		U, S, V = functions.fastSVD(L)
		UtY = np.dot(U.T, y)
		x = np.zeros(L.shape[1], dtype=float)
		for r in np.arange(args.ridge):
			H = np.dot(V*(1.0/(S*S + lmbda[r])), V.T)
			p = np.dot(U*(S*S/(S*S + lmbda[r])), UtY)
			assoc_cy.loocvLevel1(L, y_prs, y_mse, H, p, y, x, r)
		
		# Free memory
		del H, p
	
	# Find optimal hyperparameter from CV
	h2_opt = np.argmin(y_mse)
	y_hat = np.copy(y_prs[h2_opt,:])
	if args.folds > 0:
		E_hat = np.copy(E_mat[:,h2_opt,:])
		del E_mat
	del y_mse, y_prs

	# Optional save of whole-genome prediction
	r2 = 1.0 - np.sum((y_hat - y)**2)/np.sum((y - np.mean(y))**2)
	np.savetxt(f"{args.out}.pred", y_hat, fmt="%.7f", header=f"R2={round(r2, 7)}")
	print(f"Saved whole-genome prediction as {args.out}.pred")

	### Create LOCO predictions
	# Create LOCO predictions if multiple chromosomes provided
	if args.filelist is not None:
		print("Obtaining LOCO predictions.")
		y_chr = np.zeros((n, N_chr), dtype=float)
		if args.folds > 0: # LOCO predictions from K-fold CV
			assoc_cy.haplotypeLOCO(L, E_hat, y_chr, y_hat, N_ind, B_list, r0)
		else: # LOCO predictions from LOOCV
			H = np.dot(V*(1.0/(S*S + lmbda[h2_opt])), V.T)
			p = np.dot(U*(S*S/(S*S + lmbda[h2_opt])), UtY)
			a = np.dot(V*(S/(S*S + lmbda[h2_opt])), UtY)
			assoc_cy.loocvLOCO(L, y_chr, y_hat, H, p, y, a, x, B_list, r0)
			del U, S, V, UtY, H, p, a, x
		np.savetxt(f"{args.out}.loco", y_chr, fmt="%.7f")
		print(f"Saved {N_chr} LOCO predictions as {args.out}.loco")


	##### Step 2 - Association testing #####
	if args.haplo_asso:
		# Perform association testing of haplotype clusters alleles
		if args.filelist is not None:
			y_chr = np.ascontiguousarray(y_chr.T)
			C_idx = 0
			W_idx = 0
			B_nxt = B_list[0]
			y_res = y - y_chr[0,:]
		else: # Use PRS (beware of proximal contamination)
			y_res = y - y_hat
		P = np.zeros((np.sum(K_vec), 8), dtype=float) # Output matrix
		B_idx = 0 # Start index for haplotype clusters
		s_env = np.linalg.norm(y_res)/sqrt(n - R_c)
		for b in np.arange(B):
			print(f"\rAssociation testing (clusters) - Block {b+1}/{B}", end="")
			B_num = np.sum(K_vec[B_arr[b]], dtype=int) - B_arr[b].shape[0]
			
			# Extract haplotype clusters and regress out covariates
			Z = np.zeros((B_num, n), dtype=float)
			assoc_cy.haplotypeAssoc(Z_mat, Z, P, B_arr[b], K_vec, B_idx)
			Z -= np.dot(np.dot(Z, U_c), U_c.T)

			# Create residualized phenotypes
			if args.filelist is not None: # Extract LOCO prediction for block
				if b == B_nxt: # Next chromosome
					C_idx += 1
					W_idx = 0
					B_nxt += B_list[C_idx]
					y_res = y - y_chr[C_idx,:]
					s_env = np.linalg.norm(y_res)/sqrt(n - R_c)
				P[B_idx:(B_idx + B_num),0] = C_idx + 1 # Chromosome info

			# Test haplotype clusters
			assoc_cy.haplotypeTest(Z, P, y_res, B_arr[b], K_vec, s_env, W_idx, B_idx)
			B_idx += B_num
			W_idx += B_arr[b].shape[0]

			# Free memory
			del Z
		print("")

		### Save association results
		P[:,7] = chi2.sf(P[:,6], df=1) # P-values (1 - cdf) - Wald's
		np.savetxt(f"{args.out}.assoc", P, fmt=["%i", "%i", "%i", "%.7f", "%.7f", \
			"%.7f", "%.7f", "%.7e"], header="chrom window cluster freq beta se chisq p",
			comments="")
		print(f"Saved association test statistics as {args.out}.haplo.assoc")



##### Main exception #####
assert __name__ != "__main__", "Please use the 'hapla regress' command!"
