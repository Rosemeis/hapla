"""
hapla.
Perform association testing using haplotype cluster alleles or SNPs.
"""

__author__ = "Jonas Meisner"

##### hapla regress #####
def main(args):
	print("hapla by Jonas Meisner (v0.2)")
	print(f"hapla asso using {args.threads} thread(s).")

	# Check input
	assert args.pheno is not None, "No phenotype file provided!"
	assert (args.loco is not None) or (args.pred is not None), \
		"Whole-genome predictions not provided! Please run 'hapla regress'!"
	if args.loco is not None:
		assert args.filelist is not None, \
			"No filelist with paths to multiple input files provided for LOCO!"
	else:
		assert (args.clusters is not None) or (args.geno is not None), \
			"No input data provided!"
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
	from cyvcf2 import VCF
	from math import ceil, sqrt
	from scipy.stats import chi2
	from src import functions
	from src import assoc_cy

	### Load data
	# Load phenotype file (outcome)
	y = np.loadtxt(args.pheno, dtype=float)
	n = y.shape[0]
	print("Loaded phenotype file.")

	# Load whole-genome predictions
	if args.loco is not None:
		y_chr = np.ascontiguousarray(np.loadtxt(args.loco, dtype=float).T)
		assert y_chr.shape[0] == n, "Number of samples differ between files!"
	else:
		y_hat = np.loadtxt(args.pred, dtype=float).reshape(-1)
		assert y_hat.shape[0] == n, "Number of samples differ between files!"
	print("Loaded whole-genome predictions.")

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

	### Residualize and scale phenotypes by covariates
	U_c, _, _ = functions.fastSVD(C)
	R_c = U_c.shape[1]
	y -= np.dot(U_c, np.dot(U_c.T, y))
	y /= np.linalg.norm(y)/sqrt(n - R_c)


	##### Step 2 - Association testing #####
	### Check number of chromosomes
	if args.filelist is not None:
		with open(args.filelist) as f:
			N_chr = 0
			filelist = []
			for c in f:
				filelist.append(c.strip("\n"))
				N_chr += 1
	
	### Perform association testing
	if args.filelist is not None:
		for c in np.arange(N_chr):
			# Load input data per chromosome
			print(f"\rAssociation testing (SNPs) - Chromosome {c+1}/{N_chr}", end="")
			v_file = VCF(files[c], threads=args.threads)
			assert len(v_file.samples) == n, "Number of samples differ between files!"
			G_mat = reader_cy.readVCF(v_file, n, ceil(2*n/8))
			del v_file
			m = G_mat.shape[0] # Number of SNPs in chromosome

			# Setup parameters for chromosome
			P = np.zeros((m, 7), dtype=float) # Output matrix
			B = ceil(m/args.block)
			G = np.zeros((args.block, n), dtype=float)
			y_res = y - y_chr[c,:] # Residualized phenotype
			s_env = np.linalg.norm(y_res)/sqrt(n - R_c)

			# Extract SNP block and regress out covariates
			for b in np.arange(B):
				B_idx = b*args.block
				if (B_idx + args.block) >= m: # Last block
					del G # Ensure no extra copy
					G = np.zeros((m - B_idx, n), dtype=float)
				assoc_cy.genotypeAssoc(G_mat, G, P, B_idx)
				G -= np.dot(np.dot(G, U_c), U_c.T)
				assoc_cy.genotypeTest(G, P, y_res, s_env, B_idx)
			P[:,6] = chi2.sf(P[:,5], df=1) # P-values (1 - cdf) - Wald's
			
			# Save association results
			P[:,0] = c+1 # Chromosome number information
			v_file = VCF(files[c], threads=args.threads)
			reader_cy.readPOS(v_file, P) # Read base positions
			if c == 0: # First chromosome
				np.savetxt(f"{args.out}.snp.assoc", P, \
					fmt=["%i", "%i", "%.7f", "%.7f", "%.7f", "%.7f", "%.7e"], \
					header="chrom pos freq beta se chisq p", comments="")
			else: # Append other chromosomes
				with open(f"{args.out}.snp.assoc", "a") as f:
					np.savetxt(f, P, \
						fmt=["%i", "%i", "%.7f", "%.7f", "%.7f", "%.7f", "%.7e"])
			del P, v_file
		print("")
	else: # Single chromosome only - beware of proximal contamination
		print("Association testing (SNPs)")
		v_file = VCF(args.vcf, threads=args.threads)
		assert len(v_file.samples) == n, "Number of samples differ between files!"
		G_mat = reader_cy.readVCF(v_file, n, ceil(2*n/8)) # 1-bit geno matrix
		del v_file
		m = G_mat.shape[0] # Number of SNPs in chromosome

		# Setup parameters
		P = np.zeros((m, 7), dtype=float) # Output matrix
		B = ceil(m/args.block)
		G = np.zeros((args.block, n), dtype=float)
		y_res = y - y_hat # Residualized phenotype
		s_env = np.linalg.norm(y_res)/sqrt(n - R_c)

		# Extract SNP block and regress out covariates
		for b in np.arange(B):
			B_idx = b*args.block
			if (B_idx + args.block) >= m: # Last block
				G = np.zeros((m - B_idx, n), dtype=float)
			assoc_cy.genotypeAssoc(G_mat, G, P, B_idx)
			G -= np.dot(np.dot(G, U_c), U_c.T)
			assoc_cy.genotypeTest(G, P, y_res, s_env, B_idx)
		P[:,6] = chi2.sf(P[:,5], df=1) # P-values (1 - cdf) - Wald's
		
		# Save association results
		if args.chrom is not None: # Chromosome number information
			P[:,0] = args.chrom
		else:
			P[:,0] = 1
		v_file = VCF(args.vcf, threads=args.threads)
		reader_cy.readPOS(v_file, P) # Read base positions
		np.savetxt(f"{args.out}.snp.assoc", P, \
			fmt=["%i", "%i", "%.7f", "%.7f", "%.7f", "%.7f", "%.7e"], \
			header="chrom pos freq beta se chisq p", comments="")
		del P, v_file
	print(f"Saved SNP association test statistics as {args.out}.snp.assoc")



##### Main exception #####
assert __name__ != "__main__", "Please use the 'hapla asso' command!"
