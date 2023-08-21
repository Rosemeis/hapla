"""
hapla.
Perform association testing using haplotype cluster alleles or SNPs.
"""

__author__ = "Jonas Meisner"

##### hapla regress #####
def main(args):
	print("hapla by Jonas Meisner (v0.2)")
	print(f"hapla asso using {args.threads} thread(s).\n")

	# Check input
	assert args.filelist is not None, "No input data provided!"
	assert args.pheno is not None, "No phenotype file provided!"
	assert args.loco is not None, "Whole-genome predictions not provided!"
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
	from src import asso_cy

	### Load data
	# Load phenotype file (outcome)
	y = np.loadtxt(args.pheno, dtype=float)
	n = y.shape[0]
	print("Loaded phenotype file.")

	# Load LOCO predictions
	y_chr = np.ascontiguousarray(np.loadtxt(args.loco, dtype=float).T)
	assert y_chr.shape[0] == n, "Number of samples differ between files!"
	print("Loaded LOCO predictions.")

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
	U_cov, _, _ = functions.fastSVD(C)
	R_cov = U_cov.shape[1]
	del C
	y_res = y - np.dot(U_cov, np.dot(U_cov.T, y))
	y_res /= np.linalg.norm(y_res)/sqrt(n - R_cov)


	##### Step 2 - Association testing #####
	### Check input type and number of chromosomes
	with open(args.filelist) as f:
		N_chr = 0
		files = []
		for c_idx in f:
			files.append(c_idx.strip("\n"))
			if N_chr == 0:
				if ("vcf" in files[0][-6:]) or ("bcf" in files[0][-6:]):
					snp_based = True
				else:
					snp_based = False
					assert ".z.npy" in files[0][-6:], "Wrong file formats provided!"
			N_chr += 1
	
	### Perform association testing
	for c_idx in np.arange(N_chr):
		y_loc = y_res - y_chr[c_idx,:] # Residualized phenotype
		s_env = np.linalg.norm(y_loc)/sqrt(n - R_cov)

		# Load input data per chromosome
		if snp_based: # SNP testing
			print(f"\rAssociation testing (SNP) - Chromosome {c_idx+1}/{N_chr}", end="")
			v_file = VCF(files[c_idx], threads=args.threads)
			assert len(v_file.samples) == n, "Number of samples differ between files!"
			G_mat = reader_cy.readVCF(v_file, n, ceil(2*n/8))
			del v_file
			m = G_mat.shape[0] # Number of SNPs in chromosome

			# Setup parameters for chromosome
			P = np.zeros((m, 7), dtype=float) # Output matrix
			B = ceil(m/args.block)
			G = np.zeros((args.block, n), dtype=float)

			# Extract SNP block, regress out covariates and test
			for b in np.arange(B):
				B_idx = b*args.block
				if (B_idx + args.block) >= m: # Last block
					del G # Ensure no extra copy
					G = np.zeros((m - B_idx, n), dtype=float)
				asso_cy.genotypeAssoc(G_mat, G, P, B_idx)
				G -= np.dot(np.dot(G, U_cov), U_cov.T)
				asso_cy.genotypeTest(G, P, y_loc, s_env, B_idx)
			
			# Save association results
			P[:,0] = c_idx+1 # Chromosome number information
			P[:,6] = chi2.sf(P[:,5], df=1) # P-values (1 - cdf) - Wald's
			v_file = VCF(files[c_idx], threads=args.threads)
			reader_cy.readPOS(v_file, P) # Read base positions
			if c_idx == 0: # First chromosome
				np.savetxt(f"{args.out}.snp.assoc", P, \
					fmt=["%i", "%i", "%.7f", "%.7f", "%.7f", "%.7f", "%.7e"], \
					header="chrom pos freq beta se chisq p", comments="")
			else: # Append other chromosomes
				with open(f"{args.out}.snp.assoc", "a") as f:
					np.savetxt(f, P, \
						fmt=["%i", "%i", "%.7f", "%.7f", "%.7f", "%.7f", "%.7e"])
			del P, v_file
		else: # Haplotype cluster allele testing
			print("\rAssociation testing (Cluster) - Chromosome "+ \
				f"{c_idx+1}/{N_chr}", end="")
			Z_mat = np.load(files[c_idx])
			W = Z_mat.shape[0]
			assert Z_mat.shape[1]//2 == n, "Number of samples differ between files!"
			K_chr = np.max(Z_mat, axis=1) + 1
			m = np.sum(K_chr)

			# Setup parameters for chromosome
			B = 0
			P = np.zeros((m, 8), dtype=float) # Output matrix

			# Extract haplotype cluster block, regress out covariates and test
			for w in np.arange(W):
				Z = np.zeros((K_chr[w], n), dtype=float)
				assoc_cy.haplotypeAssoc(Z_mat, Z, P, B, w)
				Z -= np.dot(np.dot(Z, U_cov), U_cov.T)
				assoc_cy.haplotypeTest(Z, P, y_loc, s_env, B, w)
				B += K_chr[w]
				del Z

			# Save association results
			P[:,0] = c_idx+1 # Chromosome number information
			P[:,7] = chi2.sf(P[:,6], df=1) # P-values (1 - cdf) - Wald's
			if c_idx == 0: # First chromosome
				np.savetxt(f"{args.out}.haplo.assoc", P, \
					fmt=["%i", "%i", "%i", "%.7f", "%.7f", "%.7f", "%.7f", "%.7e"], \
					header="chrom window cluster freq beta se chisq p", comments="")
			else: # Append other chromosomes
				with open(f"{args.out}.haplo.assoc", "a") as f:
					np.savetxt(f, P, \
						fmt=["%i", "%i", "%i", "%.7f", "%.7f", "%.7f", "%.7f", "%.7e"])
			del P
		print("")
	if snp_based:
		print(f"\nSaved SNP association test statistics as {args.out}.snp.assoc")
	else:
		print("\nSaved haplotype cluster association test statistics as " + \
			f"{args.out}.haplo.assoc")



##### Main exception #####
assert __name__ != "__main__", "Please use the 'hapla asso' command!"
