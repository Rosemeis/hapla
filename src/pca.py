"""
hapla.
Perform PCA using haplotype cluster alleles.
"""

__author__ = "Jonas Meisner"

# Libraries
import os

##### hapla pca #####
def main(args):
	print("hapla by Jonas Meisner (v0.2)")
	print(f"hapla pca using {args.threads} thread(s).")

	# Check input
	assert (args.filelist is not None) or (args.clusters is not None), \
		"No input data (--filelist or --clusters)!"
	if args.min_freq is not None:
		assert args.min_freq > 0.0, "Empty haplotype clusters not allowed!"

	# Control threads of external numerical libraries
	os.environ["MKL_NUM_THREADS"] = str(args.threads)
	os.environ["OMP_NUM_THREADS"] = str(args.threads)
	os.environ["NUMEXPR_NUM_THREADS"] = str(args.threads)
	os.environ["OPENBLAS_NUM_THREADS"] = str(args.threads)

	# Import numerical libraries and cython functions
	import numpy as np
	from scipy.sparse.linalg import svds
	from src import functions
	from src import shared_cy

	# Load data (and concatentate across windows)
	if args.filelist is not None:
		Z_list = []
		with open(args.filelist) as f:
			file_c = 1
			for chr in f:
				Z_list.append(np.load(chr.strip("\n")))
				print(f"\rParsed file #{file_c}", end="")
				file_c += 1
		Z_mat = np.concatenate(Z_list, axis=0)
		del Z_list
	else:
		Z_mat = np.load(args.clusters)
	print("\rLoaded haplotype cluster alleles of " + \
		f"{Z_mat.shape[1]} haplotypes in {Z_mat.shape[0]} windows.")
	W = Z_mat.shape[0]
	n = Z_mat.shape[1]//2

	# Estimate total number of haplotype cluster alleles
	K_vec = np.max(Z_mat, axis=1) # Dummy encoding
	m = np.sum(K_vec, dtype=int)

	# Populate full matrix and estimate summary statistics
	Z = np.zeros((m, n), dtype=np.uint8)
	mu = np.zeros(m, dtype=float)
	si = np.zeros(m, dtype=float)
	shared_cy.haplotypeAggregate(Z_mat, Z, mu, si, K_vec)
	del Z_mat

	# Mask non-rare haplotype clusters
	if args.min_freq is not None:
		mask = (mu >= args.min_freq) & (mu <= (1 - args.min_freq))
		mask = mask.astype(np.uint8)
		m = np.sum(mask, dtype=int)

		# Filter out masked haplotype clusters
		shared_cy.filterZ(Z, mu, si, mask)
		Z = Z[:m,:]
		mu = mu[:m]
		si = si[:m]

	# Perform PCA
	if args.randomized:
		# Randomized SVD
		print(f"Performing randomized SVD, extracting {args.n_eig} eigenvectors.")
		U, S, V = functions.randomizedSVD(Z, mu, si, args.n_eig, args.batch, \
			args.threads)

		# Save matrices
		np.savetxt(f"{args.out}.eigenvec", V, fmt="%.7f")
		print(f"Saved eigenvectors as {args.out}.eigenvec")
		np.savetxt(f"{args.out}.eigenval", (S*S)/float(m), fmt="%.7f")
		print(f"Saved eigenvalues as {args.out}.eigenval")
		if args.loadings:
			np.savetxt(f"{args.out}.loadings", U, fmt="%.7f")
			print(f"Saved loadings as {args.out}.loadings")
	else:
		Z_std = np.zeros((m, n), dtype=float)
		shared_cy.standardizeZ(Z, Z_std, mu, si, args.threads)
		del Z
		if args.grm:
			# Estimate GRM
			print("Estimating genome-wide relationship matrix (GRM)")
			G = np.dot(Z_std.T, Z_std)/float(m)
		
			# Save matrix
			np.savetxt(f"{args.out}.grm", G, fmt="%.7f")
			print(f"Saved genome-wide relationship matrix (GRM) as {args.out}.grm")
		else:
			# Truncated SVD (Arnoldi)
			print(f"Performing truncated SVD, extracting {args.n_eig} eigenvectors.")
			U, S, Vt = svds(Z_std, k=args.n_eig)

			# Save matrices
			np.savetxt(f"{args.out}.eigenvec", Vt[::-1,:].T, fmt="%.7f")
			print(f"Saved eigenvectors as {args.out}.eigenvec")
			np.savetxt(f"{args.out}.eigenval", (S[::-1]*S[::-1])/float(m), fmt="%.7f")
			print(f"Saved eigenvalues as {args.out}.eigenval")
			if args.loadings:
				np.savetxt(f"{args.out}.loadings", U[:,::-1], fmt="%.7f")
				print(f"Saved loadings as {args.out}.loadings")



##### Main exception #####
assert __name__ != "__main__", "Please use the 'hapla pca' command!"
