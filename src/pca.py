"""
hapla.
Perform PCA using haplotype cluster assignments.
"""

__author__ = "Jonas Meisner"

# Libraries
import os

##### hapla pca #####
def main(args):
	print("hapla pca by Jonas Meisner (v0.1)")
	print(f"Using {args.threads} thread(s).")

	# Check input
	assert (args.filelist is not None) or (args.clusters is not None), \
		"No input data (--filelist or --clusters)!"
	assert args.min_count > 0, "Empty haplotype clusters not allowed!"

	# Control threads of external numerical libraries
	os.environ["MKL_NUM_THREADS"] = str(args.threads)
	os.environ["OMP_NUM_THREADS"] = str(args.threads)
	os.environ["NUMEXPR_NUM_THREADS"] = str(args.threads)
	os.environ["OPENBLAS_NUM_THREADS"] = str(args.threads)

	# Import numerical libraries and cython functions
	import numpy as np
	from scipy.sparse.linalg import svds
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
	print("\rLoaded haplotype cluster assignments of " + \
		f"{Z_mat.shape[1]} haplotypes in {Z_mat.shape[0]} windows.")
	n = Z_mat.shape[1]//2

	# Estimate total number of haplotype cluster assignments
	K_vec = np.max(Z_mat, axis=1) + 1
	m = np.sum(K_vec)

	# Estimate haplotype cluster frequencies and populate full matrix
	pi = np.zeros(m, dtype=float)
	Z_tilde = np.zeros((m, n), dtype=float)
	shared_cy.haplotypeFreqs(Z_mat, Z_tilde, K_vec, pi)
	del Z_mat

	# Mask non-rare haplotype clusters
	freq = args.min_count/float(n)
	mask = (pi >= freq) & (pi <= (1 - freq))
	mask = mask.astype(np.uint8)
	m_new = np.sum(mask)

	# Filter out masked haplotype clusters
	shared_cy.filterZ(Z_tilde, pi, mask)
	Z_tilde = Z_tilde[:m_new,:]
	pi = pi[:m_new]

	# Standardize Z matrix of individuals
	shared_cy.standardizeZ(Z_tilde, pi, args.threads)
	if not args.cov:
		# Perform SVD
		print(f"Performing SVD, extracting {args.n_eig} eigenvectors.")
		U, S, Vt = svds(Z_tilde, k=args.n_eig)

		# Save matrices
		np.savetxt(f"{args.out}.eigenvec", Vt[::-1,:].T, fmt="%.7f")
		print(f"Saved eigenvectors as {args.out}.eigenvec")
		np.savetxt(f"{args.out}.eigenval", (S[::-1]**2)/float(m), fmt="%.7f")
		print(f"Saved eigenvalues as {args.out}.eigenval")
		if args.loadings:
			np.savetxt(f"{args.out}.loadings", U[:,::-1], fmt="%.7f")
			print(f"Saved loadings as {args.out}.loadings")
		del Z_tilde, Vt
		
		# Project other samples into vector space
		if (args.project is not None) or (args.project_filelist is not None):
			if args.project_filelist is not None:
				Z_list = []
				with open(args.project_filelist) as f:
					file_c = 1
					for chr in f:
						Z_list.append(np.load(chr.strip("\n")))
						print(f"\rParsed file #{file_c}", end="")
						file_c += 1
				Z_mat = np.concatenate(Z_list, axis=0)
				del Z_list
			else:
				Z_mat = np.load(args.project)
			print("\rLoaded haplotype cluster assignments of " + \
				f"{Z_mat.shape[1]} haplotypes in {Z_mat.shape[0]} windows.")
			n = Z_mat.shape[1]//2

			# Extract haplotype cluster assignments and standardize
			Z_tilde = np.zeros((m_new, n), dtype=float)
			shared_cy.updateZ(Z_mat, Z_tilde, pi, K_vec, mask)
			del Z_mat

			# Projection and save output
			V = np.dot(Z_tilde.T, U*(1.0/S))
			if args.project_out is not None:
				project_out = args.project_out
			else:
				project_out = args.out
			np.savetxt(f"{project_out}.projection", V[:,::-1], fmt="%.7f")
			print(f"Saved projections as {args.out}.projection")
	else:
		# Estimate covariance matrix
		print("Estimating covariance matrix (GRM)")
		C = np.dot(Z_tilde.T, Z_tilde)/float(m)
		
		# Save matrix
		np.savetxt(f"{args.out}.cov", C, fmt="%.7f")
		print(f"Saved covariance matrix (GRM) as {args.out}.cov")



##### Main exception #####
assert __name__ != "__main__", "Please use the 'hapla pca' command!"
