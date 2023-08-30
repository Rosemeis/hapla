"""
hapla.
Perform PCA using haplotype cluster alleles.
"""

__author__ = "Jonas Meisner"

# Libraries
import os
from time import time

##### hapla pca #####
def main(args):
	print("--------------------------------")
	print("hapla by Jonas Meisner (v0.2)")
	print(f"hapla pca using {args.threads} thread(s)")
	print("--------------------------------\n")

	# Check input
	assert (args.filelist is not None) or (args.clusters is not None), \
		"No input data (--filelist or --clusters)!"
	if args.grm or args.hsm:
		assert args.iid is not None, "Provide sample list for GCTA format!"
	if args.min_freq is not None:
		assert args.min_freq > 0.0, "Empty haplotype clusters not allowed!"
	start = time()

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
	W = Z_mat.shape[0]
	n = Z_mat.shape[1]//2
	print(f"\rLoaded haplotype cluster assignments of {n} samples in {W} windows.")

	# Estimate either HSM, GRM or perform PCA
	if args.hsm: # Haplotype sharing matrix
		print("Estimating haplotype sharing matrix (HSM).")
		K = n*(n+1)//2
		Z = np.ascontiguousarray(Z_mat.T)
		del Z_mat
		if args.gower:
			G = np.zeros((n, n), dtype=np.float32)
			shared_cy.hsmFull(Z, G, K, args.threads)
			
			# Gower centering
			print("Performing Gower centering on HSM.")
			P = np.eye(n, dtype=np.float32) - 1.0/float(n)
			G = (float(n-1)/np.trace(np.dot(P, np.dot(G, P))))*G
			
			# Save matrix
			G = G[np.tril_indices(n)]
		else:
			G = np.zeros(K, dtype=np.float32)
			shared_cy.hsmCondensed(Z, G, args.threads)
		
		# Save matrix
		G.tofile(f"{args.out}.hsm.grm.bin")
		del G
		np.full(K, 2*W, dtype=np.float32).tofile(f"{args.out}.hsm.grm.N.bin")
		iid = np.loadtxt(f"{args.iid}", dtype=np.str_).reshape(-1,1)
		if args.fid is not None:
			fid = np.loadtxt(f"{args.fid}", dtype=np.str_).reshape(-1,1)
			fam = np.hstack((fid, iid))
		else:
			fam = np.hstack((np.zeros((n, 1), dtype=np.uint8), iid))
		np.savetxt(f"{args.out}.hsm.grm.id", fam, delimiter="\t", fmt="%s")
		print("Saved haplotype sharing matrix (HSM) in GCTA format:")
		print(f"- {args.out}.hsm.grm.bin\n" + \
			f"- {args.out}.hsm.grm.N.bin\n" + \
			f"- {args.out}.hsm.grm.id")
	elif args.grm: # Genome-wide relationship matrix
		print("Estimating genome-wide relationship matrix (GRM).")
		K = n*(n+1)//2
		K_vec = np.max(Z_mat, axis=1) # Dummy encoding
		m = np.sum(K_vec, dtype=int)

		# Populate full matrix and estimate frequencies
		Z_pop = np.zeros((m, n), dtype=np.uint8)
		p = np.zeros(m, dtype=np.float32)
		shared_cy.haplotypeAggregate(Z_mat, Z_pop, p, K_vec)
		del Z_mat
		s = np.power(2*p*(1-p), args.alpha)
		Z = np.ascontiguousarray(Z_pop.T)
		del Z_pop
		if args.gower:
			G = np.zeros((n, n), dtype=np.float32)
			shared_cy.grmFull(Z, G, p, s, K, args.threads)

			# Gower centering
			print("Performing Gower centering on GRM.")
			P = np.eye(n, dtype=np.float32) - 1.0/float(n)
			G = (float(n-1)/np.trace(np.dot(P, np.dot(G, P))))*G
			
			# Save matrix
			G = G[np.tril_indices(n)]
		else:
			G = np.zeros(K, dtype=np.float32)
			shared_cy.grmCondensed(Z, G, p, s, args.threads)

		# Save matrix
		G.tofile(f"{args.out}.grm.bin")
		np.full(K, m, dtype=np.float32).tofile(f"{args.out}.grm.N.bin")
		iid = np.loadtxt(f"{args.iid}", dtype=np.str_).reshape(-1,1)
		if args.fid is not None:
			fid = np.loadtxt(f"{args.fid}", dtype=np.str_).reshape(-1,1)
			fam = np.hstack((fid, iid))
		else:
			fam = np.hstack((np.zeros((n, 1), dtype=np.uint8), iid))
		np.savetxt(f"{args.out}.grm.id", fam, delimiter="\t", fmt="%s")
		print("Saved genome-wide relationship matrix (GRM) in GCTA format:")
		print(f"- {args.out}.grm.bin\n" + \
			f"- {args.out}.grm.N.bin\n" + \
			f"- {args.out}.grm.id")
	else: # Principal component analysis
		print("Performing principal component analysis (PCA).")
		K_vec = np.max(Z_mat, axis=1) # Dummy encoding
		m = np.sum(K_vec, dtype=int)

		# Populate full matrix and estimate frequencies
		Z = np.zeros((m, n), dtype=np.uint8)
		p = np.zeros(m, dtype=np.float32)
		shared_cy.haplotypeAggregate(Z_mat, Z, p, K_vec)
		del Z_mat

		# Mask non-rare haplotype clusters
		if args.min_freq is not None:
			mask = (p >= args.min_freq) & (p <= (1 - args.min_freq))
			mask = mask.astype(np.uint8)
			print(f"Removed {m-np.sum(np.sum(mask, dtype=int))} haplotype clusters.")
			m = np.sum(mask, dtype=int)

			# Filter out masked haplotype clusters
			shared_cy.filterZ(Z, p, mask)
			Z = Z[:m,:]
			p = p[:m]

		# Perform PCA or estimate genome-wide relationship matrices
		if args.randomized:
			# Randomized SVD
			print(f"Computing randomized SVD, extracting {args.eig} eigenvectors.")
			U, S, V = functions.randomizedSVD(Z, p, args.eig, args.batch, args.threads)

			# Save matrices
			np.savetxt(f"{args.out}.eigenvec", V, fmt="%.7f")
			print(f"Saved eigenvectors as {args.out}.eigenvec")
			np.savetxt(f"{args.out}.eigenval", (S*S)/float(m), fmt="%.7f")
			print(f"Saved eigenvalues as {args.out}.eigenval")
			if args.loadings:
				np.savetxt(f"{args.out}.loadings", U, fmt="%.7f")
				print(f"Saved loadings as {args.out}.loadings")
		else:
			Z_std = np.zeros((m, n), dtype=np.float32)
			shared_cy.standardizeZ(Z, Z_std, p, args.threads)
			del Z

			# Truncated SVD (Arnoldi)
			print(f"Computing truncated SVD, extracting {args.eig} eigenvectors.")
			U, S, Vt = svds(Z_std, k=args.eig)

			# Save matrices
			np.savetxt(f"{args.out}.eigenvec", Vt[::-1,:].T, fmt="%.7f")
			print(f"Saved eigenvectors as {args.out}.eigenvec")
			np.savetxt(f"{args.out}.eigenval", (S[::-1]*S[::-1])/float(m), \
				fmt="%.7f")
			print(f"Saved eigenvalues as {args.out}.eigenval")
			if args.loadings:
				np.savetxt(f"{args.out}.loadings", U[:,::-1], fmt="%.7f")
				print(f"Saved loadings as {args.out}.loadings")

	# Print elapsed time for estimation
	t_tot = time()-start
	t_min = int(t_tot//60)
	t_sec = int(t_tot - t_min*60)
	print(f"Total elapsed time: {t_min}m{t_sec}s")



##### Main exception #####
assert __name__ != "__main__", "Please use the 'hapla pca' command!"
