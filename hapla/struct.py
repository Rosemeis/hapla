"""
hapla.
Population structure inference using haplotype cluster alleles.
"""

__author__ = "Jonas Meisner"

# Libraries
import os
from time import time

##### hapla struct #####
def main(args):
	print("-----------------------------------")
	print("hapla by Jonas Meisner (v0.5)")
	print(f"hapla struct using {args.threads} thread(s)")
	print("-----------------------------------\n")

	# Check input
	assert (args.filelist is not None) or (args.clusters is not None), \
		"No input data (--filelist or --clusters)!"
	assert (args.grm or (args.pca is not None)), "No analysis selected (--grm, --pca)!"
	if args.grm:
		assert args.iid is not None, "Provide sample list for GCTA format (--iid)!"
	if args.pca is not None:
		assert args.pca > 0, "Please select a valid number of eigenvectors!"
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
	from hapla import functions
	from hapla import shared_cy

	# Load data (and concatentate across windows)
	if args.filelist is not None:
		Z_list = []
		with open(args.filelist) as f:
			file_c = 1
			for chr in f:
				if args.hsm:
					Z_list.append(np.ascontiguousarray(np.load(chr.strip("\n")).T))
				else:
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

	# Count haplotype cluster alleles
	K_vec = np.max(Z_mat, axis=1) + 1
	m = np.sum(K_vec, dtype=int)

	# Populate full matrix and estimate haplotype cluster allele frequencies
	Z = np.zeros((m, n), dtype=np.uint8)
	p = np.zeros(m, dtype=np.float32)
	s = np.zeros(m, dtype=np.float32)
	shared_cy.haplotypeAggregate(Z_mat, Z, p, s, K_vec)
	del Z_mat

	# Mask non-rare haplotype clusters
	if args.min_freq is not None:
		mask = (p >= args.min_freq) & (p <= (1 - args.min_freq))
		mask = mask.astype(np.uint8)
		print(f"Removed {m-np.sum(np.sum(mask, dtype=int))}/{m} haplotype clusters.")
		m = np.sum(mask, dtype=int)

		# Filter out masked haplotype clusters
		shared_cy.filterZ(Z, p, mask)
		Z = Z[:m,:]
		p = p[:m]
		s = s[:m]
	
	# Estimate genome-wide relationship matrix
	if args.grm: # 
		print("Estimating genome-wide relationship matrix (GRM).")
		K = n*(n+1)//2
		
		# Standardize
		s = np.power(s, 0.5*args.alpha)
		Z_s = np.zeros((m, n), dtype=np.float32)
		shared_cy.standardizeZ(Z, Z_s, p, s, args.threads)

		# Estimate GRM
		G = np.dot(Z_s.T, Z_s)/float(m)
		del Z_s
		
		# Centering
		print("Centering GRM.")
		p = np.mean(G, axis=1)
		G -= p.reshape(1, n)
		p = np.mean(G, axis=1)
		G -= p.reshape(n, 1)

		# Gower centering
		G *= float(n-1)/np.trace(G)

		# Save matrix
		G = G[np.tril_indices(n)]
		G.tofile(f"{args.out}.grm.bin")
		np.full(K, m, dtype=np.float32).tofile(f"{args.out}.grm.N.bin")
		iid = np.loadtxt(f"{args.iid}", dtype=np.str_).reshape(-1,1)
		if args.fid is not None:
			fid = np.loadtxt(f"{args.fid}", dtype=np.str_).reshape(-1,1)
			fam = np.hstack((fid, iid))
		else:
			fam = np.hstack((np.zeros((n, 1), dtype=np.uint8), iid))
		np.savetxt(f"{args.out}.grm.id", fam, delimiter="\t", fmt="%s")
		print("Saved genome-wide relationship matrix in GCTA format:\n" + \
			f"- {args.out}.grm.bin\n" + \
			f"- {args.out}.grm.N.bin\n" + \
			f"- {args.out}.grm.id")
	
	# Infer population structure
	if args.pca is not None:
		if args.randomized:
			print(f"Computing randomized SVD, extracting {args.eig} eigenvectors.")

			# Randomized SVD in batches
			s = np.power(s, -0.5)
			U, S, V = functions.randomizedSVD(Z, p, s, args.eig, args.batch, \
				args.threads)

			# Save matrices
			if args.iid is not None:
				iid = np.loadtxt(f"{args.iid}", dtype=np.str_).reshape(-1,1)
				if args.fid is not None:
					fid = np.loadtxt(f"{args.fid}", dtype=np.str_).reshape(-1,1)
					fam = np.hstack((fid, iid))
				else:
					fam = np.hstack((np.zeros((n, 1), dtype=np.uint8), iid))
				out = np.hstack((fam, np.round(V, 7)))
				np.savetxt(f"{args.out}.eigenvec", out, fmt="%s")
			else:
				np.savetxt(f"{args.out}.eigenvec", V, fmt="%.7f")
			print(f"Saved eigenvectors as {args.out}.eigenvec")
			np.savetxt(f"{args.out}.eigenval", (S*S)/float(m), fmt="%.7f")
			print(f"Saved eigenvalues as {args.out}.eigenval")
			if args.loadings:
				np.savetxt(f"{args.out}.loadings", U, fmt="%.7f")
				print(f"Saved loadings as {args.out}.loadings")
		else:
			print(f"Computing truncated SVD, extracting {args.eig} eigenvectors.")

			# Standardize
			s = np.power(s, -0.5)
			Z_s = np.zeros((m, n), dtype=np.float32)
			shared_cy.standardizeZ(Z, Z_s, p, s, args.threads)

			# Truncated SVD (Arnoldi)
			U, S, Vt = svds(Z_s, k=args.eig)
			del Z_s

			# Save matrices
			if args.iid is not None:
				iid = np.loadtxt(f"{args.iid}", dtype=np.str_).reshape(-1,1)
				if args.fid is not None:
					fid = np.loadtxt(f"{args.fid}", dtype=np.str_).reshape(-1,1)
					fam = np.hstack((fid, iid))
				else:
					fam = np.hstack((np.zeros((n, 1), dtype=np.uint8), iid))
				out = np.hstack((fam, np.round(Vt[::-1,:].T, 7)))
				np.savetxt(f"{args.out}.eigenvec", out, fmt="%s")
			else:
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
assert __name__ != "__main__", "Please use the 'hapla struct' command!"
