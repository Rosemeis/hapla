"""
hapla.
Estimate polygenic risk scores using summary statistics or prediction model.
"""

__author__ = "Jonas Meisner"

# Libraries
import os

##### hapla prs #####
def main(args):
	print("hapla prs by Jonas Meisner (v0.1)")
	print(f"Using {args.threads} thread(s).")

	# Check input
	assert (args.filelist is not None) or (args.clusters is not None), \
		"No input data (--filelist or --clusters)!"
	assert args.assoc is not None, \
		"Please provide summary statistics (--assoc)!"

	# Control threads of external numerical libraries
	os.environ["MKL_NUM_THREADS"] = str(args.threads)
	os.environ["OMP_NUM_THREADS"] = str(args.threads)
	os.environ["NUMEXPR_NUM_THREADS"] = str(args.threads)
	os.environ["OPENBLAS_NUM_THREADS"] = str(args.threads)

	# Import numerical libraries and cython functions
	import numpy as np
	from hapla import assoc_cy

	### Load data
	# Load haplotype cluster assignments (and concatentate across windows)
	if args.filelist is not None:
		Z_list = []
		with open(args.filelist) as f:
			file_c = 1
			for chr in f:
				Z_list.append(np.load(chr.strip("\n")))
				print(f"\rParsed file #{file_c}", end="")
				file_c += 1
		Z = np.concatenate(Z_list, axis=0)
		del Z_list
	else:
		Z = np.load(args.clusters)
	print("\rLoaded haplotype cluster assignments of " + \
		f"{Z.shape[1]} haplotypes in {Z.shape[0]} windows.")
	W = Z.shape[0]
	n = Z.shape[1]//2

	# Estimate total number of haplotype cluster assignments
	w, b, p = np.loadtxt(args.assoc, dtype=float, unpack=True, \
		usecols=(1,4,7))
	w = w.astype(int)
	p = np.ascontiguousarray(p)
	assert w[-1] == W, "Number of windows differ between files!"
	K_vec = np.max(Z, axis=1) + 1
	m = np.sum(K_vec)
	assert b.shape[0] == m, "Number of haplotype cluster differ between files!"

	# Perform clumping
	if args.block is not None:
		print(f"Performing clumping in blocks of {args.block} windows.")
		mask = np.zeros(m, dtype=np.uint8)
		maskW = np.zeros(W, dtype=np.uint8)
		assoc_cy.clumpWindows(mask, maskW, K_vec, p, args.block)
		print(f"After clumping, {W-np.sum(maskW, dtype=int)} windows remain.")
		b[mask.astype(bool)] = 0
		del mask, maskW

	# Populate full matrix
	Z_tilde = np.zeros((m, n), dtype=np.uint8)
	assoc_cy.updateZ(Z, Z_tilde, K_vec)
	del Z

	### Standard PRS using summary statistics
	y_hat = np.dot(Z_tilde.T, b)
	np.savetxt(f"{args.out}.sumstats.prs", y_hat, fmt="%.7f")
	print(f"Saved PRS from summary statistics as {args.out}.sumstats.prs")



##### Main exception #####
assert __name__ != "__main__", "Please use the 'hapla prs' command!"
