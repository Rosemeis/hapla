"""
hapla.
Estimate admixture proportions using EM algorithm.
"""

__author__ = "Jonas Meisner"

# Libraries
import os
from time import time

##### hapla admix #####
def main(args):
	print("-----------------------------------")
	print("hapla by Jonas Meisner (v0.8)")
	print(f"hapla admix using {args.threads} thread(s)")
	print("-----------------------------------\n")

	# Check input
	assert (args.filelist is not None) or (args.clusters is not None), \
		"No input data (--filelist or --clusters)!"
	assert args.K > 1, "Please set K > 1 (--K)!"
	start = time()

	# Control threads of external numerical libraries
	os.environ["MKL_NUM_THREADS"] = str(args.threads)
	os.environ["OMP_NUM_THREADS"] = str(args.threads)
	os.environ["NUMEXPR_NUM_THREADS"] = str(args.threads)
	os.environ["OPENBLAS_NUM_THREADS"] = str(args.threads)

	# Import numerical libraries and cython functions
	import numpy as np
	from hapla import admix_cy
	from hapla import functions

	# Load data (and concatentate across windows)
	if args.filelist is not None:
		Z_list = []
		with open(args.filelist) as f:
			for z_file in f:
				Z_list.append(z_file.strip("\n"))
	else:
		Z_list = [args.clusters]
	print(f"Parsing {len(Z_list)} file(s).")

	# Load and merge haplotype cluster assignments
	Z_tmp = []
	for z in np.arange(len(Z_list)):
		Z_tmp.append(np.load(Z_list[z]))
		print(f"\rParsed file {z+1}/{len(Z_list)}", end="")
	Z = np.concatenate(Z_tmp, axis=0)
	del Z_tmp
	W = Z.shape[0]
	n = Z.shape[1]
	if args.haplo:
		N = 1
		S = float(W)
		n_str = "haplotypes"
	else:
		N = 2
		S = float(2*W)
		n_str = "samples"

	# Count haplotype cluster alleles
	k_vec = np.max(Z, axis=1) + 1
	C = np.max(k_vec)
	m = np.sum(k_vec, dtype=int)

	# Print information
	print(f"\rLoaded haplotype cluster assignments:\n" + \
		f"- {n//N} {n_str}\n" + \
		f"- {W} windows\n" + \
		f"- {m} clusters\n")
	print(f"Estimating admixture proportions: K={args.K}, seed={args.seed}.")
	
	# Initialize P and Q matrix
	np.random.seed(args.seed) # Set random seed
	P = np.random.rand(W, args.K, C)
	P.clip(min=1e-5, max=1-(1e-5), out=P)
	admix_cy.createP(P, k_vec)
	Q = np.random.rand(n//N, args.K)
	Q.clip(min=1e-5, max=1-(1e-5), out=Q)
	Q /= np.sum(Q, axis=1, keepdims=True)
	Q_new = np.zeros_like(Q)
	
	# Setup containers for EM algorithm
	l_vec = np.zeros(W)
	P0 = np.zeros_like(P)
	Q0 = np.zeros_like(Q)
	dP1 = np.zeros_like(P)
	dP2 = np.zeros_like(P)
	dP3 = np.zeros_like(P)
	dQ1 = np.zeros_like(Q)
	dQ2 = np.zeros_like(Q)
	dQ3 = np.zeros_like(Q)

	# Estimate initial log-likelihood
	ts = time()
	admix_cy.loglike(Z, P, Q, l_vec, N, args.threads)
	L_pre = np.sum(l_vec)
	print(f"Initial loglike: {round(L_pre,1)}\n")

	# Prime iteration
	admix_cy.updateP(Z, P, Q, Q_new, k_vec, N, args.threads)
	admix_cy.updateQ(Q, Q_new, S, args.threads)

	# Accelerated EM algorithm
	ts = time()
	print(f"Accelerated SQUAREM.")
	for it in range(args.iter):
		# SQUAREM full update
		functions.squarem(Z, P, Q, P0, Q0, Q_new, dP1, dP2, dP3, dQ1, dQ2, dQ3, \
			k_vec, S, N, args.threads)
		
		# Stabilization step
		admix_cy.updateP(Z, P, Q, Q_new, k_vec, N, args.threads)
		admix_cy.updateQ(Q, Q_new, S, args.threads)

		# Log-likelihood convergence check
		if ((it+1) % args.check) == 0:
			admix_cy.loglike(Z, P, Q, l_vec, N, args.threads)
			L_cur = np.sum(l_vec)
			print(f"({it+1})\tLog-like: {round(L_cur,1)}\t" + \
				f"({round(time()-ts,1)}s)", flush=True)
			if (abs(L_cur - L_pre) < args.tole):
				print("Converged!")
				print(f"Final log-likelihood: {round(L_cur,1)}")
				break
			L_pre = L_cur
			ts = time()

	# Save output
	np.savetxt(f"{args.out}.K{args.K}.s{args.seed}.Q", Q, fmt="%.6f")
	print(f"Saved Q matrix as {args.out}.K{args.K}.s{args.seed}.Q")
	if not args.no_freq:
		np.save(f"{args.out}.K{args.K}.s{args.seed}.P", P)
		print(f"Saved P matrix as {args.out}.K{args.K}.s{args.seed}.P.npy")

	# Print elapsed time for computation
	t_tot = time()-start
	t_min = int(t_tot//60)
	t_sec = int(t_tot - t_min*60)
	print(f"Total elapsed time: {t_min}m{t_sec}s")



##### Main exception #####
assert __name__ != "__main__", "Please use the 'hapla admix' command!"
