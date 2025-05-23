"""
hapla.
Ancestry estimation using accelerated EM algorithm.
"""

__author__ = "Jonas Meisner"

# Libraries
import os
from datetime import datetime
from time import time

VERSION = "0.25.0"

##### hapla admix #####
def main(args, deaf):
	print("-----------------------------------")
	print(f"hapla by Jonas Meisner (v{VERSION})")
	print(f"hapla admix using {args.threads} thread(s)")
	print("-----------------------------------\n")

	# Check input
	assert (args.filelist is not None) or (args.clusters is not None), "No input data (--filelist or --clusters)!"
	assert args.K > 1, "Please select K > 1!"
	assert args.threads > 0, "Please select a valid number of threads!"
	assert args.seed >= 0, "Please select a valid seed!"
	assert args.iter > 0, "Please select a valid number of iterations!"
	assert args.tole >= 0.0, "Please select a valid tolerance!"
	assert args.batches > 0, "Please select a valid number of mini-batches!"
	assert args.check > 0, "Please select a valid value for convergence check!"
	start = time()

	# Create log-file of used arguments
	full = vars(args)
	mand = ["seed", "batches"]
	with open(f"{args.out}.K{args.K}.s{args.seed}.log", "w") as log:
		log.write(f"hapla v{VERSION}\n")
		log.write("hapla admix\n")
		log.write(f"Time: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")
		log.write(f"Directory: {os.getcwd()}\n")
		log.write("Options:\n")
		for key in full:
			if full[key] != deaf[key]:
				if type(full[key]) is bool:
					log.write(f"\t--{key}\n")
				else:
					log.write(f"\t--{key} {full[key]}\n")
			elif key in mand:
				log.write(f"\t--{key} {full[key]}\n")
	del full, deaf, mand

	# Control threads of external numerical libraries
	os.environ["MKL_NUM_THREADS"] = str(args.threads)
	os.environ["MKL_MAX_THREADS"] = str(args.threads)
	os.environ["OMP_NUM_THREADS"] = str(args.threads)
	os.environ["OMP_MAX_THREADS"] = str(args.threads)
	os.environ["NUMEXPR_NUM_THREADS"] = str(args.threads)
	os.environ["NUMEXPR_MAX_THREADS"] = str(args.threads)
	os.environ["OPENBLAS_NUM_THREADS"] = str(args.threads)
	os.environ["OPENBLAS_MAX_THREADS"] = str(args.threads)

	# Import numerical libraries and cython functions
	import numpy as np
	from math import ceil
	from hapla import functions
	from hapla import admix_cy

	# Prepare list of data files
	if args.filelist is not None:
		W = 0 # Counter for windows
		Z_list = [] # List of filenames
		with open(args.filelist) as f:
			for z_file in f:
				# Check input across files (chromosomes) and count windows
				z = z_file.strip("\n")
				Z_list.append(z)
				assert os.path.isfile(f"{z}.bca"), "bca file doesn't exist!"
				assert os.path.isfile(f"{z}.ids"), "ids file doesn't exist!"
				assert os.path.isfile(f"{z}.win"), "win file doesn't exist!"
				if W == 0: # First file
					z_ids = np.genfromtxt(f"{z}.ids", dtype=np.str_)
					k_vec = np.genfromtxt(f"{z}.win", dtype=np.uint8, usecols=[5])
					N = z_ids.shape[0]
					W = k_vec.shape[0]
					w_list = [W]
				else: # Loop files
					t_ids = np.genfromtxt(f"{z}.ids", dtype=np.str_)
					assert np.sum(z_ids != t_ids) == 0, "Samples don't match across files!"
					k_tmp = np.genfromtxt(f"{z}.win", dtype=np.uint8, usecols=[5])
					k_vec = np.append(k_vec, k_tmp)
					W += k_tmp.shape[0]
					w_list.append(k_tmp.shape[0])
		w_vec = np.array(w_list, dtype=np.uint32)
		del z_ids, t_ids, k_tmp, w_list
	else: # Single file (chromosome)
		Z_list = [args.clusters]
		assert os.path.isfile(f"{Z_list[0]}.bca"), "bca file doesn't exist!"
		assert os.path.isfile(f"{Z_list[0]}.ids"), "ids file doesn't exist!"
		assert os.path.isfile(f"{Z_list[0]}.win"), "win file doesn't exist!"
		k_vec = np.genfromtxt(f"{Z_list[0]}.win", dtype=np.uint8, usecols=[5])
		N = np.genfromtxt(f"{Z_list[0]}.ids", dtype=np.str_).shape[0]
		W = k_vec.shape[0]
		w_vec = np.array([W], dtype=np.uint32)
	c_vec = np.insert(np.cumsum(k_vec[:-1]*args.K, dtype=np.uint32), 0, 0)
	print(f"Parsing {len(Z_list)} file(s).")

	# Load haplotype cluster assignments from binary hapla format
	B = 0
	Z = np.zeros((W, 2*N), dtype=np.uint8)
	for z in np.arange(len(Z_list)):
		with open(f"{Z_list[z]}.bca", "rb") as f:
			# Check magic numbers
			magic = np.fromfile(f, dtype=np.uint8, count=3)
			assert np.allclose(magic, np.array([7, 9, 13], dtype=np.uint8)), "Magic number doesn't match file format!"
			
			# Add haplotype cluster assignments to container
			z_tmp = np.fromfile(f, dtype=np.uint8)
			z_tmp.shape = (w_vec[z], 2*N)
			Z[B:(B + w_vec[z]),:] = z_tmp
			B += w_vec[z]
		print(f"\rParsed file {z+1}/{len(Z_list)}", end="")
	del magic, z_tmp

	# Count haplotype cluster alleles
	M = np.sum(k_vec, dtype=np.uint32)

	# Print information
	print("\rLoaded haplotype cluster assignments:\n" + \
		f"- {N} samples\n" + \
		f"- {W} windows\n" + \
		f"- {M} clusters\n")
	print(f"Estimating ancestry proportions: K={args.K}, seed={args.seed}.")

	# Initialize parameters randomly
	rng = np.random.default_rng(args.seed)
	P = rng.random(size=(M*args.K)).clip(min=1e-5, max=1-(1e-5))
	Q = rng.random(size=(2*N, args.K)).clip(min=1e-5, max=1-(1e-5))
	Q /= np.sum(Q, axis=1, keepdims=True)
	admix_cy.createP(P, k_vec, c_vec, args.K)

	# Supervised setting
	if args.supervised is not None:
		print("Ancestry estimation in supervised mode!")
		y = np.genfromtxt(args.supervised, dtype=np.uint8).reshape(-1)
		assert y.shape[0] == N, f"Number of samples differ between files!"
		assert np.max(y) <= args.K, "Wrong number of ancestral sources!"
		assert np.min(y) >= 0, "Wrong format in population assignments!"
		print(f"{np.sum(y > 0)}/{N} samples with fixed ancestry.")
		y = np.repeat(y, 2) # Repeat the vector easy computations
		admix_cy.superQ(Q, y)
	else:
		y = None
	
	# Set up containers for EM algorithm
	P1 = np.zeros_like(P)
	P2 = np.zeros_like(P)
	Q1 = np.zeros_like(Q)
	Q2 = np.zeros_like(Q)
	P_tmp = np.zeros_like(P)
	Q_tmp = np.zeros_like(Q)

	# Estimate initial log-likelihood
	L_pre = admix_cy.loglike(Z, P, Q, k_vec, c_vec)
	print(f"Initial log-like: {L_pre:.1f}\n")

	# Prime iterations
	functions.steps(Z, P, Q, P_tmp, Q_tmp, k_vec, c_vec, y)
	functions.quasi(Z, P, Q, P_tmp, Q_tmp, P1, P2, Q1, Q2, k_vec, c_vec, y)
	functions.steps(Z, P, Q, P_tmp, Q_tmp, k_vec, c_vec, y)

	# Set up mini-batch parameters
	if args.batches > 1:
		L_bat = L_pre
		s_win = np.arange(W, dtype=np.uint32)
		W_bat = ceil(W/args.batches)
		print("Mini-batch accelerated EM algorithm.")
	else:
		print(f"Accelerated EM algorithm.")

	# Accelerated EM algorithm
	ts = time()
	for it in range(args.iter):
		if args.batches > 1: # Quasi-Newton mini-batch updates
			rng.shuffle(s_win) # Shuffle window order
			for b in np.arange(args.batches):
				s_bat = s_win[(b*W_bat):min((b+1)*W_bat, M)]
				functions.batQuasi(Z, P, Q, P_tmp, Q_tmp, P1, P2, Q1, Q2, k_vec, c_vec, s_bat, y)
			functions.quasi(Z, P, Q, P_tmp, Q_tmp, P1, P2, Q1, Q2, k_vec, c_vec, y)
		else: # Full updates
			functions.quasi(Z, P, Q, P_tmp, Q_tmp, P1, P2, Q1, Q2, k_vec, c_vec, y)
			functions.steps(Z, P, Q, P_tmp, Q_tmp, k_vec, c_vec, y)

		# Log-likelihood convergence check
		if ((it+1) % args.check) == 0:
			L_cur = admix_cy.loglike(Z, P, Q, k_vec, c_vec)
			print(f"({it+1})\tLog-like: {L_cur:.1f}\t({time()-ts:.1f}s)", flush=True)
			if args.batches > 1:
				if (L_cur < L_bat) or (abs(L_cur - L_bat) < args.tole):
					args.batches = args.batches//2 # Halve number of batches
					if args.batches > 1:
						print(f"Halving mini-batches to {args.batches}.")
						L_bat = float('-inf')
						W_bat = ceil(W/args.batches)
					else: # Turn off mini-batch acceleration
						print("Running standard updates.")
					L_pre = L_cur
				else:
					L_bat = L_cur
			else: # Check for convergence
				if (abs(L_cur - L_pre) < args.tole):
					print("Converged!")
					print(f"Final log-likelihood: {L_cur:.1f}")
					break
				L_pre = L_cur
			ts = time()

	# Save output
	Q_fin = np.zeros((N, args.K)) # Final ancestry proportions
	admix_cy.convertQ(Q, Q_fin) # Average contribution from two haplotypes
	np.savetxt(f"{args.out}.K{args.K}.s{args.seed}.Q", Q_fin, fmt="%.6f")
	print(f"Saved Q matrix as {args.out}.K{args.K}.s{args.seed}.Q")
	if not args.no_freq:
		if args.filelist is not None: # Save P file for each file (chromosome)
			w_cnt = c_cnt = 0
			for z in np.arange(len(Z_list)):
				p_num = np.sum(k_vec[w_cnt:(w_cnt + w_vec[z])], dtype=np.uint32)*args.K
				P[c_cnt:(c_cnt + p_num)].tofile(
					f"{args.out}.K{args.K}.s{args.seed}.{args.prefix}{z+1}.P.bin"
				)
				w_cnt += w_vec[z]
				c_cnt += p_num
			print("Saved P matrices (binary) as " + \
		 		f"{args.out}.K{args.K}.s{args.seed}.{args.prefix}{{1..{len(Z_list)}}}.P.bin")
		else: # Single file (chromosome)
			P.tofile(f"{args.out}.K{args.K}.s{args.seed}.P.bin")
			print(f"Saved P matrix (binary) as {args.out}.K{args.K}.s{args.seed}.P.bin")

	# Print elapsed time for computation
	t_tot = time() - start
	t_min = int(t_tot//60)
	t_sec = int(t_tot - t_min*60)
	print(f"\nTotal elapsed time: {t_min}m{t_sec}s")

	# Write to log-file
	with open(f"{args.out}.K{args.K}.s{args.seed}.log", "a") as log:
		log.write(f"\nFinal log-likelihood: {L_cur:.1f}\n")
		log.write(f"Converged in {it+1} iterations.\n")
		log.write(f"\nSaved Q matrix as {args.out}.K{args.K}.s{args.seed}.Q\n")
		if not args.no_freq:
			if args.filelist is not None:
				log.write("Saved P matrices (binary) as " + \
			  		f"{args.out}.K{args.K}.s{args.seed}.{args.prefix}{{1..{len(Z_list)}}}.P.bin\n")
			else:
				log.write(f"Saved P matrix (binary) as {args.out}.K{args.K}.s{args.seed}.P.bin\n")
		log.write(f"\nTotal elapsed time: {t_min}m{t_sec}s\n")



##### Main exception #####
assert __name__ != "__main__", "Please use the 'hapla admix' command!"
