"""
hapla.
Infer local ancestry tracts using a hidden Markov model.
"""

__author__ = "Jonas Meisner"

# Libraries
import os
from datetime import datetime
from time import time

VERSION = "0.32.1"

##### hapla fatash #####
def main(args, deaf):
	print("-----------------------------------")
	print(f"hapla by Jonas Meisner (v{VERSION})")
	print(f"hapla fatash using {args.threads} thread(s)")
	print("-----------------------------------\n")

	# Check input
	assert (args.filelist is not None) or (args.clusters is not None), "No input data (--filelist or --clusters)!"
	assert (args.pfilelist is not None) or (args.pfile is not None), "No P files provided (--pfilelist or --pfile)!"
	if args.filelist is not None:
		assert args.pfilelist is not None, "Input formats don't match!"
	if args.clusters is not None:
		assert args.pfile is not None, "Input formats don't match!"
	assert args.qfile is not None, "No Q file provided (--qfile)!"
	assert args.threads > 0, "Please select a valid number of threads!"
	assert args.alpha > 0.0, "Please select a valid alpha value!"
	start = time()

	# Control threads of external numerical libraries
	os.environ["MKL_NUM_THREADS"] = str(args.threads)
	os.environ["MKL_MAX_THREADS"] = str(args.threads)
	os.environ["OMP_NUM_THREADS"] = str(args.threads)
	os.environ["OMP_MAX_THREADS"] = str(args.threads)
	os.environ["NUMEXPR_NUM_THREADS"] = str(args.threads)
	os.environ["NUMEXPR_MAX_THREADS"] = str(args.threads)
	os.environ["OPENBLAS_NUM_THREADS"] = str(args.threads)
	os.environ["OPENBLAS_MAX_THREADS"] = str(args.threads)

	# Create log-file of used arguments
	full = vars(args)
	mand = ["alpha"]
	with open(f"{args.out}.log", "w") as log:
		log.write(f"hapla v{VERSION}\n")
		log.write("hapla fatash\n")
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

	# Import numerical libraries and cython functions
	import numpy as np
	from hapla import fatash_cy

	# Prepare list of data files
	if args.filelist is not None:
		Z_list = [] # List of filenames
		with open(args.filelist) as f:
			for z_file in f:
				# Check input across files and count windows
				z = z_file.strip("\n")
				Z_list.append(z)
				assert os.path.isfile(f"{z}.bca"), "bca file doesn't exist!"
				assert os.path.isfile(f"{z}.ids"), "ids file doesn't exist!"
				assert os.path.isfile(f"{z}.win"), "win file doesn't exist!"
				if args.medians:
					assert os.path.isfile(f"{z}.blk"), "blk file doesn't exist!"
				if len(Z_list) == 1: # First file
					z_ids = np.genfromtxt(f"{z}.ids", dtype=np.str_)
					w_tmp = np.genfromtxt(f"{z}.win", dtype=int, usecols=[1,2,5])
					k_vec = w_tmp[:,2].astype(np.uint32)
					N = 2*z_ids.shape[0]
					w_list = [k_vec.shape[0]]
				else: # Loop files
					t_ids = np.genfromtxt(f"{z}.ids", dtype=np.str_)
					assert np.sum(z_ids != t_ids) == 0, "Samples do not match across files!"
					w_tmp = np.genfromtxt(f"{z}.win", dtype=int, usecols=[1,2,5])
					k_vec = np.append(k_vec, w_tmp[:,2].astype(np.uint32))
					w_list.append(w_tmp.shape[0])
		F = len(Z_list)
		w_vec = np.array(w_list, dtype=np.uint32)
		del z_ids, t_ids, w_tmp, w_list
	else: # Single file (chromosome)
		F = 1
		Z_list = [args.clusters]
		assert os.path.isfile(f"{Z_list[0]}.bca"), "bca file doesn't exist!"
		assert os.path.isfile(f"{Z_list[0]}.ids"), "ids file doesn't exist!"
		assert os.path.isfile(f"{Z_list[0]}.win"), "win file doesn't exist!"
		if args.medians:
			assert os.path.isfile(f"{Z_list[0]}.blk"), "blk file doesn't exist!"
		w_tmp = np.genfromtxt(f"{Z_list[0]}.win", dtype=int, usecols=[1,2,5])
		k_vec = w_tmp[:,2].astype(np.uint32)
		w_vec = np.array([k_vec.shape[0]], dtype=np.uint32)
		N = 2*np.genfromtxt(f"{Z_list[0]}.ids", dtype=np.str_).shape[0]
		del w_tmp
	print(f"Parsing {F} file(s).")

	# Load Q matrix
	Q = np.genfromtxt(args.qfile, dtype=float)
	Q = np.repeat(Q, 2, axis=0)
	assert Q.shape[0] == N, "Number of samples doesn't match!"
	Q_log = np.log(Q)
	K = Q.shape[1]
	v = np.zeros(K)

	# Prepare list of P files
	if F > 1:
		w_cnt = 0
		P_list = []
		with open(args.pfilelist) as f:
			for p, p_file in enumerate(f):
				assert os.path.isfile(p_file.strip("\n")), f"The {p + 1}/{F} matrix file doesn't exist!"
				P_list.append(p_file.strip("\n"))
		assert len(P_list) == F, "Number of files doesn't match!"

	# Loop over chromosomes
	print(f"Inferring local ancestry tracts with {K} ancestral sources.\n")
	for z in np.arange(F): # Loop through files
		print(f"Processing file {z + 1}/{F}")
		s_tmp = time()

		# Load P matrix file
		if F > 1:
			P_tmp = np.fromfile(P_list[z], dtype=float)
			k_tmp = k_vec[w_cnt:(w_cnt + w_vec[z])]
			w_cnt += w_vec[z]
		else:
			P_tmp = np.fromfile(args.pfile, dtype=float)
			k_tmp = k_vec
		c_tmp = np.insert(np.cumsum(k_tmp[:-1]*K, dtype=np.uint32), 0, 0)
		p_num = np.sum(k_tmp, dtype=int)*K
		assert P_tmp.shape[0] == p_num, "Number of clusters doesn't match!"

		# Load haplotype cluster assignment file
		with open(f"{Z_list[z]}.bca", "rb") as f:
			# Check magic numbers
			magic = np.fromfile(f, dtype=np.uint8, count=3)
			assert np.allclose(magic, np.array([7, 9, 13], dtype=np.uint8)), "Magic number doesn't match file format!"
			
			# Add haplotype cluster assignments to container
			Z_tmp = np.fromfile(f, dtype=np.uint8)
			Z_tmp.shape = (w_vec[z], N)
		Z_tmp = np.ascontiguousarray(Z_tmp.T) # Transpose for easier computations
		assert np.max(k_tmp) == (np.max(Z_tmp) + 1), "Number of clusters doesn't match!"

		# Load haplotype cluster medians log-likelihoods
		if args.medians:
			with open(f"{Z_list[z]}.blk", "rb") as f:
				# Check magic numbers
				magic = np.fromfile(f, dtype=np.uint8, count=3)
				assert np.allclose(magic, np.array([7, 9, 13], dtype=np.uint8)), \
					"Magic number doesn't match file format!"

				# Add haplotype cluster medians log-likelihoods to container
				L_tmp = np.fromfile(f, dtype=np.float32)
		del magic

		# Containers
		E = np.zeros((N, w_vec[z], K)) # Emission probabilities
		T = np.zeros((K, K)) # Transition probabilities
		A = np.zeros((w_vec[z], K)) # Forward matrix
		if args.viterbi:
			I = np.zeros((w_vec[z], K), dtype=np.uint8) # Index matrix
			V = np.zeros((N, w_vec[z]), dtype=np.uint8) # Viterbi path
		else:
			B = np.zeros((w_vec[z], K)) # Backward matrix
			L = np.zeros((N, w_vec[z], K)) # Posterior probabilities

		# Compute emission probabilities and distance vectors
		if args.medians:
			# Normalize log-likelihoods
			x_tmp = np.insert(np.cumsum(k_tmp[:-1]*k_tmp[:-1], dtype=np.uint32), 0, 0)
			fatash_cy.createLikes(L_tmp, k_tmp, x_tmp)
			fatash_cy.softEmissions(Z_tmp, E, P_tmp, L_tmp, k_tmp, c_tmp, x_tmp)
			del x_tmp
		else:
			fatash_cy.hardEmissions(Z_tmp, E, P_tmp, c_tmp)
		del Z_tmp, P_tmp, k_tmp, c_tmp

		# HMM for each haplotype
		for i in np.arange(N):
			print(f"\rHaplotype {i+1}/{N}", end="")
			fatash_cy.calcTransition(T, Q[i], args.alpha)
			if args.viterbi: # Compute Viterbi and decoding
				fatash_cy.viterbi(E[i], Q_log[i], T, A, I, V[i])
			else: # Compute posterior probabilities
				fatash_cy.calcFwdBwd(E[i], L[i], Q_log[i], T, A, B, v)
		print(".")

		# Save matrices
		if F > 1:
			if args.viterbi:
				np.savetxt(f"{args.out}.{args.prefix}{z + 1}.path", V, fmt="%i")
				print(f"Saved Viterbi decoding as {args.out}.{args.prefix}{z + 1}.path")
			else:
				np.savetxt(f"{args.out}.{args.prefix}{z + 1}.path", L.argmax(axis=2), fmt="%i")
				print(f"Saved posterior decoding as {args.out}.{args.prefix}{z + 1}.path")
				if args.save_posterior:
					L.tofile(f"{args.out}.{args.prefix}{z + 1}.post.bin")
					print(f"Saved posteriors (binary) as {args.out}.{args.prefix}{z + 1}.post.bin")

			# Print elapsed time of chromosome 
			t_tmp = time() - s_tmp
			t_min = int(t_tmp//60)
			t_sec = int(t_tmp - t_min*60)
			print(f"Elapsed time: {t_min}m{t_sec}s\n")
		else:
			if args.viterbi:
				np.savetxt(f"{args.out}.path", V, fmt="%i")
				print(f"Saved Viterbi decoding path as {args.out}.path")
			else:
				np.savetxt(f"{args.out}.path", L.argmax(axis=2), fmt="%i")
				print(f"Saved posterior decoding path as {args.out}.path")
				if args.save_posterior:
					np.save(f"{args.out}.post", L)
					print(f"Saved posteriors as {args.out}.post.npy")
			print("")
		del E, T, A
		if args.viterbi:
			del I, V
		else:
			del B, L

	# Print elapsed time for computation
	t_tot = time() - start
	t_min = int(t_tot//60)
	t_sec = int(t_tot - t_min*60)
	print(f"Total elapsed time: {t_min}m{t_sec}s")

	# Write to log-file
	with open(f"{args.out}.log", "a") as log:
		if F > 1:
			if args.viterbi:
				log.write(f"\nSaved Viterbi decoding path as {args.out}.{args.prefix}{{1..{len(Z_list)}}}.path\n")
			else:
				log.write(f"\nSaved posterior decoding path as {args.out}.{args.prefix}{{1..{len(Z_list)}}}.path\n")
				if args.save_posterior:
					log.write(f"Saved posteriors as {args.out}.{args.prefix}{{1..{len(Z_list)}}}.post.npy\n")
		else:
			if args.viterbi:
				log.write(f"\nSaved Viterbi decoding path as {args.out}.path\n")
			else:
				log.write(f"\nSaved posterior decoding path as {args.out}.path\n")
				if args.save_posterior:
					log.write(f"Saved posteriors as {args.out}.post.npy")
		log.write(f"\nTotal elapsed time: {t_min}m{t_sec}s\n")



##### Main exception #####
assert __name__ != "__main__", "Please use the 'hapla fatash' command!"
