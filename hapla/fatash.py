"""
hapla.
Infer local ancestry tracts from genome-wide ancestry estimates.
"""

__author__ = "Jonas Meisner"

# Libraries
import os
from datetime import datetime
from time import time
from hapla import __version__


##### hapla fatash #####
def main(args, deaf):
	print("-----------------------------------")
	print(f"hapla by Jonas Meisner (v{__version__})")
	print(f"hapla fatash using {args.threads} thread(s)")
	print("-----------------------------------\n")

	# Check input
	assert (args.filelist is not None) or (args.clusters is not None), "No input data (--filelist or --clusters)!"
	assert (args.pfilelist is not None) or (args.pfile is not None), "No P file(s) provided (--pfilelist or --pfile)!"
	if args.filelist is not None:
		assert args.pfilelist is not None, "Input formats don't match!"
	if args.clusters is not None:
		assert args.pfile is not None, "Input formats don't match!"
	assert args.qfile is not None, "No Q file provided (--qfile)!"
	assert args.threads > 0, "Please select a valid number of threads!"
	if args.alpha is not None:
		assert args.alpha > 0.0, "Please select a valid alpha value!"
	assert args.alpha_min < args.alpha_max, "Please select valid alpha exponents!"
	assert (args.alpha_max - args.alpha_min) % 2 == 0, "Please select an uneven number alpha exponents!"
	if not args.genome_wide:
		assert args.admix_seed >= 0, "Please select a valid seed!"
		assert args.admix_iter > 0, "Please select a valid number of iterations!"
		assert args.admix_tole >= 0.0, "Please select a valid tolerance!"
		assert args.admix_batches > 0, "Please select a valid number of mini-batches!"
		assert args.admix_check > 0, "Please select a valid value for convergence check!"
	mode = "Viterbi" if args.viterbi else "posterior"
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
	with open(f"{args.out}.log", "w") as log:
		log.write(f"hapla v{__version__}\n")
		log.write("hapla fatash\n")
		log.write(f"Time: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")
		log.write(f"Directory: {os.getcwd()}\n")
		log.write("Options:\n")
		for key in full:
			if full[key] != deaf[key]:
				log.write(f"\t--{key}\n") if (type(full[key]) is bool) else log.write(f"\t--{key} {full[key]}\n")
	del full, deaf

	# Import numerical libraries and cython functions
	import numpy as np
	from hapla import functions
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
	if args.genome_wide: # Use genome-wide ancestry proportions
		Q_chr = Q
		Q_log = np.log(Q)
	else: # Estimate file-specific ancestry proportions
		rng = np.random.default_rng(args.admix_seed)
		Q1 = np.zeros_like(Q)
		Q2 = np.zeros_like(Q)
		Q_tmp = np.zeros_like(Q)
	K = Q.shape[1]

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
	print(f"Inferring local ancestry tracts from {K} ancestral sources.")
	for z in np.arange(F): # Loop through files
		print(f"Processing file {z + 1}/{F}")
		t_chr = time()
		W_chr = w_vec[z]

		# Load P matrix file
		if F > 1:
			P_chr = np.genfromtxt(P_list[z], dtype=float).reshape(-1)
			k_chr = k_vec[w_cnt:(w_cnt + W_chr)]
			w_cnt += W_chr
		else:
			P_chr = np.genfromtxt(args.pfile, dtype=float).reshape(-1)
			k_chr = k_vec
		c_chr = np.insert(np.cumsum(k_chr*K, dtype=np.uint32), 0, 0)
		L_chr = np.sum(k_chr, dtype=int)
		assert P_chr.shape[0] == (L_chr*K), "Number of clusters doesn't match!"

		# Load haplotype cluster assignment file
		with open(f"{Z_list[z]}.bca", "rb") as f:
			# Check magic numbers
			magic = np.fromfile(f, dtype=np.uint8, count=3)
			assert np.allclose(magic, np.array([7, 9, 13], dtype=np.uint8)), "Magic number doesn't match file format!"
			
			# Add haplotype cluster assignments to container
			Z_chr = np.fromfile(f, dtype=np.uint8)
			Z_chr.shape = (W_chr, N)
		assert np.max(k_chr) == (np.max(Z_chr) + 1), "Number of clusters doesn't match!"

		# Estimate chromosome-specific ancestry proportions
		if not args.genome_wide:
			print("Estimating file-specific ancestry proportions.", end="", flush=True)
			ts = time()
			Q_chr = np.copy(Q)
			L_pre = fatash_cy.loglike(Z_chr, P_chr, Q_chr, c_chr, L_chr)

			# Set up mini-batch parameters
			L_bat = L_pre
			B_bat = args.admix_batches
			s_chr = np.arange(W_chr, dtype=np.uint32)
			W_bat = W_chr//B_bat

			# Priming iteration
			functions.laiSteps(Z_chr, P_chr, Q_chr, Q_tmp, c_chr)
			functions.laiQuasi(Z_chr, P_chr, Q_chr, Q_tmp, Q1, Q2, c_chr)
			functions.laiSteps(Z_chr, P_chr, Q_chr, Q_tmp, c_chr)

			# Accelerated EM algorithm (projection mode)
			for it in range(args.admix_iter):
				if B_bat > 1: # Quasi-Newton mini-batch updates
					rng.shuffle(s_chr) # Shuffle window order
					for b in np.arange(B_bat):
						s_beg = b*W_bat
						s_end = W_chr if b == (B_bat - 1) else (b + 1)*W_bat
						s_bat = s_chr[s_beg:s_end]
						functions.laiBatch(Z_chr, P_chr, Q_chr, Q_tmp, Q1, Q2, c_chr, s_bat)
					functions.laiQuasi(Z_chr, P_chr, Q_chr, Q_tmp, Q1, Q2, c_chr)
				else:
					functions.laiQuasi(Z_chr, P_chr, Q_chr, Q_tmp, Q1, Q2, c_chr)
					functions.laiSteps(Z_chr, P_chr, Q_chr, Q_tmp, c_chr)
			
				# Log-likelihood convergence check
				if ((it + 1) % args.admix_check) == 0:
					L_cur = fatash_cy.loglike(Z_chr, P_chr, Q_chr, c_chr, L_chr)
					if B_bat > 1:
						if L_cur < (L_bat + args.admix_tole):
							B_bat = B_bat//2 # Halve number of batches
							if B_bat > 1:
								L_bat = float('-inf')
								W_bat = W_chr//B_bat
							L_pre = L_cur
							functions.laiQuasi(Z_chr, P_chr, Q_chr, Q_tmp, Q1, Q2, c_chr)
						else:
							L_bat = L_cur
					else: # Check for convergence
						if L_cur < (L_pre + args.admix_tole):
							break
						L_pre = L_cur
			del s_chr, s_bat
			Q_log = np.log(Q_chr)
			print(f"\rEstimating file-specific ancestry proportions.\t\t({time() - ts:.1f}s)")

		# Load haplotype cluster medians log-likelihoods
		if args.medians:
			with open(f"{Z_list[z]}.blk", "rb") as f:
				# Check magic numbers
				magic = np.fromfile(f, dtype=np.uint8, count=3)
				assert np.allclose(magic, np.array([7, 9, 13], dtype=np.uint8)), \
					"Magic number doesn't match file format!"

				# Add haplotype cluster medians log-likelihoods to container
				M_chr = np.fromfile(f, dtype=np.float32)
		del magic

		# Compute emission probabilities
		Z_chr = np.ascontiguousarray(Z_chr.T) # Transpose for easier computations
		E_chr = np.zeros((N, W_chr, K)) # Emission probabilities
		if args.medians:
			# Normalize log-likelihoods
			x_chr = np.insert(np.cumsum(k_chr*k_chr, dtype=np.uint32), 0, 0)
			fatash_cy.createLikes(M_chr, k_chr, x_chr)
			fatash_cy.softEmissions(Z_chr, E_chr, P_chr, M_chr, k_chr, c_chr, x_chr)
			del M_chr, x_chr
		else:
			fatash_cy.hardEmissions(Z_chr, E_chr, P_chr, c_chr)
		del Z_chr, P_chr, k_chr, c_chr

		# Multithreaded HMM inference
		print(f"Performing {mode} decoding ", end="")
		if args.viterbi: # Compute Viterbi decoding
			D_ind = np.zeros((N, W_chr), dtype=np.uint8)
			if args.alpha is None:
				alpha = [10**(-a) for a in range(args.alpha_min, args.alpha_max + 1)]
				print(f"with majority voting across {len(alpha)} alphas.")
				D_bag = np.zeros((len(alpha), N, W_chr), dtype=np.uint8)
				for a in range(len(alpha)):
					fatash_cy.viterbi(E_chr, D_bag[a], Q_chr, Q_log, alpha[a])
				fatash_cy.voting(D_bag, D_ind, K)
				del D_bag
			else:
				print("with fixed alpha.")
				fatash_cy.viterbi(E_chr, D_ind, Q_chr, Q_log, args.alpha)
		else: # Compute posterior decoding
			L_ind = np.zeros((N, W_chr, K))
			if args.alpha is None:
				alpha = [10**(-a) for a in range(args.alpha_min, args.alpha_max + 1)]
				print(f"with majority voting across {len(alpha)} alphas.")
				D_ind = np.zeros((N, W_chr), dtype=np.uint8)
				D_bag = np.zeros((len(alpha), N, W_chr), dtype=np.uint8)
				for a in range(len(alpha)):
					fatash_cy.fwdbwd(E_chr, L_ind, Q_chr, Q_log, alpha[a])
					D_bag[a] = L_ind.argmax(axis=2).astype(np.uint8)
				fatash_cy.voting(D_bag, D_ind, K)
				del D_bag
			else:
				print("with fixed alpha.")
				fatash_cy.fwdbwd(E_chr, L_ind, Q_chr, Q_log, args.alpha)
				D_ind = L_ind.argmax(axis=2).astype(np.uint8)
			del L_ind
		del E_chr

		# Save matrices
		if F > 1:
			np.savetxt(f"{args.out}.{args.prefix}{z + 1}.path", D_ind, fmt="%i")
			print(f"Saved {mode} decoding path as {args.out}.{args.prefix}{z + 1}.path")

			# Print elapsed time of chromosome 
			t_tmp = time() - t_chr
			t_min = int(t_tmp//60)
			t_sec = int(t_tmp - t_min*60)
			print(f"Elapsed time: {t_min}m{t_sec}s\n")
		else:
			np.savetxt(f"{args.out}.path", D_ind, fmt="%i")
			print(f"Saved {mode} decoding path as {args.out}.path\n")

		# Extra clean up
		if not args.genome_wide:
			del Q_chr, Q_log
		del D_ind

	# Print elapsed time for computation
	t_tot = time() - start
	t_min = int(t_tot//60)
	t_sec = int(t_tot - t_min*60)
	print(f"Total elapsed time: {t_min}m{t_sec}s")

	# Write to log-file
	with open(f"{args.out}.log", "a") as log:
		if F > 1:
			log.write(f"\nSaved {mode} decoding path as {args.out}.{args.prefix}{{1..{len(Z_list)}}}.path\n")
		else:
			log.write(f"\nSaved {mode} decoding path as {args.out}.path\n")
		log.write(f"\nTotal elapsed time: {t_min}m{t_sec}s\n")



##### Main exception #####
assert __name__ != "__main__", "Please use the 'hapla fatash' command!"
