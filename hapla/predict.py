"""
hapla.
Haplotype clustering using pre-estimated cluster medians.
"""

__author__ = "Jonas Meisner"

# Libraries
import os
import re
from datetime import datetime
from time import time
from hapla import __version__


##### hapla predict #####
def main(args, deaf):
	print("-----------------------------------")
	print(f"hapla by Jonas Meisner (v{__version__})")
	print(f"hapla predict using {args.threads} thread(s)")
	print("-----------------------------------\n")
	
	# Check input
	assert args.vcf is not None or args.bfile is not None, "Please provide genotype file (--bcf, --vcf, --bfile)!"
	if args.vcf is not None:
		assert os.path.isfile(f"{args.vcf}"), "VCF/BCF file doesn't exist!"
		assert os.path.isfile(f"{args.vcf}.csi") or os.path.isfile(f"{args.vcf}.tbi"), "VCF/BCF index doesn't exist!"
	else:
		assert os.path.isfile(f"{args.bfile}.bed"), "PLINK bed file doesn't exist!"
		assert os.path.isfile(f"{args.bfile}.bim"), "PLINK bim file doesn't exist!"
		assert os.path.isfile(f"{args.bfile}.fam"), "PLINK fam file doesn't exist!"
		unphased = True
	assert args.ref is not None, "Please provide pre-estimated reference haplotype cluster medians (--ref)!"
	assert os.path.isfile(f"{args.ref}.bcm"), "bcm file doesn't exist!"
	assert os.path.isfile(f"{args.ref}.win"), "win file doesn't exist!"
	assert os.path.isfile(f"{args.ref}.wix"), "wix file doesn't exist!"
	assert args.threads > 0, "Please select a valid number of threads!"
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
		log.write("hapla predict\n")
		log.write(f"Time: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")
		log.write(f"Directory: {os.getcwd()}\n")
		log.write("Options:\n")
		for key in full:
			if full[key] != deaf[key]:
				log.write(f"\t--{key}\n") if (type(full[key]) is bool) else log.write(f"\t--{key} {full[key]}\n")
	del full, deaf

	# Import numerical libraries and cython functions
	import numpy as np
	from cyvcf2 import VCF
	from math import ceil
	from hapla import reader_cy
	from hapla import memory_cy
	from hapla import shared_cy

	# Load genotype data
	if args.vcf is not None:
		print("\rLoading VCF/BCF file...", end="")
		v_file = VCF(args.vcf, threads=min(args.threads, 4))
		s_list = np.array(v_file.samples).reshape(-1, 1)
		M = v_file.num_records

		# Check phasing and set parameters
		first = next(v_file)
		chrom = str(first.CHROM) # Extract chromosome information
		V = first.genotype.array() # Extract genotype array
		unphased = not bool(V[0,2]) # Extract phasing information
		N = s_list.shape[0] if unphased else 2*s_list.shape[0]
		B = ceil(N/4)

		# Allocate arrays
		G = np.zeros((M, B), dtype=np.uint8) if args.memory else np.zeros((M, N), dtype=np.uint8)
		v_vec = np.zeros(M, dtype=np.uint32)
		v_vec[0] = first.POS

		# Read variants into matrix
		if not unphased: # Haplotypes
			memory_cy.predBit(G[0], V, N//2) if args.memory else reader_cy.predVar(G[0], V, N//2)
			for j, variant in enumerate(v_file):
				V = variant.genotype.array()
				memory_cy.predBit(G[j + 1], V, N//2) if args.memory else reader_cy.predVar(G[j + 1], V, N//2)
				v_vec[j + 1] = variant.POS
		else: # Genotypes
			memory_cy.genoBit(G[0], V, N) if args.memory else reader_cy.genoVar(G[0], V, N)
			for j, variant in enumerate(v_file):
				V = variant.genotype.array()
				memory_cy.genoBit(G[j + 1], V, N) if args.memory else reader_cy.genoVar(G[j + 1], V, N)
				v_vec[j + 1] = variant.POS
		del first, V, v_file
	else:
		print("\rLoading PLINK files..", end="")
		N = 0
		with open(f"{args.bfile}.fam", "r") as fam:
			for _ in fam:
				N += 1
		B = ceil(N/4)

		# Read .bed file
		with open(f"{args.bfile}.bed", "rb") as bed:
			D = np.fromfile(bed, dtype=np.uint8, offset=3)
		assert (D.shape[0] % B) == 0, "bim file doesn't match!"
		M = D.shape[0]//B
		D.shape = (M, B)

		# Expand genotypes into 8-bit array
		if not args.memory:
			G = np.zeros((M, N), dtype=np.uint8)
			reader_cy.readPlink(D, G)
			del D
		else:
			G = D

		# Read sample and variant names
		bim = np.genfromtxt(f"{args.bfile}.bim", dtype=np.str_, usecols=[0,3])
		v_vec = bim[:,1].astype(np.uint32)
		chrom = str(bim[-1,0])
		s_list = np.genfromtxt(f"{args.bfile}.fam", dtype=np.str_, usecols=[1]).reshape(-1, 1)
		del bim
	if unphased:
		print(f"\rLoaded unphased genotype data: {N} samples and {M} SNPs.")
		N *= 2
	else:
		print(f"\rLoaded phased genotype data: {N} haplotypes and {M} SNPs.")

	# Load window information from reference
	w_mat = np.genfromtxt(f"{args.ref}.win", dtype=np.str_, skip_header=1)
	assert re.search(r"(\d+)$", w_mat[0,0]).group(1) == re.search(r"(\d+)$", chrom).group(1), \
		"Chromosome number differ between files!"
	assert int(w_mat[0,1]) == v_vec[0], "Positions differ between files!"
	assert int(w_mat[-1,2]) == v_vec[-1], "Positions differ between files!"
	s_vec = w_mat[:,1].astype(np.uint32)
	b_vec = w_mat[:,4].astype(np.uint32)
	k_vec = w_mat[:,5].astype(np.uint32)
	W = k_vec.shape[0]

	# Haplotype cluster medians
	with open(f"{args.ref}.bcm", "rb") as f:
		# Check magic numbers
		magic = np.fromfile(f, dtype=np.uint8, count=3)
		assert np.allclose(magic, np.array([7, 9, 13], dtype=np.uint8)), "Magic number doesn't match file format!"
		R_arr = np.fromfile(f, dtype=np.uint8)
		
	# Load window setup files
	w_vec = np.genfromtxt(f"{args.ref}.wix", dtype=np.uint32)
	assert np.allclose(v_vec[w_vec], s_vec), "SNP set doesn't match!"
	del v_vec, magic

	# Containers
	Z = np.zeros((W, N), dtype=np.uint8) # Haplotype cluster alleles

	# Clustering
	B = 0
	K = 0
	print(f"Clustering {W} windows.")
	for w in np.arange(W):
		print(f"\rWindow {w + 1}/{W}", end="")
		b_win = int(b_vec[w])
		k_win = int(k_vec[w])

		# Load haplotype window
		R_mat = R_arr[B:(B + k_win*b_win)]
		R_mat.shape = (k_win, b_win)
		if not unphased: # Haplotypes
			X = np.zeros((N, R_mat.shape[1]), dtype=np.uint8)
			memory_cy.expandBit(G, X, w_vec[w]) if args.memory else reader_cy.convertWin(G, X, w_vec[w])
		else: # Genotypes
			X = np.zeros((N//2, R_mat.shape[1]), dtype=np.uint8)
			memory_cy.expandGeno(G, X, w_vec[w]) if args.memory else reader_cy.convertWin(G, X, w_vec[w])

		# Cluster assignment
		shared_cy.genoCluster(X, R_mat, Z[w]) if unphased else shared_cy.predictCluster(X, R_mat, Z[w])

		# Update counter
		B += k_win*b_win
		K += k_win
	del G, X, R_mat, R_arr, w_vec
	if "D" in locals():
		del D
	print(".\n")

	# Save hapla output and print info
	h_win = ["#CHROM", "START", "END", "LENGTH", "SIZE", "K"]
	with open(f"{args.out}.bca", "wb") as f:
		np.array([7, 9, 13], dtype=np.uint8).tofile(f) # Add magic numbers
		Z.tofile(f) # Save haplotype cluster assignments to binary file
	np.savetxt(f"{args.out}.ids", s_list, fmt="%s")
	np.savetxt(f"{args.out}.win", w_mat, fmt="%s", delimiter="\t", comments="", header="\t".join(h_win))
	print("\rSaved haplotype clusters in binary hapla format:\n" + \
		f"- {args.out}.bca\n" + \
		f"- {args.out}.ids\n" + \
		f"- {args.out}.win\n")

	# Save haplotype cluster assignments in binary PLINK format
	if args.plink:
		print("\rGenerating binary PLINK output.", end="")
		B = ceil(N/8)
		K_tot = np.sum(k_vec, dtype=int)
		P_mat = np.zeros((K_tot, 3), dtype=np.uint32)
		Z_bin = np.zeros((K_tot, B), dtype=np.uint8)
		c_vec = np.insert(np.cumsum(k_vec[:-1], dtype=np.uint32), 0, 0)
		reader_cy.convertPlink(Z, Z_bin, P_mat, k_vec, c_vec, b_vec)
		
		# Save .bed file including magic numbers
		with open(f"{args.out}.bed", "w") as bfile:
			np.array([108, 27, 1], dtype=np.uint8).tofile(bfile)
			Z_bin.tofile(bfile)
		del c_vec, b_vec, Z_bin, Z

		# Save .bim file
		tmp = np.array([f"{chrom}_W{w}_K{k}_B{l}" for w,k,l in P_mat])
		bim = np.hstack((
			np.array([chrom]).repeat(K_tot).reshape(-1, 1),
			tmp.reshape(-1, 1),
			np.zeros((K_tot, 1), dtype=np.uint32),
			s_vec.repeat(k_vec).reshape(-1, 1),
			np.array(["K"]).repeat(K_tot).reshape(-1, 1),
			np.zeros((K_tot, 1), dtype=np.uint32)
		))
		np.savetxt(f"{args.out}.bim", bim, fmt="%s", delimiter="\t")
		del k_vec, s_vec, bim, tmp, P_mat
		
		# Save .fam file
		if args.duplicate_fid:
			s_list = s_list.repeat(2, axis=1)
		else:
			s_list = np.hstack((np.zeros((N//2, 1), dtype=np.uint8), s_list))
		fam = np.hstack((
			s_list,
			np.zeros((N//2, 3), dtype=np.uint8),
			np.full((N//2, 1), -9, dtype=np.int8)
		))
		np.savetxt(f"{args.out}.fam", fam, fmt="%s", delimiter="\t")

		# Print info
		print("\rSaved haplotype cluster alleles in binary PLINK format:\n" + \
			f"- {args.out}.bed\n" + \
			f"- {args.out}.bim\n" + \
			f"- {args.out}.fam\n")
		del fam, s_list

	# Print elapsed time for computation
	t_tot = time() - start
	t_min = int(t_tot//60)
	t_sec = int(t_tot - t_min*60)
	print(f"Total elapsed time: {t_min}m{t_sec}s")

	# Write to log-file
	with open(f"{args.out}.log", "a") as log:
		log.write("\nSaved haplotype clusters in binary hapla format:\n" + \
			f"- {args.out}.bca\n" + \
			f"- {args.out}.ids\n" + \
			f"- {args.out}.win\n")
		if args.plink:
			log.write("\nSaved haplotype cluster alleles in binary PLINK format:\n" + \
				f"- {args.out}.bed\n" + \
				f"- {args.out}.bim\n" + \
				f"- {args.out}.fam\n")
		log.write(f"\nTotal elapsed time: {t_min}m{t_sec}s\n")



##### Main exception #####
assert __name__ != "__main__", "Please use the 'hapla predict' command!"
