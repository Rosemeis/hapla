"""
hapla.
Haplotype clustering using pre-estimated cluster medians.
"""

__author__ = "Jonas Meisner"

# Libraries
import os

##### hapla predict #####
def main(args):
	print("---------------------------------")
	print("hapla by Jonas Meisner (v0.3)")
	print(f"hapla predict using {args.threads} thread(s)")
	print("---------------------------------\n")
	
	# Check input
	assert args.vcf is not None, \
		"Please provide phased genotype file (--bcf or --vcf)!"
	assert args.medians is not None, \
		"Please provide pre-estimated haplotype cluster medians (--medians)!"

	# Control threads of external numerical libraries
	os.environ["MKL_NUM_THREADS"] = str(args.threads)
	os.environ["OMP_NUM_THREADS"] = str(args.threads)
	os.environ["NUMEXPR_NUM_THREADS"] = str(args.threads)
	os.environ["OPENBLAS_NUM_THREADS"] = str(args.threads)

	# Import numerical libraries and cython functions
	import numpy as np
	from cyvcf2 import VCF
	from math import ceil
	from hapla import reader_cy
	from hapla import shared_cy

	### Load data into 1-bit matrix
	print("\rLoading VCF/BCF file...", end="")
	v_file = VCF(args.vcf, threads=args.threads)
	n = 2*len(v_file.samples)
	B = ceil(n/4)
	G = reader_cy.readVCF(v_file, n//2, B)
	del v_file
	m = G.shape[0]
	print(f"\rLoaded phased genotype data: {n} haplotypes and {m} SNPs.")

	### Setup windows	
	if args.windows is None: # Fixed window length
		W = m//args.fixed
		winList = [w*args.fixed for w in range(W)]
		winList = np.array(winList, dtype=int)
		print(f"Clustering in {W} windows of fixed size ({args.fixed} SNPs).")
	else: # Use provided window lengths
		winList = np.genfromtxt(args.windows, dtype=int)
		W = winList.shape[0] - 1
		assert winList[-1] == m, "Window splits doesn't match genotype file!"
		print(f"Clustering in {W} windows of provided lengths.")
	
		# Filter out causal SNPs --- DEBUG FOR SIMULATION STUDIES ONLY!
		if args.filter is not None:
			mask = np.loadtxt(args.filter, dtype=np.uint8)
			m = np.sum(mask) # New number of variants
			reader_cy.filterSNPs(G, winList, mask) # Fix data and window arrays
			G = G[:m,:]
			print(f"Removed {np.sum(mask==0)} causal SNPs.")
			del mask
	
	# Haplotype cluster medians
	M_mat = np.load(args.medians)
	assert m == M_mat.shape[0], "Number of SNPs does not match between files!"

	# Containers
	Z_mat = np.zeros((W, n), dtype=np.uint8) # Haplotype cluster alleles

	### Clustering
	for w in np.arange(W):
		# Load haplotype segment
		if w < (W-1):
			if args.windows is None:
				H = np.zeros((args.fixed, n), dtype=np.uint8)
				M = np.ascontiguousarray(M_mat[winList[w]:(winList[w]+args.fixed)].T)
			else:
				H = np.zeros((winList[w+1]-winList[w], n), dtype=np.uint8)
				M = np.ascontiguousarray(M_mat[winList[w]:winList[w+1]].T)
		else:
			H = np.zeros((m-winList[w], n), dtype=np.uint8)
			M = np.ascontiguousarray(M_mat[winList[w]:m].T)
		K = np.sum(np.sum(M, axis=1, dtype=int) >= 0) # Number of clusters to evaluate
		reader_cy.predictBit(G, H, winList[w])
		Ht = np.ascontiguousarray(H.T)
		del H
		
		# Cluster assignment
		shared_cy.predictCluster(Ht, M, Z_mat, K, w, args.threads)
		del Ht, M

	### Save output
	np.save(f"{args.out}.z", Z_mat)
	print(f"Saved predicted haplotype cluster alleles as {args.out}.z.npy")



##### Main exception #####
assert __name__ != "__main__", "Please use the 'hapla predict' command!"
