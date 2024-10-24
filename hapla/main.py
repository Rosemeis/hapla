"""
Main caller of hapla.
"""

__author__ = "Jonas Meisner"

# Libraries
import argparse
import sys

# Main function
def main():
	# Argparser
	parser = argparse.ArgumentParser(prog="hapla")
	subparsers = parser.add_subparsers(title="hapla commands")

	### Commands
	# hapla cluster
	parser_cluster = subparsers.add_parser("cluster")
	parser_cluster.add_argument("-g", "--vcf", "--bcf", metavar="FILE",
		help="Input phased genotype file in VCF/BCF format")
	parser_cluster.add_argument("-f", "--fixed", type=int,
		metavar="INT", help="Use fixed window size")
	parser_cluster.add_argument("-w", "--windows", metavar="FILE",
		help="Use provided window indices")
	parser_cluster.add_argument("-l", "--lmbda", type=float, default=0.1,
		metavar="FLOAT", help="Set lambda hyperparameter (0.1)")
	parser_cluster.add_argument("-s", "--step", type=int,
		metavar="INT", help="Step-size for sliding window")
	parser_cluster.add_argument("-t", "--threads", type=int, default=1,
		metavar="INT", help="Number of threads (1)")
	parser_cluster.add_argument("-o", "--out", default="hapla.cluster",
		metavar="OUTPUT", help="Output prefix")
	parser_cluster.add_argument("--min-freq", type=float, default=0.01,
		metavar="INT", help="Minimum haplotype cluster frequency (0.01)")
	parser_cluster.add_argument("--max-clusters", type=int, default=256,
		metavar="INT", help="Maximum number of haplotype clusters per window (256)")
	parser_cluster.add_argument("--max-iterations", type=int, default=500,
		metavar="INT", help="Maximum number of iterations (500)")
	parser_cluster.add_argument("--medians", action="store_true",
		help="Save haplotype cluster medians")
	parser_cluster.add_argument("--plink", action="store_true",
		help="Generate binary PLINK output")
	parser_cluster.add_argument("--duplicate-fid", action="store_true",
		help="Use sample list as family ID (PLINK 1.9 compatibility)")
	parser_cluster.add_argument("--memory", action="store_true",
		help="Store haplotypes in 1-bit matrix")

	# hapla struct
	parser_struct = subparsers.add_parser("struct")
	parser_struct.add_argument("-f", "--filelist", metavar="FILE",
		help="Filelist with paths to haplotype cluster alleles files")
	parser_struct.add_argument("-z", "--clusters", metavar="FILE",
		help="Path to a single haplotype cluster alleles file")
	parser_struct.add_argument("-t", "--threads", type=int, default=1,
		metavar="INT", help="Number of threads (1)")
	parser_struct.add_argument("-o", "--out", default="hapla.pca",
		metavar="OUTPUT", help="Output prefix")
	parser_struct.add_argument("--grm", action="store_true",
		help="Estimate genome-wide relationship matrix (GRM)")
	parser_struct.add_argument("--batch", type=int, default=4096,
		metavar="INT", help="Number of cluster alleles in batches (4096)")
	parser_struct.add_argument("--no-centering", action="store_true",
		help="Do not perform Gower and data centering of GRM")
	parser_struct.add_argument("--duplicate-fid", action="store_true",
		help="Use sample list as family ID for GCTA format")
	parser_struct.add_argument("--pca", type=int,
		metavar="INT", help="Perform PCA and extract eigenvectors")
	parser_struct.add_argument("--loadings", action="store_true",
		help="Save loadings of SVD")
	parser_struct.add_argument("--randomized", action="store_true",
		help="Use randomized SVD (for very large sample sizes)")
	parser_struct.add_argument("--raw", action="store_true",
		help="Raw output without '*.fam' info")

	# hapla predict
	parser_predict = subparsers.add_parser("predict")
	parser_predict.add_argument("-g", "--vcf", "--bcf", metavar="FILE",
		help="Input phased genotype file in VCF/BCF format")
	parser_predict.add_argument("-r", "--ref", metavar="FILE",
		help="Input reference prefix of pre-estimated cluster medians")
	parser_predict.add_argument("-t", "--threads", type=int, default=1,
		metavar="INT", help="Number of threads (1)")
	parser_predict.add_argument("-o", "--out", default="hapla.predict",
		metavar="OUTPUT", help="Output prefix")
	parser_predict.add_argument("--plink", action="store_true",
		help="Generate binary PLINK output")
	parser_predict.add_argument("--duplicate-fid", action="store_true",
		help="Use sample list as family ID (PLINK 1.9 compatibility)")
	parser_predict.add_argument("--memory", action="store_true",
		help="Store haplotypes in 2-bit matrix")

	# hapla admix
	parser_admix = subparsers.add_parser("admix")
	parser_admix.add_argument("-f", "--filelist", metavar="FILE",
		help="Filelist with paths to haplotype cluster alleles files")
	parser_admix.add_argument("-z", "--clusters", metavar="FILE",
		help="Path to a single haplotype cluster alleles file")
	parser_admix.add_argument("-k", "--K", type=int,
		metavar="INT", help="Number of ancestral components")
	parser_admix.add_argument("-t", "--threads", type=int, default=1,
		metavar="INT", help="Number of threads (1)")
	parser_admix.add_argument("-o", "--out", default="hapla.admix",
		metavar="OUTPUT", help="Output prefix")
	parser_admix.add_argument("--seed", type=int, default=42,
		metavar="INT", help="Random seed (42)")
	parser_admix.add_argument("--iter", type=int, default=1000,
		metavar="INT", help="Maximum number of iterations (1000)")
	parser_admix.add_argument("--tole", type=float, default=1.0,
		metavar="FLOAT", help="Tolerance in log-likelihood between iterations (1.0)")
	parser_admix.add_argument("--check", type=int, default=10,
		metavar="INT", help="Check for convergence every c-th iteration (10)")
	parser_admix.add_argument("--haplotype", action="store_true",
		help="Estimate admixture proportions for each haplotype")
	parser_admix.add_argument("--supervised", metavar="FILE",
		help="Path to population assignment file")
	parser_admix.add_argument("--no-freq", action="store_true",
		help="Do not save haplotype cluster frequencies")

	# hapla fatash
	parser_fatash = subparsers.add_parser("fatash")
	parser_fatash.add_argument("-f", "--filelist", metavar="FILE",
		help="Filelist with paths to haplotype cluster alleles files")
	parser_fatash.add_argument("-z", "--clusters", metavar="FILE",
		help="Path to a single haplotype cluster alleles file")
	parser_fatash.add_argument("-e", "--pfilelist", metavar="FILE",
		help="Filelist with paths to haplotype cluster frequencies files")
	parser_fatash.add_argument("-p", "--pfile", metavar="FILE",
		help="Path to file with haplotype cluster frequencies")
	parser_fatash.add_argument("-q", "--qfile", metavar="FILE",
		help="Path to file with admixture proportions")
	parser_fatash.add_argument("-t", "--threads", type=int, default=1,
		metavar="INT", help="Number of threads (1)")
	parser_fatash.add_argument("-o", "--out", default="hapla.fatash",
		metavar="OUTPUT", help="Output prefix")
	parser_fatash.add_argument("--alpha", type=float, default=1.0,
		metavar="FLOAT", help="Set fixed alpha rate (1.0)")
	parser_fatash.add_argument("--viterbi", action="store_true",
		help="Perform Viterbi decoding")
	parser_fatash.add_argument("--save-posterior", action="store_true",
		help="Save posterior probabilities")

	# Parse arguments
	args = parser.parse_args()
	if len(sys.argv) < 2:
		parser.print_help()
		sys.exit()

	### Run specified command
	# hapla cluster
	if sys.argv[1] == "cluster":
		if len(sys.argv) < 3:
			parser_cluster.print_help()
			sys.exit()
		else:
			from hapla import cluster
			cluster.main(args)
	
	# hapla struct
	if sys.argv[1] == "struct":
		if len(sys.argv) < 3:
			parser_struct.print_help()
			sys.exit()
		else:
			from hapla import struct
			struct.main(args)

	# hapla predict
	if sys.argv[1] == "predict":
		if len(sys.argv) < 3:
			parser_predict.print_help()
			sys.exit()
		else:
			from hapla import predict
			predict.main(args)

	# hapla admix
	if sys.argv[1] == "admix":
		if len(sys.argv) < 3:
			parser_admix.print_help()
			sys.exit()
		else:
			from hapla import admix
			admix.main(args)

	# hapla fatash
	if sys.argv[1] == "fatash":
		if len(sys.argv) < 3:
			parser_fatash.print_help()
			sys.exit()
		else:
			from hapla import fatash
			fatash.main(args)



##### Define main #####
if __name__ == "__main__":
	main()
