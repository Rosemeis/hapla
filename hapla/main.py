"""
Main caller of hapla.
"""

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
	parser_cluster.add_argument("-f", "--fixed", type=int, default=10,
		metavar="INT", help="Use fixed window length (10)")
	parser_cluster.add_argument("-w", "--windows",
		metavar="FILE", help="Use provided window lengths")
	parser_cluster.add_argument("-l", "--lmbda", type=float, default=0.05,
		metavar="FLOAT", help="Set lambda hyperparameter (0.05)")
	parser_cluster.add_argument("-e", "--max_iterations", type=int, default=500,
		metavar="INT", help="Maximum number of iterations (500)")
	parser_cluster.add_argument("-t", "--threads", type=int, default=1,
		metavar="INT", help="Number of threads (1)")
	parser_cluster.add_argument("-o", "--out", default="hapla.cluster",
		metavar="OUTPUT", help="Output prefix")
	parser_cluster.add_argument("--min_freq", type=float, default=0.005,
		metavar="INT", help="Minimum frequency for haplotype cluster (0.005)")
	parser_cluster.add_argument("--max_clusters", type=int, default=64,
		metavar="INT", help="Maximum number of haplotype clusters per window (64)")
	parser_cluster.add_argument("--medians", action="store_true",
		help="Save haplotype cluster medians")
	parser_cluster.add_argument("--loglike", action="store_true",
		help="Compute log-likelihoods for ancestry estimation")
	parser_cluster.add_argument("--plink", action="store_true",
		help="Generate binary PLINK output")
	parser_cluster.add_argument("--duplicate_fid", action="store_true",
		help="Use sample list as FID")
	parser_cluster.add_argument("--verbose", action="store_true",
		help="Verbose output from each iteration")
	parser_cluster.add_argument("--filter",
		metavar="FILE", help="DEBUG FEATURE: filter out sites")

	# hapla pca
	parser_pca = subparsers.add_parser("pca")
	parser_pca.add_argument("-f", "--filelist", metavar="FILE",
		help="Filelist with paths to haplotype cluster alleles files")
	parser_pca.add_argument("-z", "--clusters", metavar="FILE",
		help="Path to a single haplotype cluster alleles file")
	parser_pca.add_argument("-e", "--eig", type=int, default=10,
		metavar="INT", help="Number of eigenvectors to extract (10)")
	parser_pca.add_argument("-t", "--threads", type=int, default=1,
		metavar="INT", help="Number of threads (1)")
	parser_pca.add_argument("-o", "--out", default="hapla.pca",
		metavar="OUTPUT", help="Output prefix")
	parser_pca.add_argument("--min_freq", type=float,
		metavar="FLOAT", help="Minimum frequency for haplotype cluster")
	parser_pca.add_argument("--loadings", action="store_true",
		help="Save loadings of SVD")
	parser_pca.add_argument("--randomized", action="store_true",
		help="Use randomized SVD (use for very large data)")
	parser_pca.add_argument("--grm", action="store_true",
		help="Estimate genome-wide relationship matrix (GRM)")
	parser_pca.add_argument("--alpha", type=float, default=0.25,
		metavar="FLOAT", help="Alpha selection parameter in GRM (0.25)")
	parser_pca.add_argument("--no_centering", action="store_true",
		help="Do not perform centering on GRM/HSM matrix")
	parser_pca.add_argument("--hsm", action="store_true",
		help="Estimate haplotype sharing matrix (HSM)")
	parser_pca.add_argument("--iid", metavar="FILE",
		help="Sample ID list for GCTA format")
	parser_pca.add_argument("--fid", metavar="FILE",
		help="Family ID list for GCTA format")
	parser_pca.add_argument("--batch", type=int, default=1000,
		metavar="INT", help="Number of clusters in batched SVD")

	# hapla predict
	parser_predict = subparsers.add_parser("predict")
	parser_predict.add_argument("-g", "--vcf", "--bcf", metavar="FILE",
		help="Input phased genotype file in VCF/BCF format")
	parser_predict.add_argument("-m", "--medians", metavar="FILE",
		help="Input haplotype cluster medians as binary NumPy array")
	parser_predict.add_argument("-f", "--fixed", type=int, default=10,
		metavar="INT", help="Use fixed window length (10)")
	parser_predict.add_argument("-w", "--windows", metavar="FILE",
		help="Use provided window lengths")
	parser_predict.add_argument("-t", "--threads", type=int, default=1,
		metavar="INT", help="Number of threads (1)")
	parser_predict.add_argument("-o", "--out", default="hapla.predict",
		metavar="OUTPUT", help="Output prefix")
	parser_predict.add_argument("--filter",
		metavar="FILE", help="DEBUG FEATURE: filter out sites")

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
	
	# hapla pca
	if sys.argv[1] == "pca":
		if len(sys.argv) < 3:
			parser_pca.print_help()
			sys.exit()
		else:
			from hapla import pca
			pca.main(args)

	# hapla predict
	if sys.argv[1] == "predict":
		if len(sys.argv) < 3:
			parser_predict.print_help()
			sys.exit()
		else:
			from hapla import predict
			predict.main(args)



##### Define main #####
if __name__ == "__main__":
	main()
