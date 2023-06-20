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
	parser_cluster.add_argument("-f", "--fixed", type=int, default=500,
		metavar="INT", help="Use fixed window length (500)")
	parser_cluster.add_argument("-w", "--windows",
		metavar="FILE", help="Use provided window lengths")
	parser_cluster.add_argument("-l", "--lmbda", type=float, default=0.166,
		metavar="FLOAT", help="Set lambda hyperparameter (0.166)")
	parser_cluster.add_argument("-m", "--max_clusters", type=int, default=40,
		metavar="INT", help="Maximum number of haplotype clusters per window (40)")
	parser_cluster.add_argument("-e", "--max_iterations", type=int, default=100,
		metavar="INT", help="Maximum number of iterations (100)")
	parser_cluster.add_argument("--min_freq", type=float, default=0.01,
		metavar="FLOAT", help="Haplotype cluster frequency threshold (0.01)")
	parser_cluster.add_argument("-t", "--threads", type=int, default=1,
		metavar="INT", help="Number of threads (1)")
	parser_cluster.add_argument("-o", "--out", default="hapla.cluster",
		metavar="OUTPUT", help="Output prefix")
	parser_cluster.add_argument("--medians", action="store_true",
		help="Save haplotype cluster medians")
	parser_cluster.add_argument("--loglike", action="store_true",
		help="Compute log-likelihoods for ancestry estimation")
	parser_cluster.add_argument("--verbose", action="store_true",
		help="Verbose output from each iteration")
	parser_cluster.add_argument("--filter",
		metavar="FILE", help="DEBUG FEATURE: filter out sites")

	# hapla pca
	parser_pca = subparsers.add_parser("pca")
	parser_pca.add_argument("-f", "--filelist", metavar="FILE",
		help="Filelist with paths to multiple haplotype cluster assignment files")
	parser_pca.add_argument("-z", "--clusters", metavar="FILE",
		help="Path to a single haplotype cluster assignment file")
	parser_pca.add_argument("-e", "--n_eig", type=int, default=10,
		metavar="INT", help="Number of eigenvectors to extract (10)")
	parser_pca.add_argument("-t", "--threads", type=int, default=1,
		metavar="INT", help="Number of threads (1)")
	parser_pca.add_argument("-o", "--out", default="hapla.pca",
		metavar="OUTPUT", help="Output prefix")
	parser_pca.add_argument("--min_freq", type=float, default=0.01,
		metavar="FLOAT", help="Haplotype cluster frequency threshold (0.01)")
	parser_pca.add_argument("--loadings", action="store_true",
		help="Save loadings of SVD")
	parser_pca.add_argument("--randomized", action="store_true",
		help="Use randomized SVD (use for very large data)")
	parser_pca.add_argument("--grm", action="store_true",
		help="Estimate genome-wide relationship matrix (only small data)")
	parser_pca.add_argument("--batch", type=int, default=1000,
		metavar="INT", help="Number of clusters in batch SVD")

	# hapla regress
	parser_regress = subparsers.add_parser("regress")
	parser_regress.add_argument("-f", "--filelist", metavar="FILE",
		help="Filelist with paths to multiple haplotype cluster assignment files")
	parser_regress.add_argument("-z", "--clusters", metavar="FILE",
		help="Path to a single haplotype cluster assignment file")
	parser_regress.add_argument("-y", "--pheno", metavar="FILE",
		help="Path to phenotype file")
	parser_regress.add_argument("-e", "--eigen", metavar="FILE",
		help="Path to file with eigenvectors (PCs)")
	parser_regress.add_argument("-c", "--covar", metavar="FILE",
		help="Path to file with covariates")
	parser_regress.add_argument("-s", "--seed", type=int, default=42,
		metavar="INT", help="Set random seed (42)")
	parser_regress.add_argument("-t", "--threads", type=int, default=1,
		metavar="INT", help="Number of threads (1)")
	parser_regress.add_argument("-o", "--out", default="hapla.asso",
		metavar="OUTPUT", help="Output prefix")
	parser_regress.add_argument("--block", type=int, default=100,
		metavar="INT", help="Number of haplotype cluster windows in a block (100)")
	parser_regress.add_argument("--folds", type=int, default=10,
		metavar="INT", help="Number of folds for cross validations (10)")
	parser_regress.add_argument("--ridge", type=int, default=5,
		metavar="INT", help="Number of ridge regressors in each level (5)")
	parser_regress.add_argument("--save_pred", action="store_true",
		help="Save whole-genome ridge regression predictions")
	parser_regress.add_argument("--save_beta", action="store_true",
		help="Save regression models for polygenic score estimation")
	parser_regress.add_argument("--save_loco", action="store_true",
		help="Save LOCO predictions")
	
	# hapla prs
	parser_prs = subparsers.add_parser("prs")
	parser_prs.add_argument("-f", "--filelist", metavar="FILE",
		help="Filelist with paths to multiple haplotype cluster assignment files")
	parser_prs.add_argument("-z", "--clusters", metavar="FILE",
		help="Path to a single haplotype cluster assignment file")
	parser_prs.add_argument("-a", "--assoc", metavar="FILE",
		help="Path to file with summary statistics")
	parser_prs.add_argument("-t", "--threads", type=int, default=1,
		metavar="INT", help="Number of threads (1)")
	parser_prs.add_argument("-o", "--out", default="hapla.prs",
		metavar="OUTPUT", help="Output prefix")

	# hapla predict
	parser_predict = subparsers.add_parser("predict")
	parser_predict.add_argument("-g", "--vcf", "--bcf", metavar="FILE",
		help="Input phased genotype file in VCF/BCF format")
	parser_predict.add_argument("-m", "--medians", metavar="FILE",
		help="Input haplotype cluster medians as binary NumPy array")
	parser_predict.add_argument("-f", "--fixed", type=int, default=500,
		metavar="INT", help="Use fixed window length (500)")
	parser_predict.add_argument("-w", "--windows", metavar="FILE",
		help="Use provided window lengths")
	parser_predict.add_argument("-t", "--threads", type=int, default=1,
		metavar="INT", help="Number of threads (1)")
	parser_predict.add_argument("-o", "--out", default="hapla.predict",
		metavar="OUTPUT", help="Output prefix")
	parser_predict.add_argument("--filter",
		metavar="FILE", help="DEBUG FEATURE: filter out sites")
	
	# hapla split
	parser_split = subparsers.add_parser("split")
	parser_split.add_argument("-g", "--vcf", "--bcf", metavar="FILE",
		help="Input phased genotype file in VCF/BCF format")
	parser_split.add_argument("-t", "--threads", type=int, default=1,
		metavar="INT", help="Number of threads (1)")
	parser_split.add_argument("-o", "--out", default="hapla.split",
		metavar="OUTPUT", help="Output files")
	parser_split.add_argument("--min_length", type=int, default=100,
		metavar="INT", help="Minimum number of SNPs in windows (100)")
	parser_split.add_argument("--max_length", type=int, default=1000,
		metavar="INT", help="Maximum number of SNPs in windows (1000)")
	parser_split.add_argument("--max_windows", type=int, default=5000,
		metavar="INT", help="Maximum number of windows allowed")
	parser_split.add_argument("--threshold", type=float, default=0.1,
		metavar="FLOAT", help="Lower bound for r^2 in window creation (0.1)")

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
			from src import cluster
			cluster.main(args)
	
	# hapla pca
	if sys.argv[1] == "pca":
		if len(sys.argv) < 3:
			parser_pca.print_help()
			sys.exit()
		else:
			from src import pca
			pca.main(args)

	# hapla regress
	if sys.argv[1] == "regress":
		if len(sys.argv) < 3:
			parser_regress.print_help()
			sys.exit()
		else:
			from src import regress
			regress.main(args)

	# hapla prs
	if sys.argv[1] == "prs":
		if len(sys.argv) < 3:
			parser_prs.print_help()
			sys.exit()
		else:
			from src import prs
			prs.main(args)

	# hapla predict
	if sys.argv[1] == "predict":
		if len(sys.argv) < 3:
			parser_predict.print_help()
			sys.exit()
		else:
			from src import predict
			predict.main(args)

	# hapla split
	if sys.argv[1] == "split":
		if len(sys.argv) < 3:
			parser_split.print_help()
			sys.exit()
		else:
			from src import split
			split.main(args)


##### Define main #####
if __name__ == "__main__":
	main()
