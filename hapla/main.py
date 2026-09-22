"""Command options and dispatch for Hapla."""

__author__ = "Jonas Meisner"

import argparse
import sys
from importlib import import_module

from hapla import __version__


### Parse command options and dispatch once
def main():
    parser = argparse.ArgumentParser(prog="hapla")
    parser.add_argument("--version", action="version", version=f"v{__version__}")
    subs = parser.add_subparsers(title="hapla commands")

    ##### Shared options
    cmds = {
        name: subs.add_parser(name)
        for name in ("cluster", "predict", "struct", "admix", "fatash", "eval")
    }
    clu, pre, pca, adm, lai, eva = cmds.values()
    for name, sub in cmds.items():
        sub.add_argument("--version", action="version", version=f"v{__version__}")
        sub.add_argument(
            "-t", "--threads", type=int, default=1, metavar="INT", help="Number of threads (1)"
        )
        sub.add_argument(
            "-o",
            "--out",
            default="hapla.pca" if name == "struct" else f"hapla.{name}",
            metavar="OUTPUT",
            help="Output prefix",
        )
    for sub in (pca, adm, lai, eva):
        sub.add_argument(
            "-f",
            "--filelist",
            metavar="FILE",
            help="File with one cluster prefix per line, in input order",
        )
        sub.add_argument(
            "-z",
            "--clusters",
            nargs="+",
            metavar="PREFIX",
            help="One or more cluster prefixes, in input order (shell expansion allowed)",
        )
    for sub in (clu, pre):
        sub.add_argument("--plink", action="store_true", help="Generate binary PLINK output")
        sub.add_argument(
            "--buffer-mb",
            type=int,
            default=256,
            metavar="INT",
            help="Input buffer budget in MiB (256). Worker scratch is extra",
        )
        sub.add_argument(
            "--io-threads",
            type=int,
            metavar="INT",
            help="HTSlib decompression threads within the total --threads budget (automatic)",
        )
        sub.add_argument(
            "--batch-windows",
            type=int,
            default=32,
            metavar="INT",
            help="Maximum windows per worker task (32)",
        )
    for sub in (clu, pre, pca):
        sub.add_argument("--duplicate-fid", action="store_true", help="Use sample ID as family ID")
    for sub in (pca, adm):
        sub.add_argument(
            "--chunk",
            type=int,
            default=4096,
            metavar="INT",
            help="Target cluster alleles per calculation block (4096)",
        )
        sub.add_argument(
            "--power",
            type=int,
            default=11,
            metavar="INT",
            help="Number of power iterations to perform (11)",
        )
        sub.add_argument("--seed", type=int, default=42, metavar="INT", help="Random seed (42)")
    for sub in (lai, eva):
        sub.add_argument(
            "-q", "--qfile", metavar="FILE", help="Path to file with ancestry proportions"
        )

    ##### hapla cluster
    clu.add_argument(
        "-g", "--vcf", "--bcf", metavar="FILE", help="Input phased genotype file in VCF/BCF format"
    )
    clu.add_argument("-f", "--size", type=int, metavar="INT", help="Use fixed sized windows")
    clu.add_argument(
        "-l", "--length", type=int, metavar="INT", help="Use length-based windows (in BP)"
    )
    clu.add_argument(
        "-w", "--windows", metavar="FILE", help="Use provided start indices for windows"
    )
    clu.add_argument(
        "-p",
        "--lmbda",
        type=float,
        default=0.1,
        metavar="FLOAT",
        help="Set lambda hyperparameter (0.1)",
    )
    clu.add_argument(
        "-s",
        "--step",
        type=int,
        metavar="INT",
        help="Step-size for sliding window using fixed sized windows",
    )
    clu.add_argument(
        "--min-freq",
        type=float,
        default=0.005,
        metavar="FLOAT",
        help="Minimum haplotype cluster frequency (0.005)",
    )
    clu.add_argument(
        "--min-mac", type=int, metavar="INT", help="Minimum haplotype cluster allele count"
    )
    clu.add_argument(
        "--max-clusters",
        type=int,
        default=255,
        metavar="INT",
        help="Maximum biological clusters per window, 1..255 (255). Byte 255 means missing",
    )
    clu.add_argument(
        "--max-iterations",
        type=int,
        default=1000,
        metavar="INT",
        help="Maximum number of iterations (1000)",
    )
    clu.add_argument("--medians", action="store_true", help="Save haplotype cluster medians")
    clu.add_argument(
        "--tail",
        choices=("include", "drop"),
        default="include",
        help="Include or drop a final incomplete fixed-size window on each chromosome (include)",
    )
    clu.add_argument(
        "--missing",
        choices=("window", "error"),
        default="window",
        help="Mark an affected haplotype window missing, or reject missing GT (window)",
    )

    ##### hapla predict
    pre.add_argument(
        "-g",
        "--vcf",
        "--bcf",
        metavar="FILE",
        help="Input phased/unphased genotype file in VCF/BCF format",
    )
    pre.add_argument(
        "-b", "--bfile", metavar="FILE", help="Input unphased genotype file in binary PLINK format"
    )
    pre.add_argument(
        "-r",
        "--ref",
        metavar="FILE",
        help="Input reference prefix of pre-estimated cluster medians",
    )
    pre.add_argument(
        "--phase-mode",
        choices=("auto", "phased", "unphased"),
        default="auto",
        help="Detect phase per sample/window, require phased GT, or ignore phase (auto)",
    )

    ##### hapla struct
    pca.add_argument(
        "--grm", action="store_true", help="Estimate genome-wide relationship matrix (GRM)"
    )
    pca.add_argument("--projection", metavar="FILE", help="Project samples on to existing PC space")
    pca.add_argument(
        "--no-centering", action="store_true", help="Do not perform Gower and data centering of GRM"
    )
    pca.add_argument("--pca", type=int, metavar="INT", help="Perform PCA and extract eigenvectors")
    pca.add_argument("--loadings", action="store_true", help="Save frequencies and loadings of SVD")
    pca.add_argument("--raw", action="store_true", help="Raw output without '*.fam' info")

    ##### hapla admix
    adm.add_argument("-k", "--K", type=int, metavar="INT", help="Number of ancestral components")
    adm.add_argument(
        "--keep", metavar="FILE", help="File with sample IDs to include, one ID per line"
    )
    adm.add_argument(
        "--iter", type=int, default=1000, metavar="INT", help="Maximum number of iterations (1000)"
    )
    adm.add_argument(
        "--tole",
        type=float,
        default=1e-9,
        metavar="FLOAT",
        help="Tolerance in log likelihood / (2 * samples * cluster alleles) (1e-9)",
    )
    adm.add_argument(
        "--batches", type=int, default=16, metavar="INT", help="Number of initial mini-batches (16)"
    )
    adm.add_argument(
        "--check",
        type=int,
        default=5,
        metavar="INT",
        help="Number of iterations between convergence checks (5)",
    )
    adm.add_argument("--supervised", metavar="FILE", help="Path to population assignment file")
    adm.add_argument(
        "--projection",
        metavar="FILE",
        help="Path to ancestral haplotype cluster allele frequencies file",
    )
    adm.add_argument(
        "--als-iter",
        metavar="INT",
        type=int,
        default=1000,
        help="Maximum number of iterations in ALS (1000)",
    )
    adm.add_argument(
        "--als-tole",
        metavar="FLOAT",
        type=float,
        default=1e-4,
        help="Tolerance for RMSE of Q between ALS iterations (1e-4)",
    )
    adm.add_argument(
        "--subsampling",
        metavar="INT",
        type=int,
        default=4,
        help="Subsampling factor for ALS/SVD initialization (4)",
    )
    adm.add_argument(
        "--no-freqs", action="store_true", help="Do not save haplotype cluster frequencies"
    )
    adm.add_argument(
        "--random-init", action="store_true", help="Random initialization of parameters"
    )
    adm.add_argument(
        "--prefix",
        default="chr",
        metavar="OUTPUT",
        help="Prefix for multiple haplotype cluster frequency files",
    )

    ##### hapla fatash
    lai.description = (
        "Local ancestry inference with regularized Baum-Welch fitting of supplied P/Q by default."
    )
    lai.add_argument(
        "-e",
        "--pfilelist",
        metavar="FILE",
        help="File with one P-file path per line, in cluster input order",
    )
    lai.add_argument(
        "-p",
        "--pfile",
        nargs="+",
        metavar="FILE",
        help="One or more P files, in cluster input order (shell expansion allowed)",
    )
    lai.add_argument(
        "--buffer-mb",
        type=int,
        default=256,
        metavar="INT",
        help="HMM workspace budget in MiB (256)",
    )
    lai.add_argument(
        "--block",
        type=int,
        metavar="INT",
        default=1,
        help="Number of genomic windows in block to increase smoothness (1)",
    )
    lai.add_argument(
        "--min-length",
        type=int,
        metavar="INT",
        help="Minimum window length in BP to include in HMM",
    )
    lai.add_argument(
        "--max-length",
        type=int,
        metavar="INT",
        help="Maximum window length in BP to include in HMM",
    )
    lai.add_argument(
        "--quantile",
        type=float,
        metavar="FLOAT",
        help="Quantile of window lengths to include in HMM",
    )
    lai.add_argument(
        "--alpha",
        type=float,
        metavar="FLOAT",
        help="Single transition rate per HMM block (not a genetic distance)",
    )
    lai.add_argument(
        "--medians", action="store_true", help="Utilize haplotype cluster probabilities"
    )
    lai.add_argument(
        "--viterbi", action="store_true", help="Perform Viterbi decoding for LAI inference (HMM)"
    )
    lai.add_argument(
        "--save-posteriors",
        action="store_true",
        help="Save posterior probabilities from posterior decoding",
    )
    lai.add_argument(
        "--phase-correct",
        nargs="?",
        const=0,
        type=int,
        metavar="INT",
        help="Correct reciprocal phase switches within a window distance (0)",
    )
    lai.add_argument(
        "--prefix", default="chr", metavar="OUTPUT", help="Prefix for multiple path files"
    )
    fit = lai.add_mutually_exclusive_group()
    fit.add_argument("--baum-welch", action="store_true", help="Fit P/Q without regularization")
    fit.add_argument(
        "--fixed-model", action="store_true", help="Decode supplied P/Q without fitting"
    )
    lai.add_argument("--iter", type=int, default=10, help="Maximum HMM fitting iterations (10)")
    lai.add_argument(
        "--tole",
        type=float,
        default=1e-5,
        help="Objective tolerance per observed assignment (1e-5)",
    )
    lai.add_argument(
        "--p-prior", type=float, default=10, help="P pseudocount mass per window and ancestry (10)"
    )
    lai.add_argument(
        "--q-prior", type=float, default=10, help="Q pseudocount mass per individual (10)"
    )
    lai.add_argument(
        "--alpha-min",
        type=int,
        default=4,
        help="Lowest negative-log10 alpha exponent in the ensemble (4)",
    )
    lai.add_argument(
        "--alpha-max",
        type=int,
        default=9,
        help="Highest negative-log10 alpha exponent in the ensemble (9)",
    )
    lai.add_argument(
        "--alpha-num", type=int, default=5, help="Number of alpha values to evaluate (5)"
    )
    lai.add_argument(
        "--simple", action="store_true", help="Use column-normalized simplified HMM transitions"
    )

    ##### hapla eval
    eva.add_argument(
        "--keep",
        metavar="FILE",
        help="File with sample IDs to include, one ID per line, in Q row order",
    )

    # Parse once and dispatch through the matching command parser
    args = parser.parse_args()
    if len(sys.argv) < 2:
        parser.print_help()
        return
    cmd = sys.argv[1]
    sub = cmds[cmd]
    if len(sys.argv) < 3:
        sub.print_help()
        return
    run = import_module(f"hapla.{cmd}").main
    args._cmd = ["hapla", *sys.argv[1:]]
    try:
        run(args)
    except (ValueError, OSError, RuntimeError) as err:
        sub.error(str(err))


if __name__ == "__main__":
    main()
