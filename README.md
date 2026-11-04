# hapla

[![Tests](https://github.com/Rosemeis/hapla/actions/workflows/ci.yml/badge.svg)](https://github.com/Rosemeis/hapla/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/hapla.svg)](https://pypi.org/project/hapla/)
[![License](https://img.shields.io/badge/license-GPL--3.0-blue)](https://github.com/Rosemeis/hapla/blob/main/LICENSE)

**hapla** groups phased haplotypes into local clusters. Use the assignments for
PCA, genomic relationship matrices, admixture, local ancestry, and haplotype
sharing. Clustering and prediction read VCF/BCF. All commands run on CPU.

This README describes **1.x**. Files from earlier versions are incompatible
and must be rebuilt from the original genotypes.

## Installation

Use Linux with Python 3.11 or newer. NumPy is the only Python runtime dependency.
Source builds also need a C compiler with OpenMP and HTSlib 1.20+ headers and
libraries.

Install the current GitHub version:

```bash
git clone https://github.com/Rosemeis/hapla.git
cd hapla
python -m pip install .
```

With Conda, use the [CPU environment](https://github.com/Rosemeis/hapla/blob/main/environment-cpu.yml) from the cloned
repository:

```bash
conda env create --file environment-cpu.yml
conda activate hapla
HTSLIB_PREFIX="$CONDA_PREFIX" python -m pip install --no-build-isolation .
```

Install a released 1.x version from PyPI with:

```bash
python -m pip install "hapla>=1.0.0"
```

The [release workflow](https://github.com/Rosemeis/hapla/blob/main/.github/workflows/release.yml) builds wheels for Linux
(glibc 2.28+) on x86-64 and ARM64. Wheels bundle HTSlib, libdeflate, and OpenMP.
No compiler or separate HTSlib/OpenMP installation is needed. Pip installs NumPy.
The system supplies glibc and zlib, as on RHEL/Rocky/AlmaLinux 8+ and Ubuntu 20.04+.
Source installations need the build dependencies above. Older systems need a source
build or a newer container. PyPI may lag behind the GitHub version.

Builds are portable by default. To tune a source build to the current CPU, use
`HAPLA_NATIVE=1 python -m pip install .` when build and compute CPUs match.
`HTSLIB_PREFIX` selects a custom HTSlib installation.
Check the installation with `hapla --version`.

## Quick start

Cluster a phased chromosome and save medians for assigning new samples:

```bash
hapla cluster --bcf data.chr1.bcf --size 16 --medians --threads 8 --out chr1
hapla predict --bcf query.chr1.bcf --ref chr1 --threads 8 --out query.chr1
```

After clustering each chromosome, fit ancestry proportions and local ancestry:

```bash
hapla admix --clusters chr{1..22} --K 5 --threads 8 --out fit
hapla local --clusters chr{1..22} \
    --pfilelist fit.K5.s42.plist --qfile fit.K5.s42.Q \
    --threads 8 --out lai
```

Bash and Zsh expand `chr{1..22}` in chromosome order. Explicit prefixes or
`--filelist prefixes.txt` also work. Cluster files must have identical sample
IDs in the same order. P-file lists must follow cluster order, and Q rows must
follow sample order.

Keep the `.bca`, `.ids`, `.win`, and `.ref` files together. Every command writes
a `.log` with its settings, results, and timings. Use separate input and output
prefixes. Run `hapla COMMAND --help` for all options and defaults.

## Clustering and prediction

Input must be sorted, diploid, and biallelic, with phased heterozygous calls.
A missing allele excludes its haplotype from that window. Windows do not cross
chromosome boundaries.

Choose `--size` for variant counts, `--length` for physical spans, `--windows`
for explicit boundaries, or `--adaptive` for variable windows. Clusters require
at least five haplotypes and 0.1% frequency by default, with at most 255 clusters
per window. Adjust these with `--min-mac`, `--min-freq`, and `--max-clusters`.
Add `--plink` for PLINK output.

Prediction requires a reference made with `--medians`. Chromosome, position,
REF, ALT, and variant order must match the reference exactly. Samples may differ.
Saved medians and their companion files must stay together.

## Population structure

```bash
hapla struct --clusters chr{1..22} --pca 20 --loadings --threads 8 --out pca
hapla struct --clusters chr{1..22} --grm --threads 8 --out grm
hapla struct --clusters query.chr{1..22} --projection pca --threads 8 --out query
```

PCA writes `.vec` and `.val`. `--loadings` also saves the files needed for
projection. Query samples may differ, but cluster references and their order
must match the fitted model. GRM output uses GCTA format, with counts of window
contrasts. PCA and GRM mean-impute missing haplotypes.

For phased sharing across ordered, nonoverlapping windows:

```bash
hapla struct --clusters chr{1..22} --hsm-svd 20 --threads 8 --out sharing
hapla struct --clusters chr{1..22} --hsm-groups 5 --threads 8 --out groups
```

HSM components and grouping use sparse sharing profiles. Add `--hsm` to export
the full kernel, `--hsm-sqrt` for square-root profiles, or `--map FILE` for genetic
weights. Full GRM and HSM output requires quadratic storage.
HSM eigenvectors have unit norm. Multiply them by the square roots of their
eigenvalues for kernel PCA scores. Groups need not represent discrete
populations. Sharing matches are not verified IBD.

## Global and local ancestry

`admix` estimates ancestry proportions Q and cluster frequencies P. The default
initializer uses SVD/ALS. Compare several seeds for each K and check convergence
in the log. Component labels can permute between fits.

- `--supervised FILE` fixes known Q rows using labels 1..K, with 0 for unknowns.
- `--projection FILE` fits Q at fixed P, using a P matrix or P-file list.
- `--bcf FILE` adds per-window weighted SNPs to a hybrid cluster/SNP fit.
  Windows must not overlap. Hybrid projection needs the same SNP input and the
  reference `.snp.P` file.
- `--loo` uses a leave-one-individual-out Q correction. It is a current-count
  correction, not an independent model refit for each individual.

Fitted files use `<out>.K<K>.s<seed>`, or `<out>.proj.K<K>.s<seed>` for projection.
They include `.Q`, `.ids`, `.log`, and fitted P files. Multiple inputs also write
`.plist`. The default seed is 42.

`local` uses matching P/Q files for local ancestry. Its default is regularized
Baum–Welch refinement. Add `--fixed-model` to decode without refinement,
`--save-posteriors` for decoded-call support, or `--viterbi` for Viterbi paths.
Use `--map FILE --time GENERATIONS` for genetic transitions, or
`--map FILE --dating` to estimate a date. Maps contain chromosome, position in
bp, and cumulative cM, and must cover all windows. Decoded-call support is a
diagnostic, not a calibrated ancestry probability.

## Ancestry-specific data and residuals

Create ancestry-specific copies from local ancestry paths:

```bash
printf '%s\n' lai.chr{1..22}.path > paths.txt
hapla deconv --clusters chr{1..22} --path-filelist paths.txt --K 5 --out deconv/lai
```

This writes ancestry-specific cluster bundles and a `.filelist`. Use
`--format bcf` or `--format both` with `--bcf-filelist FILE` for masked BCF output.
Genotype sample order and coordinates must match the cluster input.

Evaluate Q through empirical, expected, and corrected residual correlations:

```bash
hapla eval --clusters chr{1..22} --qfile fit.K5.s42.Q --threads 8 --out residuals
```

Outputs are `.bhat`, `.chat`, and `.corres`. Missing haplotypes use observed-count
fitted means. Evaluation memory and output scale quadratically with sample count.

## Development and releases

After installing the package:

```bash
python tests/run.py --installed
python tests/run.py --installed --threads 2
ruff check .
ruff format --check .
```

[CI](https://github.com/Rosemeis/hapla/actions/workflows/ci.yml) checks Python style, Cython warnings, distributions,
and the installed regression suite. Tests use `unittest`, fixed seeds, and small
fixtures. Helpers use camelCase names, `###` headings, and a blank line before
comments unless they start an indented block.

Release-related pull requests and manual runs of the
[publish workflow](https://github.com/Rosemeis/hapla/actions/workflows/release.yml)
build and test release artifacts. Publishing a GitHub release named `v<version>` uploads
them to PyPI after validation. The tag must match `hapla/__init__.py`.
Configure a [PyPI Trusted Publisher](https://docs.pypi.org/trusted-publishers/adding-a-publisher/)
for owner `Rosemeis`, repository `hapla`, workflow `release.yml`, and environment
`pypi` before the first automated publication.

## Citation and support

- **hapla cluster**: [Nature Communications](https://doi.org/10.1038/s41467-024-55477-3), [preprint](https://doi.org/10.1101/2024.04.30.24306654)
- **hapla admix**: [HGG Advances](https://doi.org/10.1016/j.xhgg.2026.100561), [preprint](https://doi.org/10.1101/2025.09.02.673718)

Licensed under [GPL v3.0](https://github.com/Rosemeis/hapla/blob/main/LICENSE). Report problems through
[GitHub Issues](https://github.com/Rosemeis/hapla/issues), including the command,
version, and error message.
