# hapla

[![Tests](https://github.com/Rosemeis/hapla/actions/workflows/ci.yml/badge.svg)](https://github.com/Rosemeis/hapla/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-GPL--3.0-blue)](LICENSE)

**hapla** groups phased haplotypes into local clusters. Use the assignments for
PCA and genomic relationship matrices, admixture, local ancestry, or residual
correlations. Clustering and prediction read VCF/BCF. All commands run on CPU.

**Files from before v1.0.0 are incompatible.** Rebuild clusters and downstream
results from the original genotypes. Do not mix files from the two formats.

## Installation

Requires Python 3.10+, NumPy >2.0, a C compiler with OpenMP, and HTSlib 1.20+
headers and libraries. NumPy is the only Python runtime dependency.

```bash
git clone https://github.com/Rosemeis/hapla.git
cd hapla
conda env create -f environment-cpu.yml
conda activate hapla
HTSLIB_PREFIX="$CONDA_PREFIX" python -m pip install --no-build-isolation .
```

Alternatively, install HTSlib and OpenMP through your system package manager,
then run `python -m pip install .`. The build finds HTSlib through `pkg-config`,
Conda, or standard prefixes. Set `HTSLIB_PREFIX` or, on macOS, `LIBOMP_PREFIX`
for custom locations. Their shared libraries must remain available at runtime.

The default build avoids host-specific CPU instructions. Set `HAPLA_NATIVE=1`
when building for the same CPU architecture and instruction set as the compute
nodes.

## Quick start

Cluster a phased BCF and save medians for assigning new samples:

```bash
hapla cluster --bcf data.chr1.bcf --size 16 --medians --threads 8 --out chr1
hapla predict --bcf query.chr1.bcf --ref chr1 --threads 8 --out query.chr1
```

After clustering each chromosome, pass the cluster prefixes in chromosome order:

```bash
hapla struct --clusters chr{1..22} --pca 20 --loadings --threads 8 --out pca
hapla admix --clusters chr{1..22} --K 5 --threads 8 --out fit
hapla fatash --clusters chr{1..22} \
    --pfile fit.K5.s42.chr{1..22}.P --qfile fit.K5.s42.Q \
    --threads 8 --out lai
hapla eval --clusters chr{1..22} --qfile fit.K5.s42.Q --threads 8 --out residuals
```

Bash and Zsh expand unquoted braces. Hapla uses the resulting order as given.
For a saved list, use `--filelist prefixes.txt` with one prefix per line.
`fatash` also accepts `--pfilelist fit.K5.s42.pfilelist`. Cluster and P-file
lists must have the same order, and sample IDs must match across cluster files.
Q rows must follow that sample order. Hapla checks the sibling `.ids` file for
saved `.Q` files when present. Check each log for convergence and input notes.
For `admix`, compare several seeds at each K. Component labels may permute
between fits.

Each command writes a `.log` with its arguments, results, and timings. Outputs
are staged before they replace files at the requested prefix. Use separate
prefixes for inputs and results.

### Common options

These options apply to every command. A dash means no default value.

| Option | Default | Description |
| --- | --- | --- |
| `-h`, `--help` | — | Show command help and exit |
| `--version` | — | Show version and exit |
| `-t`, `--threads INT` | `1` | CPU thread budget |
| `-o`, `--out PREFIX` | `hapla.<command>` | Output prefix, except `struct` uses `hapla.pca` |

`hapla --help` and `hapla --version` also work without a subcommand.

### Cluster input options

Shared by `struct`, `admix`, `fatash`, and `eval`. Select exactly one input form.

| Option | Default | Description |
| --- | --- | --- |
| `-z`, `--clusters PREFIX...` | — | One or more cluster prefixes in input order |
| `-f`, `--filelist FILE` | — | File with one cluster prefix per line |

A cluster bundle contains `.bca`, `.ids`, `.win`, and `.ref.json`. The `.bca`
format stores one byte per haplotype per window. Labels 0–254 represent up to
255 clusters. Byte 255 means missing. A window with no observations has K=0.
Keep bundles and their model files together.

### Genotype reader options

`cluster` and `predict` read VCF/BCF sequentially through HTSlib without an
index. Only `FORMAT/GT` is used. Nonstandard `FORMAT/PP` header warnings are
shown as one input note.

| Option | Default | Description |
| --- | --- | --- |
| `-g`, `--vcf FILE`, `--bcf FILE` | — | Genotype input |
| `--buffer-mb INT` | `256` | Input and queued-window budget in MiB |
| `--io-threads INT` | Automatic | HTSlib decompression threads within the CPU budget |
| `--batch-windows INT` | `32` | Maximum windows per worker task |
| `--plink` | Off | Also write `.bed`, `.bim`, and `.fam` |
| `--duplicate-fid` | Off | Use sample ID as FID instead of `0` |

One window must fit the reader's quarter of the input budget. Worker scratch,
mapped references, HTSlib, and runtime libraries use additional memory.
Thread allocation includes reading, window workers, and the HTSlib coordinator.
The log records the HTSlib version and actual thread and batch allocation.

## hapla cluster

Requires sorted, phased, diploid, biallelic GT. Select one window definition.
Windows never cross chromosome boundaries. Any missing allele marks its
haplotype missing for the entire window. A fully observed partner remains usable.
PLINK output marks the diploid genotype missing if either haplotype is missing.

| Option | Default | Description |
| --- | --- | --- |
| `-f`, `--size INT` | — | Variants per window |
| `-l`, `--length INT` | — | Physical span in bp, from the first variant through start + span |
| `-w`, `--windows FILE` | — | Increasing zero-based start indices, beginning at zero |
| `-s`, `--step INT` | Window size | Step for overlapping `--size` windows |
| `-p`, `--lmbda FLOAT` | `0.1` | Window fraction defining the Hamming-distance growth threshold |
| `--min-freq FLOAT` | `0.005` | Minimum cluster frequency among observed haplotypes |
| `--min-mac INT` | — | Minimum cluster count, overriding `--min-freq` |
| `--max-clusters INT` | `255` | Maximum clusters per window, from 1 to 255 |
| `--max-iterations INT` | `1000` | Iteration limit for fitting each window |
| `--tail {include,drop}` | `include` | Keep or omit incomplete final fixed-size windows |
| `--missing {window,error}` | `window` | Mark affected haplotypes missing or reject missing GT |
| `--medians` | Off | Save reference medians and variant metadata for prediction |

For overlapping windows, a short tail is added only if it covers new variants.
A `--windows` file may end with the genotype record count as an EOF marker.
The fitter deduplicates haplotypes, grows binary medians using packed Hamming
distances, then reassigns haplotypes from clusters below the size threshold.
The 255-cluster cap keeps assignments to one byte per haplotype. A count
threshold above the observed haplotype count is invalid. Fitting stops if a
window does not converge.

Clustering is invariant to REF/ALT swaps with corresponding GT recoding when
sample/haplotype order is fixed. Major alleles define the internal orientation.
At 50:50 sites, the first complete haplotype breaks the tie. Saved medians retain
the input allele coding.

Outputs are `.bca`, `.ids`, `.win`, `.ref.json`, and `.log`. `--medians` adds
`.bcm` medians, `.blk` cluster log-likelihood scores, `.wix` window indices, and
`.sites` ordered variants.

## hapla predict

Assign new samples to reference clusters. Provide genotype input or a PLINK
prefix, and a reference produced by `cluster --medians`. The entire variant set
must match chromosome, position, REF, ALT, and order. Samples may differ.

| Option | Default | Description |
| --- | --- | --- |
| `-b`, `--bfile PREFIX` | — | SNP-major PLINK `.bed`, `.bim`, and `.fam` input |
| `-r`, `--ref PREFIX` | Required | Cluster reference prefix |
| `--phase-mode {auto,phased,unphased}` | `auto` | Detect phase per sample/window, require phase, or ignore it |

`auto` uses the cluster-pair heuristic for samples with ambiguous unphased calls
in a window. Otherwise, it assigns haplotypes independently. Missing phased
alleles affect only their haplotype. Missing unphased alleles affect both.
PLINK input is always unphased, with BIM A2 matching REF and A1 matching ALT.
Alleles are not flipped automatically. **Unphased predictions are unsuitable
for local ancestry inference.**

Required reference files are `.bcm`, `.wix`, `.sites`, `.win`, and `.ref.json`.
Outputs are a new assignment bundle and `.log`, with optional PLINK files.

## hapla struct

Estimate PCA, a genomic relationship matrix, or project onto saved PCs.
Select at least one operation. Missing haplotypes use mean imputation.
PCA requires empirical dosage variation and enough rank for the requested PCs.

| Option | Default | Description |
| --- | --- | --- |
| `--pca INT` | — | Number of principal components |
| `--grm` | Off | Estimate the GRM |
| `--projection PREFIX` | — | Project onto a saved PCA model |
| `--loadings` | Off | Save loadings, frequencies, and PCA identity metadata |
| `--raw` | Off | Write PC values without FID/IID columns |
| `--duplicate-fid` | Off | Use sample ID as FID instead of `0` |
| `--no-centering` | Off | Disable Gower and data centering of the GRM |
| `--chunk INT` | `4096` | Target cluster alleles per calculation block |
| `--power INT` | `11` | Randomized PCA power iterations |
| `--seed INT` | `42` | Random seed |

PCA writes `.eigenvecs` and `.eigenvals`. `--loadings` also writes `.loadings`,
`.freqs`, and `.pca.json`, and requires assignment `.ref.json` files. Projection
writes `.project.eigenvecs` and checks the saved model's ordered references and
file partitioning. Query samples may differ.

GRM writes `.grm.bin`, `.grm.N.bin`, `.grm.id`, and `.grm.meta.json`. Each window
contributes `max(observed clusters - 1, 0)` to the contrast count.
Counts are constant across pairs under mean imputation. SNP-count-weighted GRM
merging is not supported. The log records the calculation times and dimensions.

## hapla admix

Estimate ancestry proportions Q and cluster frequencies P. Missing assignments
contribute no likelihood or counts. Empty windows are allowed. Individuals with
no observed assignments receive uniform Q unless fixed by supervision.

| Option | Default | Description |
| --- | --- | --- |
| `-k`, `--K INT` | Required | Ancestry components, with 1 < K < 100000 |
| `--keep FILE` | — | Sample IDs to retain, preserving cluster sample order |
| `--supervised FILE` | — | One population label per sample, 0 unknown and 1..K fixed |
| `--projection FILE` | — | Fixed P matrix, or P-file list for multiple cluster inputs |
| `--random-init` | Off | Use random P/Q instead of SVD/ALS initialization |
| `--source-init` | Off | Seed SVD/ALS from supported source extremes |
| `--iter INT` | `1000` | Maximum outer fitting iterations |
| `--tole FLOAT` | `1e-9` | Tolerance in log likelihood or objective / (2 × samples × cluster alleles) |
| `--p-prior FLOAT` | `0` | P pseudocount mass per window and ancestry, 0 disables shrinkage |
| `--batches INT` | `16` | Initial mini-batches, reduced during fitting |
| `--check INT` | `5` | Iterations between convergence checks and progress reports |
| `--chunk INT` | `4096` | Target cluster alleles per SVD calculation block |
| `--power INT` | `11` | SVD power iterations |
| `--seed INT` | `42` | Random seed |
| `--als-iter INT` | `1000` | Maximum ALS initialization iterations |
| `--als-tole FLOAT` | `1e-4` | ALS RMSE tolerance for Q |
| `--subsampling INT` | `4` | Chromosome subsampling factor for SVD initialization |
| `--no-freqs` | Off | Omit P outputs |
| `--prefix TEXT` | `chr` | Label for numbered P outputs |

Supervision and projection are mutually exclusive. Supervised labels must fit
0..255. Projection requires P rows to match the cluster order and fixes P while
fitting Q. The default unsupervised initializer uses SVD/ALS. `--source-init`
seeds ALS from supported extremes in the sample SVD scores and cannot be
combined with `--random-init`, supervision, or projection. Opt-in P shrinkage
uses pooled cluster frequencies within each window and cannot update a fixed
projection P. With shrinkage, convergence uses the regularized objective while
the log also records the likelihood. Timings cover each `--check` interval and
any final partial interval. Warm-up is separate.

EM uses float64 sample tiles with an 8 MiB target for Q scratch, subject to a
minimum of one sample per window partition. P counts reuse the existing output
buffers. Parameter arrays and input storage still scale with the dataset.

Outputs use `<out>.K<K>.s<seed>`, or `<out>.project.K<K>.s<seed>` for projection.
They include `.Q`, `.ids`, and `.log`. Fitted frequencies use `.P` for one input
or `.chr1.P`, `.chr2.P`, etc. plus `.pfilelist` for multiple inputs. P-file lists
contain absolute paths. Rerunning with `--no-freqs` removes prior P outputs for
the current input set.

## hapla fatash

Infer haploid local ancestry from admix P/Q. Regularized Baum–Welch refines P
and Q by default, with one Q per individual across all chromosomes. Each
window has its own P. Missing and length-filtered windows have neutral emissions.
Chains reset at chromosome boundaries.

| Option | Default | Description |
| --- | --- | --- |
| `-q`, `--qfile FILE` | Required | Q matrix in cluster sample order, with 1..255 ancestry columns |
| `-p`, `--pfile FILE...` | — | Ordered P files, one per cluster input |
| `-e`, `--pfilelist FILE` | — | File containing ordered P-file paths |
| `--baum-welch` | Off | Fit P/Q without regularization |
| `--fixed-model` | Off | Decode supplied P/Q without fitting |
| `--iter INT` | `10` | Maximum completed Baum–Welch updates |
| `--tole FLOAT` | `1e-5` | Objective improvement tolerance per observed assignment |
| `--p-prior FLOAT` | `10` | P pseudocount mass per window and ancestry |
| `--q-prior FLOAT` | `10` | Q pseudocount mass per individual |
| `--alpha FLOAT` | — | Single transition rate per HMM block |
| `--alpha-min INT` | `4` | Lower negative-log10 exponent of the alpha ensemble |
| `--alpha-max INT` | `9` | Upper negative-log10 exponent of the alpha ensemble |
| `--alpha-num INT` | `5` | Number of log-spaced alpha values |
| `--buffer-mb INT` | `256` | HMM batch workspace budget in MiB |
| `--block INT` | `1` | Windows per emission block |
| `--min-length INT` | — | Minimum included window length in bp |
| `--max-length INT` | — | Maximum included window length in bp |
| `--quantile FLOAT` | — | Central fraction of window lengths to retain, between 0 and 1 |
| `--medians` | Off | Use `.blk` cluster scores in emissions |
| `--simple` | Off | Use simplified column-normalized transitions |
| `--viterbi` | Off | Decode the most probable path for each alpha |
| `--save-posteriors` | Off | Save confidence for the mean-posterior calls |
| `--phase-correct [INT]` | Off | Correct reciprocal switches up to INT windows apart, default 0 when set |
| `--prefix TEXT` | `chr` | Label for numbered chromosome outputs |

Select exactly one P-input form. `--baum-welch` and `--fixed-model` are mutually
exclusive. `--medians`, `--simple`, and `--block` other than 1 require
`--fixed-model`. Quantile filtering cannot be combined with explicit length
bounds. `--save-posteriors` requires posterior decoding.

Regularization uses Dirichlet parameters `1 + mass × initial`, giving updates
`(expected counts + mass × initial) / (total counts + mass)`. P counts use
observed clusters. Q counts include chain starts and latent refresh events.
Individuals with no included observations retain input Q. Parameters without
counts retain their input values. Zero input probabilities are not smoothed
into positive support. Prior masses are tuning parameters.

Convergence is checked after each update using mean-alpha log likelihood minus
P/Q prior penalties. A material decrease restores the last accepted model.
Progress reports cover five updates and any final partial block. The initial
score is unnumbered. Fitting uses posterior expectations even with `--viterbi`.
The batch budget excludes parameter tables, mapped inputs, and runtime overhead.

Default decoding takes the largest mean ancestry posterior across five alpha
values from 1e-9 to 1e-4. Alpha is a rate per block, not a genetic distance or
estimated admixture time. A single-alpha Viterbi path is exact. Multiple-alpha
Viterbi uses per-window plurality. Phase correction is a heuristic.

Outputs include `.Q`, `.P`, `.ids`, `.path`, and `.log`. `.path` has one row per
haplotype and one column per window, with zero-based ancestry labels.
`--save-posteriors` adds `.prob`. Multiple inputs use `.chr1.P`, `.chr1.path`,
etc. and a `.pfilelist`. Reuse the same alpha and decoding options with saved
P/Q and `--fixed-model` to reproduce an analysis.

## hapla eval

Evaluate Q through empirical, model-expected, and corrected residual
correlations. Missing haplotypes use observed-count fitted means. All-missing
windows contribute zero. Memory and output scale quadratically with samples.

| Option | Default | Description |
| --- | --- | --- |
| `-q`, `--qfile FILE` | Required | Q matrix in cluster sample order or explicit keep order |
| `--keep FILE` | — | Sample IDs to retain, in Q row order |

Writes `.bhat`, `.chat`, `.corres`, `.ids`, and `.log`. Correlation matrices use
four decimal places. Fully unobserved or undefined rows are zero.

## Development

The test suite uses small fixtures and needs no external genotype data. After
installing hapla, run:

```bash
python tests/run.py --installed
python tests/run.py --installed --threads 2
python -m ruff check .
python -m ruff format --check .
```

[CI](.github/workflows/ci.yml) checks Python style and Cython warnings, validates
the wheel and source archive, and runs the installed tests with one and two
native threads. Pushes to `main` and pull requests use Ubuntu with Python 3.14.
Manual runs add Ubuntu and macOS 15 wheels for Python 3.10, 3.12, and 3.14.

Use short module summaries, `###` headings above functions and test classes,
and camelCase helper names. Keep a blank line before comments unless they start
an indented block. Tests use `unittest`, fixed seeds, and small shared fixtures.

## Citation

- **hapla cluster**: [Nature Communications](https://doi.org/10.1038/s41467-024-55477-3), [preprint](https://doi.org/10.1101/2024.04.30.24306654)
- **hapla admix**: [HGG Advances](https://doi.org/10.1016/j.xhgg.2026.100561), [preprint](https://doi.org/10.1101/2025.09.02.673718)
