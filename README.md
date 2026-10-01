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
`fatash` also accepts `--pfilelist fit.K5.s42.plist`. Cluster and P-file
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

A cluster bundle contains `.bca`, `.ids`, `.win`, and `.ref`. The `.bca`
format stores one byte per haplotype per window. Labels 0–254 represent up to
255 clusters. Byte 255 means missing. A window with no observations has K=0.
Keep bundles and their model files together.

`.ref` and `.pca` are compact text records with reference identities and file
hashes. They prevent mixing cluster definitions or PCA model files. Regenerate
older references and PCA models for this format.

### Genotype reader options

`cluster` and `predict` read VCF/BCF sequentially through HTSlib without an
index. Only `FORMAT/GT` is used. Nonstandard `FORMAT/PP` header warnings are
shown as one input note.

| Option | Default | Description |
| --- | --- | --- |
| `-g`, `--vcf FILE`, `--bcf FILE` | — | Phased VCF/BCF input |
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
| `-p`, `--lmbda FLOAT` | `0` | Hamming-distance growth threshold as a window fraction, zero allows any mismatch |
| `--min-freq FLOAT` | `0.001` | Minimum cluster frequency among observed haplotypes, combined with `--min-mac` |
| `--min-mac INT` | `5` | Minimum cluster count, combined with `--min-freq` |
| `--max-clusters INT` | `255` | Maximum clusters per window, from 1 to 255 |
| `--max-iterations INT` | `1000` | Iteration limit per growth or refinement phase |
| `--tail {include,drop}` | `include` | Keep or omit incomplete final fixed-size windows |
| `--missing {window,error}` | `window` | Mark affected haplotypes missing or reject missing GT |
| `--medians` | Off | Save reference medians and variant metadata for prediction |

For overlapping windows, a short tail is added only if it covers new variants.
A `--windows` file may end with the genotype record count as an EOF marker.
The fitter deduplicates haplotypes and grows binary medians using XOR/popcount
Hamming distances. Each step selects the single furthest pattern from its nearest
median, breaking ties by pattern count and canonical allele order. Seeds require
at least `max(1, ceil(lmbda × window size))` differences, so any mismatch is
eligible by default. Medians are refined between insertions.

Final clusters require `max(min_mac, ceil(min_freq × observed haplotypes))`
members per window. Both limits apply, giving `max(5, ceil(0.001 × observed))`
by default. Use `--min-freq 0` for count-only support. Seed counts may be lower
if neighboring patterns supply enough members after refinement. Unsupported
clusters are pruned and their members reassigned until support and medians stabilize.

When one mismatch is eligible, supported unique patterns within the cluster cap
are returned directly with zero error. The 255-cluster cap keeps assignments to
one byte per haplotype. A support threshold above the observed haplotype count
is invalid. Fitting stops if growth or pruning does not converge.

Clustering is invariant to REF/ALT swaps with corresponding GT recoding when
sample/haplotype order is fixed. Major alleles define the internal orientation.
At 50:50 sites, the first complete haplotype breaks the tie. Saved medians retain
the input allele coding.

Outputs are `.bca`, `.ids`, `.win`, `.ref`, and `.log`. `--medians` adds
`.bcm` medians, `.blk` cluster log-likelihood scores, `.wix` window indices, and
`.sites` ordered variants.

## hapla predict

Assign phased haplotypes to reference clusters. Provide a diploid, biallelic
VCF/BCF and a reference produced by `cluster --medians`. The entire variant set
must match chromosome, position, REF, ALT, and order. Samples may differ.

Predicting the original data reproduces the cluster assignments exactly,
including missing labels. Both commands resolve equal Hamming distances in
favor of the higher cluster index.

| Option | Default | Description |
| --- | --- | --- |
| `-r`, `--ref PREFIX` | Required | Cluster reference prefix |

Missing alleles mark only the affected haplotype window missing. Unphased
heterozygous or partially missing calls are rejected. Unambiguous calls such as
`0/0`, `1/1`, and `./.` are accepted. Alleles are not flipped automatically.

Required reference files are `.bcm`, `.wix`, `.sites`, `.win`, and `.ref`.
Outputs are a new assignment bundle and `.log`, with optional PLINK files.

## hapla struct

Estimate PCA, a genomic relationship matrix, or project onto saved PCs.
Select at least one operation. PCA and GRM mean-impute missing haplotypes.
PCA requires empirical dosage variation and enough rank for the requested PCs.
The haplotype sharing matrix (`--hsm`) uses phased matches across windows.

| Option | Default | Description |
| --- | --- | --- |
| `--pca INT` | — | Number of principal components |
| `--hsm` | Off | Export the full haplotype sharing kernel in GCTA format |
| `--hsm-svd INT` | — | Number of eigenvectors from sharing profiles |
| `--hsm-sqrt` | Off | Square-root sharing profiles before centering |
| `--hsm-matches INT` | `16` | Maximum tied haplotypes per maximal sharing match |
| `--hsm-gap INT` | `1000000` | Break sharing runs across gaps larger than this many bp |
| `--map FILE` | — | Genetic map for HSM weights and covered lengths |
| `--grm` | Off | Estimate the GRM |
| `--projection PREFIX` | — | Project onto a saved PCA model |
| `--loadings` | Off | Save loadings, frequencies, and PCA identity metadata |
| `--raw` | Off | Write PC values without FID/IID columns |
| `--duplicate-fid` | Off | Use sample ID as FID instead of `0` |
| `--grm-no-center` | Off | Disable GRM centering and Gower scaling, requires `--grm` |
| `--chunk INT` | `4096` | Target cluster alleles per calculation block |
| `--power INT` | `11` | Randomized PCA power iterations |
| `--seed INT` | `42` | Random seed |

PCA writes `.vec` and `.val`. `--loadings` also writes `.load`,
`.freq`, and `.pca`, and requires assignment `.ref` files. Projection
writes `.proj.vec` and checks the saved model's ordered references and
file partitioning. Query samples may differ.
PCA is approximate. Training and reprojected coordinates can differ slightly.
Increase `--power` when tighter agreement is needed.

GRM writes `.grm.bin`, `.grm.N.bin`, and `.grm.id`. Each window
contributes `max(observed clusters - 1, 0)` to the contrast count.
Counts are constant across pairs under mean imputation. SNP-count-weighted GRM
merging is not supported. The log records normalization, counts, dimensions, and timings.

### Haplotype sharing matrix

```sh
hapla struct --clusters chr{1..22} --hsm-svd 20 --threads 8 --out result
hapla struct --clusters chr{1..22} --hsm --hsm-svd 20 --hsm-sqrt --threads 8 --out result
```

HSM compares matching cluster-label runs across phased, ordered, nonoverlapping
windows. Matches exclude the individual's own haplotypes and use
[PBWTpaint](https://doi.org/10.1038/s41467-025-57601-3) weights with physical distance
in Mb, or genetic distance in Morgans with `--map`. The map changes both match
weights and window lengths. `--hsm-gap` remains in bp. Missing labels, constant
windows, chromosome boundaries, and large gaps break matches. Samples without
matched coverage are rejected.

Profiles are combined across chromosomes, normalized by matched coverage, and
centred. Linear profiles are the default. `--hsm-sqrt` applies square roots before
centering for both outputs.

- `--hsm-svd K` writes unit-norm `.hsm.vec` and kernel `.hsm.val`.
  Multiply eigenvectors by `sqrt(eigenvalue)` for kernel PCA scores.
- `--hsm` writes the full PSD kernel with trace `N - 1`, up to rounding, as GCTA
  float32 `.hsm.grm.bin` and `.hsm.grm.id`. No SNP-count file is written.

Both flags share one matching pass. `.hsm.cov` reports matched lengths and
fractions. The log records the transformation and `Gower scale`. SVD uses sparse
profiles and temporary storage. Full matrix export requires quadratic disk space.
HSM runs separately from PCA, GRM, and projection. Matches are not verified IBD,
and heritability use needs separate validation.

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
| `--loo` | Off | Leave both haplotypes out of P when updating their individual's Q |
| `--iter INT` | `1000` | Maximum outer fitting iterations |
| `--tole FLOAT` | `1e-9` | Normalized likelihood/objective tolerance, or parameter RMSE with `--loo` |
| `--p-prior FLOAT` | `0` | P pseudocount mass per window and ancestry, 0 disables shrinkage |
| `--batches INT` | `16` | Initial mini-batches, reduced during fitting |
| `--check INT` | `5` | Iterations between progress reports and non-LOO convergence checks |
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
fitting Q. The default unsupervised initializer uses SVD/ALS. Opt-in P shrinkage
uses pooled cluster frequencies within each window and cannot update a fixed
projection P. With shrinkage, convergence uses the regularized objective while
the log also records the likelihood. Timings cover each `--check` interval and
any final partial interval. Warm-up is separate.

`--loo` subtracts each individual's current expected cluster counts before
updating its Q. The pooled P prior also excludes that individual's observed
haplotypes. Private clusters and windows without other observations provide
no ancestry information. Shared P still uses all individuals. Projection
cannot be combined with `--loo`.

LOO uses full-data updates with half-step damping, without mini-batches,
quasi-Newton acceleration, or warm-up. Convergence uses the larger P/Q RMSE
per update. The likelihood and regularized objective are diagnostic and may
decrease. This is a current-count correction, not a separate model refit for
each excluded individual. Initialization still uses the full cohort.
Without LOO, `--tole` applies to log likelihood or objective divided by
2 × samples × cluster alleles.

EM uses float64 sample tiles with an 8 MiB target for Q scratch, subject to a
minimum of one sample per window partition. P counts reuse the existing output
buffers. Parameter arrays and input storage still scale with the dataset.

Outputs use `<out>.K<K>.s<seed>`, or `<out>.proj.K<K>.s<seed>` for projection.
They include `.Q`, `.ids`, and `.log`. Fitted frequencies use `.P` for one input
or `.chr1.P`, `.chr2.P`, etc. plus `.plist` for multiple inputs. P-file lists
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
| `--loo` | Off | Leave both haplotypes out of P for Q refinement only |
| `--iter INT` | `10` | Maximum refinement updates |
| `--tole FLOAT` | `1e-5` | Objective improvement per observation, or parameter RMSE with `--loo` |
| `--p-prior FLOAT` | `10` | P pseudocount mass per window and ancestry |
| `--q-prior FLOAT` | `10` | Q pseudocount mass per individual |
| `--map FILE` | — | Genetic map, requires `--dating` or `--time` |
| `--time FLOAT` | — | Fixed admixture time in generations |
| `--dating` | Off | Fit one time using supplied P/Q, then decode |
| `--date-samples FILE` | All | Sample IDs used for dating, one per line |
| `--date-min FLOAT` | `1` | Lower search bound in generations |
| `--date-max FLOAT` | `500` | Upper search bound in generations |
| `--date-jackknife` | Off | Leave out each chromosome for approximate date uncertainty |
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
| `--save-posteriors` | Off | Save call support for the mean-posterior calls |
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

Both commands use the same P pseudocount rule. `admix` targets pooled cluster
frequencies to stabilize estimation. `fatash` targets the supplied
ancestry-specific P to keep local refinement near the initial model.

`--loo` removes each individual's posterior cluster counts from the P used
for its Q update. Shared P and final local ancestry decoding still use the
full cohort. The supplied P/Q prior targets stay fixed, including any
influence from the individual in the original admix fit. Unsupported
leave-out emissions are neutral. LOO uses half-step damping and stops on
the largest RMSE across Q and chromosome P arrays. Its likelihood and
objective are diagnostic and may decrease. It requires refinement and
cannot be combined with `--fixed-model`.

LOO is a current-count correction, not a full leave-one-out model refit.
It adds one posterior pass when a whole chromosome cohort fits in one batch,
or two when posteriors must be recomputed in bounded batches. Individual P
matrices are never stored.

Without LOO, convergence is checked after each update using mean-alpha log likelihood minus
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
etc. and a `.plist`. Reuse the same alpha and decoding options with saved
P/Q and `--fixed-model` to reproduce an analysis. With a map, reuse `--map` and
the fitted `--time` instead of alpha.

### Genetic distances and dating

Both `struct` and `fatash` interpolate the supplied map internally. Use the
same genome build as the clusters, with increasing bp and nondecreasing cM.
The whitespace-separated map contains all required chromosomes:

```text
CHR BP CM
1 100 0
1 1000000 1.2
2 100 0
2 1000000 0.9
```

The header is optional. Headered SHAPEIT `pos chr cM` and chromosome-labelled
HapMap columns are also accepted, including `.gz` files. `chr1` and `1` match.
Linear interpolation covers window endpoints and midpoints. Coordinates outside
map coverage are rejected. Flat intervals retain zero genetic distance.

With `--map`, FATASH uses `T = exp(-t d) I + (1 - exp(-t d)) Q` for adjacent
window midpoints, where `d` is in Morgans. `--time` fixes `t` while retaining
the selected P/Q refinement mode. Genetic transitions require `--block 1`
and replace the alpha ensemble. They cannot use `--alpha` or `--simple`.

```bash
hapla fatash --clusters chr{1..22} --pfile fit.chr{1..22}.P --qfile fit.Q \
    --map genetic.map --dating --date-samples admixed.ids --out dated
hapla fatash --clusters chr{1..22} --pfile fit.chr{1..22}.P --qfile fit.Q \
    --map genetic.map --time 20 --fixed-model --out paths
hapla struct --clusters chr{1..22} --map genetic.map --hsm-svd 20 --out sharing
```

Dating holds input P/Q fixed and fits one shared single-pulse time for the
selected cohort. It integrates over ancestry paths, without counting decoded
switches. All samples are decoded afterwards. Use a cohort with a common
admixture history. `--dating` cannot use `--baum-welch`, `--loo`, or `--medians`.

`.date` records generations, search status, and likelihood. A grid followed by
bounded refinement checks the best region in log time. Flat likelihoods are
rejected and boundary estimates are labelled. `--date-jackknife` requires at
least three chromosomes and adds approximate 95% limits conditional on input
P/Q. Limits are omitted when a replicate is flat or hits a search bound.

Likelihood passes score cached chromosome labels directly, with two state
vectors per worker. Jackknifing adds a fit per chromosome. Maps and transition
tables are outside the batch memory budget. Residual LD, phasing errors, and
incorrect P/Q can bias the effective HMM date. Simulation calibration is needed
before interpreting it as a historical admixture event.

## hapla deconv

Expand decoded local-ancestry paths into one ancestry-specific copy per
individual and ancestry. The command reads the same ordered cluster inputs used
by `fatash`, plus a matching list of `.path` files. It can write expanded Hapla
cluster bundles for downstream `admix`, `struct`, or `eval`, masked phased BCFs
for SNP-based analyses, or both. The input genotype files must be the phased,
biallelic files from which the matching cluster assignments were made.

```bash
printf '%s\n' lai.chr{1..22}.path > paths.txt
printf '%s\n' input.chr{1..22}.bcf > genotypes.txt
hapla deconv --clusters chr{1..22} --path-filelist paths.txt --K 4 \
    --format both --bcf-filelist genotypes.txt --include-original \
    --min-call-support 0.95 --min-tract-windows 6 --out deconv/lai
```

Cluster output is written as `<out>.<input-suffix>.bca`, `.ids`, `.win`, and
`.ref.json`; `<out>.filelist` lists those prefixes in input order. BCF output
uses matching `<out>.<input-suffix>.bcf` files and `<out>.bcfs` lists them.
Each retained copy is named `SAMPLE_K`, where `K` is its zero-based ancestry
label. `--include-original` retains the unmasked samples before these copies.

| Option | Default | Description |
| --- | --- | --- |
| `--path-filelist FILE` | Required | Ordered decoded-path files |
| `--K INT` | Required | Number of ancestry labels |
| `--format clusters\|bcf\|both` | `clusters` | Requested output type |
| `--bcf-filelist FILE` | — | Ordered phased genotype files for BCF output |
| `--min-call-support FLOAT` | Off | Mask calls below this decoded-call support |
| `--min-tract-windows INT` | Off | Mask retained tracts shorter than this length |
| `--homozygous-only` | Off | Mask windows whose two haplotypes differ in ancestry |
| `--min-fraction FLOAT` | `0.05` | Drop copies with less retained data |
| `--include-original` | Off | Include unmasked originals in the output |
| `--save-filtered-paths` | Off | Write the filtered paths used for output |

Filtering is applied in this order: call support, homozygous-ancestry masking,
then short-tract masking. A short tract is masked rather than reassigned. Call
support is useful for conservative masking, but is a decoded-call diagnostic and
not a calibrated confidence interval.

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

For source tests, build the native kernels with `python setup.py build_ext --inplace`
and run `python tests/run.py`. The default tests the checkout. `--installed`
tests the installed package and rejects a checkout on the import path.

[CI](.github/workflows/ci.yml) checks Python style and Cython warnings, validates
the wheel and source archive, and runs the installed tests with one and two
native threads. Pushes to `main` and pull requests use Ubuntu with Python 3.14.
Manual runs add Ubuntu and macOS 15 wheels for Python 3.10, 3.12, and 3.14.

Use short module summaries and camelCase helper names. Use `#####` for sections,
`###` above functions and test classes, and `#` inside functions. Keep two blank
lines between top-level definitions and a blank line before comments unless
they start an indented block. Tests use `unittest`, fixed seeds, and small shared fixtures.

## Citation

- **hapla cluster**: [Nature Communications](https://doi.org/10.1038/s41467-024-55477-3), [preprint](https://doi.org/10.1101/2024.04.30.24306654)
- **hapla admix**: [HGG Advances](https://doi.org/10.1016/j.xhgg.2026.100561), [preprint](https://doi.org/10.1101/2025.09.02.673718)
