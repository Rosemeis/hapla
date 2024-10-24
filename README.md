# hapla (v0.12)
***hapla*** is a framework for performing window-based haplotype clustering in phased genotype data. The inferred haplotype cluster alleles can be used to infer fine-scale population structure, perform polygenic prediction and haplotype cluster based association studies.

### Citation
[medRxiv preprint](https://doi.org/10.1101/2024.04.30.24306654)

## Installation
```bash
# Build and install via PyPI
pip install hapla

# or download source and install via pip
git clone https://github.com/Rosemeis/hapla.git
cd hapla
pip install .

# or download source and install in new Conda environment
git clone https://github.com/Rosemeis/hapla.git
conda env create -f environment.yml
conda activate hapla

# The "hapla" main caller will now be available
```

## Quick start
***hapla*** contains the following subcommands at this moment:
- cluster
- struct
- predict
- admix
- fatash


### Haplotype clustering
***hapla cluster***\
Window-based haplotype clustering in a phased VCF/BCF.
```bash
# Cluster haplotypes in a chromosome with fixed window size (8 SNPs)
hapla cluster --bcf data.chr1.bcf --fixed 8 --threads 16 --out hapla.chr1
# Saves inferred haplotype cluster assignments in binary hapla format
#	- hapla.chr1.bca
#	- hapla.chr1.ids
#	- hapla.chr1.win
```
`hapla cluster` outputs three files. A **.bca**-file (binary cluster assignments), which stores the cluster assignments as *unsigned char*s, a **.ids**-file with sample names and a **.win**-file with information about the genomic windows.

```bash
# Cluster haplotypes in a chromosome with fixed size and overlapping windows (step size 4)
hapla cluster --bcf data.chr1.bcf --fixed 8 --step 4 --threads 16 --out hapla.chr1

# Cluster haplotypes in all chromosomes and save output path in a filelist
for c in {1..22}
do
	hapla cluster --bcf data.chr${c}.bcf --fixed 8 --threads 16 --out hapla.chr${c}
	realpath hapla.chr${c} >> hapla.filelist
done
```

Optionally, the haplotype cluster alleles can be saved in binary PLINK format (**.bed**, **.bim**, **.fam**) for ease of use with other software. Note that window information needs to be inferred from **.bim**-file for downstream analyses in this case.
```bash
hapla cluster --bcf data.chr1.bcf --threads 16 --out hapla.chr1 --plink
# Saves inferred haplotype cluster alleles in a binary PLINK format
#	- hapla.chr1.bed
#	- hapla.chr1.bim
#	- hapla.chr1.fam
```

### GRM estimation and population structure inference
***hapla struct***\
Estimate genome-wide relationship matrix (GRM) and infer population structure using the haplotype cluster alleles.
```bash
# Construct genome-wide relationship matrix (GRM)
hapla struct --filelist hapla.filelist --threads 16 --grm --out hapla
# Saves the GRM in binary GCTA format (float)
#	- hapla.grm.bin
#	- hapla.grm.N.bin
#	- hapla.grm.id

# Perform PCA on all chromosomes (genome-wide) using filelist and extract top 20 eigenvectors
hapla struct --filelist hapla.filelist --threads 16 --pca 20 --out hapla
# Saves eigenvalues and eigenvectors in text-format
#	- hapla.eigenvecs
#	- hapla.eigenvals

# Or perform PCA on a single chromosome and extract top 20 eigenvectors
hapla struct --clusters hapla.chr1 --threads 16 --pca 20 --out hapla.chr1
# Saves eigenvalues and eigenvectors in text-format
#	- hapla.chr1.eigenvecs
#	- hapla.chr1.eigenvals

# A faster randomized SVD approach can also be utilized for large datasets (>5,000 individuals)
hapla struct --filelist hapla.filelist --threads 16 --pca 20 --randomized --out hapla
```

### Predict haplotype cluster assignments
***hapla predict***\
Predict haplotype cluster assignments using pre-computed cluster medians in new haplotypes. All SNPs must be overlapping.
```bash
# Cluster haplotypes in a chromosome with 'hapla cluster' and save cluster medians (--medians)
hapla cluster --bcf ref.chr1.bcf --fixed 8 --threads 16 --out ref.chr1 --medians
# Saves haplotype cluster medians (besides standard binary hapla format)
#	- ref.chr1.bcm
#	- ref.chr1.wix
#	- ref.chr1.hcc

# Predict assignments in a set of new haplotypes using haplotype cluster medians
hapla predict --bcf new.chr1.bcf --threads 16 --out new.chr1 --ref ref.chr1
# Saves predicted haplotype cluster assignments in binary hapla format
#	- new.chr1.bca
#	- new.chr1.ids
#	- new.chr1.win
```
Using `--medians` in `hapla cluster` outputs three extra files. A **.bcm**-file (binary cluster medians), which stores the cluster medians as *unsigned char*s, a **.wix**-file with window index information and a **.hcc**-file with haplotype cluster counts. All files are needed to predict haplotype clusters in a new set of haplotypes.

### Admixture estimation
***hapla admix***\
Estimate ancestry proportions and ancestral haplotype cluster frequencies with a pre-specified number of sources (K). Using a modified ADMIXTURE model for haplotype clusters.
```bash
# Estimate ancestry proportions assuming K=3 ancestral sources for a single chromosome
hapla admix --clusters hapla.chr1 --K 3 --seed 1 --threads 16 --out hapla.chr1
# Saves Q matrix and P matrix in a text-file format
#	- hapla.chr1.K3.s1.Q
#	- hapla.chr1.K3.s1.P

# Estimate ancestry proportions assuming K=3 ancestral sources using filelist with all chromosomes
hapla admix --filelist hapla.filelist --K 3 --seed 1 --threads 16 --out hapla
# Saves Q matrix in a text-file and separate text-files of P matrices for each file
#	- hapla.K3.s1.Q
#	- hapla.K3.s1.file{1..22}.P
```

### Local ancestry inference
***hapla fatash***\
Infer local ancestry tracts using the admixture estimation in a hidden markov model. Using a modified fastPHASE model for haplotype clusters.
```bash
# Infer local ancestry tracts for a single chromosome (posterior decoding)
hapla fatash --clusters hapla.chr1 --qfile hapla.chr1.K3.s1.Q --pfile hapla.chr1.K3.s1.P --threads 16 --out hapla.chr1
# Saves posterior decoding path in text-format
#	- hapla.chr1.path

# Infer local ancestry tracts using filelist with all chromosomes (Viterbi decoding)
for c in {1..22}; do realpath hapla.chr1.K3.s1.file${c}.P >> hapla.K3.s1.pfilelist; done
hapla fatash --filelist hapla.filelist --qfile hapla.K3.s1.Q --pfilelist hapla.K3.s1.pfilelist --threads 16 --out hapla --viterbi
# Saves Viterbi decoding paths in text-files
#	- hapla.file{1..22}.path
```
