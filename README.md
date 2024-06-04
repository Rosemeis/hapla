# hapla (v0.9)
***hapla*** is a framework for performing window-based haplotype clustering in phased genotype data. The inferred haplotype cluster alleles can be used to infer fine-scale population structure, perform polygenic prediction and haplotype cluster based association studies.

### Citation
[medRxiv preprint](https://doi.org/10.1101/2024.04.30.24306654)

## Install and build
```bash
git clone https://github.com/Rosemeis/hapla.git
cd hapla
pip3 install .

# The "hapla" main caller will now be available
```

### Tutorial
Coming soon.

## Quick start
***hapla*** contains the following subcommands at this moment:
- cluster
- struct
- predict
- admix
- fatash


### Haplotype clustering
**hapla cluster**
Window-based haplotype clustering in a phased VCF/BCF.
```bash
# Cluster haplotypes in a chromosome with fixed window size (8 SNPs)
hapla cluster --bcf pop.chr1.bcf --fixed 8 --threads 16 --out hapla.chr1
# Saves inferred haplotype cluster assignments in a binary NumPy file ("hapla.chr1.z.npy")

# Cluster haplotypes in all chromosomes and save output path in a filelist
for c in {1..22}
do
	hapla cluster --bcf pop.chr${c}.bcf --fixed 8 --threads 16 --out hapla.chr${c}
	realpath hapla.chr${c}.z.npy >> hapla.filelist
done
```

Optionally, the haplotype cluster alleles can be saved in binary PLINK format for ease of use with other software. Be aware that one allele is removed from each window to prevent identifiability issues (rarest allele removed).
```bash
hapla cluster --bcf pop.chr1.bcf --threads 16 --out hapla.chr1 --plink
# Saves inferred haplotype cluster alleles in a binary PLINK format
#	- hapla.chr1.bed
#	- hapla.chr1.bim
#	- hapla.chr1.fam
```

### GRM estimation and population structure inference
**hapla struct**
Estimate genome-wide relationship matrix (GRM) and infer population structure using the haplotype cluster alleles.
```bash
# Construct genome-wide relationship matrix (GRM)
hapla struct --filelist hapla.filelist --threads 16 --grm --out hapla --iid pop.samples
# Saves the GRM in binary GCTA format (float)
#	- hapla.grm.bin
#	- hapla.grm.N.bin
#	- hapla.grm.id
# Requires a file with sample names ("--iid")

# Perform PCA on all chromosomes (genome-wide) using filelist and extract top 10 eigenvectors
hapla struct --filelist hapla.filelist --threads 16 --pca 10 --out hapla
# Saves eigenvalues and eigenvectors in text-format ("hapla.eigenvec" and "hapla.eigenval")

# Or perform PCA on a single chromosome and extract top 10 eigenvectors
hapla struct --clusters hapla.chr1.z.npy --threads 16 --pca 10 --out hapla.chr1
# Saves eigenvalues and eigenvectors in text-format ("hapla.chr1.eigenvec" and "hapla.chr1.eigenval")

# A randomized SVD approach can also be utilized for very large datasets
hapla struct --filelist hapla.filelist --threads 16 --pca 10 --randomized --out hapla
# Saves eigenvalues and eigenvectors in text-format ("hapla.eigenvec" and "hapla.eigenval")
```

### Predict haplotype cluster assignments
**hapla predict**
Predict haplotype cluster assignments using pre-computed cluster medians in new haplotypes. All SNPs must be overlapping.
```bash
# Cluster haplotypes in a chromosome with 'hapla cluster' and save cluster medians
hapla cluster --bcf pop.chr1.bcf --fixed 8 --threads 16 --out hapla.chr1 --medians

# Predict assignments in new haplotypes
hapla predict --bcf new.chr1.bcf --threads 16 --out new.chr1 --medians hapla.chr1.medians.npz
# Saves predicted haplotype cluster assignments in a binary NumPy file ("new.chr1.z.npy")
```

### Admixture estimation
**hapla admix**
Estimate ancestry proportions and ancestral haplotype cluster frequencies with a pre-specified number of sources (K). Using a modified ADMIXTURE model for haplotype clusters.
```bash
# Estimate ancestry proportions assuming K=3 ancestral sources for a single chromosome
hapla admix --clusters hapla.chr1.z.npy --K 3 --seed 1 --threads 16 --out hapla.chr1
# Saves Q matrix in a text-file and P matrix in a binary NumPy file
#	- hapla.chr1.K3.s1.Q
#	- hapla.chr1.K3.s1.P.npy

# Estimate ancestry proportions assuming K=3 ancestral sources using filelist with all chromosomes
hapla admix --filelist hapla.filelist --K 3 --seed 1 --threads 16 --out hapla
# Saves Q matrix in a text-file and P matrix in a binary NumPy file
#	- hapla.K3.s1.Q
#	- hapla.K3.s1.P.npy
```

### Local ancestry inference
**hapla fatash**
Infer local ancestry tracts unsupervised using the admixture estimation in a hidden markov model. Using a modified fastPHASE model for haplotype clusters.
```bash
# Infer local ancestry tracts for a single chromosome
hapla fatash --clusters hapla.chr1.z.npy --qfile hapla.chr1.K3.s1.Q --pfile hapla.chr1.K3.s1.P.npy --threads 16 --out hapla.chr1
# Saves posterior decoding path in text-format ("hapla.chr1.path")

# Infer local ancestry tracts using filelist with all chromosome and optimize alpha rates in transition matrix
hapla fatash --filelist hapla.filelist --qfile hapla.K3.s1.Q --pfile hapla.K3.s1.P.npy --threads 16 --out hapla --optim --save-alpha
# Saves path and alpha rates in a text-file
#	- hapla.path
#	- hapla.alpha
```
