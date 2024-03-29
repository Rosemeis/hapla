# hapla (v0.6)
***hapla*** is a framework for performing window-based haplotype clustering in phased genotype data. The inferred haplotype cluster alleles can be used to infer fine-scale population structure, perform polygenic prediction and haplotype cluster based association studies.

### Citation
Coming soon.

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
- score


### Haplotype clustering
**hapla cluster**
Window-based haplotype clustering in a phased VCF/BCF.
```bash
# Cluster haplotypes in a chromosome with default window size (8 SNPs)
hapla cluster --bcf pop.chr1.bcf --threads 16 --out hapla.chr1
# Saves inferred haplotype cluster alleles in a binary NumPy file ("hapla.z.npy")

# Cluster haplotypes in all chromosomes and save output path in a filelist
for c in {1..22}
do
	hapla cluster --bcf pop.chr${c}.bcf --threads 16 --out hapla.chr${c}
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
# Saves the GRM in binary GCTA format
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
