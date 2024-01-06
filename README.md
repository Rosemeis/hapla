# hapla (v0.4)
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
***hapla*** contains the following subcommands:
- cluster
- struct
- predict


### Haplotype clustering
**hapla cluster**
Haplotype clustering is performed on a phased VCF/BCF.
```bash
# Cluster haplotypes in a chromosome with default window size (10 SNPs)
hapla cluster --bcf file.chr1.bcf --threads 32 --out hapla.chr1
# Saves inferred haplotype cluster alleles in a binary NumPy file ("hapla.z.npy")

# Cluster haplotypes in all chromosomes and save output path in a filelist (needed in downstream analyses)
for c in {1..22}
do
	hapla cluster --bcf file.chr${c}.bcf --threads 32 --out hapla.chr${c}
	realpath hapla.chr${c}.z.npy >> hapla.filelist
done
```

Optionally, the haplotype cluster alleles can be saved in binary PLINK format.
```bash
hapla cluster --bcf file.chr1.bcf --threads 32 --out hapla.chr1 --plink
# Saves inferred haplotype cluster alleles in a binary PLINK format
#	- hapla.chr1.bed
#	- hapla.chr1.bim
#	- hapla.chr1.fam
```

### Population structure inference
**hapla struct**
Inferring population structure using the haplotype cluster alleles.
```bash
# Perform PCA on all chromosomes (genome-wide)
hapla struct --filelist hapla.filelist --threads 32 --out hapla
# Saves eigenvalues and eigenvectors in text-format ("hapla.eigenvec" and "hapla.eigenval")

# Or perform PCA on a single chromosome
hapla struct --clusters hapla.chr1.z.npy --threads 32 --out hapla.chr1
# Saves eigenvalues and eigenvectors in text-format ("hapla.chr1.eigenvec" and "hapla.chr1.eigenval")

# A randomized SVD approach can also be utilized for very large datasets
hapla struct --filelist hapla.filelist --threads 32 --randomized --out hapla
# Saves eigenvalues and eigenvectors in text-format ("hapla.eigenvec" and "hapla.eigenval")

# A genome-wide relationship matrix (GRM) can also be constructed (not recommended for very large datasets)
hapla struct --filelist hapla.filelist --threads 32 --grm --out hapla
# Saves the GRM in text-format ("hapla.grm")
```
