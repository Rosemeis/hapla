# hapla
***hapla*** is a framework for performing window-based haplotype clustering in phased genotype data. The inferred haplotype cluster alleles can be used to infer fine-scale population structure, perform polygenic prediction and SNP-based or haplotype cluster-based association studies.

### Citation
Please cite our preprint on *BioRxiv*: Work in progress

## Install and build
```bash
git clone https://github.com/Rosemeis/hapla.git
cd hapla
pip3 install .

# The "hapla" main caller will now be available
```

### Tutorial
A detailed tutorial on how to use all features in ***hapla*** with a toy dataset can be found [here](https://github.com/Rosemeis/hapla).

## Quick start
***hapla*** contains the following subcommands:
- cluster
- pca
- predict


**hapla cluster** (Haplotype clustering)

Haplotype clustering is performed on a phased VCF/BCF.
```bash
# Cluster haplotypes in a chromosome with default window size (100 SNPs)
hapla cluster --bcf file.chr1.bcf --threads 32 --out hapla.chr1
# Saves clustering in a binary NumPy file ("hapla.z.npy")

# Cluster haplotypes in all chromosomes and save output path in a filelist (needed in downstream analyses)
for c in {1..22}
do
	hapla cluster --bcf file.chr${c}.bcf --threads 32 --out hapla.chr${c}
	realpath hapla.chr${c}.z.npy >> hapla.filelist
done
```


**hapla pca** (Population structure inference)
PCA is performed on the output from the haplotype clustering.
```bash
# Perform PCA on all chromosomes (genome-wide)
hapla pca --filelist hapla.filelist --threads 32 --out hapla
# Saves eigenvalues and eigenvectors in text-format ("hapla.eigenvec" and "hapla.eigenval")

# Or perform PCA on a single chromosome (not recommended)
hapla pca --clusters hapla.chr1.z.npy --threads 32 --out hapla.chr1
# Saves eigenvalues and eigenvectors in text-format ("hapla.chr1.eigenvec" and "hapla.chr1.eigenval")

# A randomized SVD approach can also be utilized for very large datasets
hapla pca --filelist hapla.filelist --threads 32 --randomized --out hapla
# Saves eigenvalues and eigenvectors in text-format ("hapla.eigenvec" and "hapla.eigenval")

# A genome-wide relationship matrix (GRM) can also be constructed (not recommended for large datasets)
hapla pca --filelist hapla.filelist --threads 32 --grm --out hapla
# Saves the GRM in text-format ("hapla.grm")
```
