# hapla
***hapla*** is a framework for performing window-based haplotype clustering in phased genotype data. The inferred haplotype cluster alleles can be used to infer fine-scale population structure, perform polygenic prediction and SNP-based or haplotype-based association studies.

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
- regress
- asso

**hapla cluster** (Haplotype clustering)

Haplotype clustering is performed on a phased VCF/BCF.
```bash
# Cluster haplotypes in a chromosome with default window size (500 SNPs)
hapla cluster --bcf file.chr1.bcf --threads 20 --out hapla.chr1
# Saves clustering in a binary NumPy file ("hapla.z.npy")

# Cluster haplotypes in all chromosomes and save output path in a filelist
for c in {1..22}
do
	hapla cluster --bcf file.chr${c}.bcf --threads 20 --out hapla.chr${c}
	realpath hapla.chr${c}.z.npy >> hapla.filelist
done
```

**hapla pca** (Population structure inference)
PCA is performed on the output from the haplotype clustering.
```bash
# Perform PCA on all chromosomes (genome-wide)
hapla pca --filelist hapla.filelist --threads 20 --out hapla
# Saves eigenvalues and eigenvectors in text-format ("hapla.eigenvec" and "hapla.eigenval")

# Or perform PCA on a single chromosome (not recommended)
hapla pca --clusters hapla.chr1.z.npy --threads 20 --out hapla.chr1
# Saves eigenvalues and eigenvectors in text-format ("hapla.chr1.eigenvec" and "hapla.chr1.eigenval")
```

**hapla regress** (Whole-genome regression and association testing)
The polygenic prediction and association testing in *hapla* are based on the *REGENIE* software, however it is haplotype cluster-based instead of SNP-based. The eigenvectors from the PCA are directly used as input to correct for population structure. Additional covariates can be provided as a simple text file only containing values (one line per individual). The provided phenotype file is expected to be a single column text file of values (one line per individual). Both files with covariates and phenotypes should have no header and no extra columns of unnessecary information, such that the ordering of individuals is expected to follow the VCF/BCF file used in the haplotype clustering.
```bash
hapla regress --filelist hapla.filelist --eigen hapla.eigenvec --pheno trait.pheno --threads 20 --out hapla.trait
# Saves association tests of haplotype clusters in text-format ("hapla.trait.assoc)
```
