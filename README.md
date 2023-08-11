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


**hapla regress** (Polygenic prediction / Whole-genome regression)
Polygenic prediction is based on the ideas of the *REGENIE* software, however it is haplotype cluster-based instead of SNP-based. The eigenvectors from the PCA are directly used as input to correct for population structure. Additional covariates can be provided as a simple text file only containing values (one line per individual). The provided phenotype file is expected to be a single column text file of values (one line per individual). Both files with covariates and phenotypes should have no header and no extra columns of unnessecary information, such that the ordering of individuals is expected to follow the VCF/BCF file used in the haplotype clustering.
```bash
# Perform polygenic prediction using default 10-fold CV ridge regressions
hapla regress --filelist hapla.filelist --eigen hapla.eigenvec --pheno trait.pheno --threads 32 --out hapla.trait
# Saves polygenic prediction and LOCO predictions in text-format ("hapla.trait.pred" and "hapla.trait.loco")

# Perform polygenic prediction using LOOCV (recommended for small and medium datasets)
hapla regress --filelist hapla.filelist --eigen hapla.eigenvec --pheno trait.pheno --threads 32 --folds 0 --out hapla.trait
# Saves polygenic prediction and LOCO predictions in text-format ("hapla.trait.pred" and "hapla.trait.loco")
```


**hapla asso** (Association testing)
Association testing can be performed on either SNP level or haplotype cluster allele level. For SNP level association testing usually the imputed SNP set would be used here. The same eigenvectors and covariates provided for the polygenic prediction *must* be provided here again as well as the LOCO (leave-one-chromosome-out) predictions for the previous step.
```bash
# Perform association testing for SNP set
hapla asso --filelist geno.filelist --eigen hapla.eigenvec --pheno trait.pheno --threads 32 --loco hapla.trait.loco --out hapla.trait
# Saves summary statistics of association tests for SNPs ("hapla.trait.snp.assoc")

# Perform association testing for haplotype cluster alleles
hapla asso --filelist hapla.filelist --eigen hapla.eigenvec --pheno trait.pheno --threads 32 --loco hapla.trait.loco --out hapla.trait
# Saves summary statistics of association tests for haplotype cluster alleles ("hapla.trait.haplo.assoc")

### The appropriate test is automatically derived from the filelist provided
```
