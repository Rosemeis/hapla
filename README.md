# hapla
***hapla*** is a software for performing window-based haplotype clustering in phased genotype data. The inferred haplotype clusters can be used to infer fine-scale population structure and perform association testing.

### Citation
Please cite our preprint on *BioRxiv*: Work in progress

### Dependencies
The ***hapla*** software relies on the following Python libaries that you can install using *conda* or *pip*:

- cython
- cyvcf2
- numpy
- scipy

You can create an environment with *conda* or install libraries with *pip* as follows:
```
# conda
conda env create -f environment.yml

# pip
pip3 install -r requirements.txt
```

## Install and build
```bash
git clone https://github.com/Rosemeis/hapla.git
cd hapla
python3 setup.py build_ext --inplace; pip3 install -e .

# The "hapla" main caller will now be available
```

### Tutorial
A detailed tutorial on how to use all features in ***hapla*** with a toy dataset can be found [here](https://github.com/Rosemeis/hapla).

## Quick start
***hapla*** contains the following subcommands:

- cluster
- pca
- regress
- predict
- prs
- split

**hapla cluster** (Haplotype clustering)

Haplotype clustering is performed on a phased VCF/BCF.
```bash
# Cluster haplotypes in a chromosome with default window size (100 SNPs)
hapla cluster --bcf file.chr1.bcf --threads 20 --out hapla.chr1
# Saves clustering in a binary NumPy file ("hapla.z.npy")

# Cluster haplotypes in all chromosomes and save output in a filelist
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

# Or perform PCA on a single chromosome
hapla pca --clusters hapla.chr1.z.npy --threads 20 --out hapla.chr1
# Saves eigenvalues and eigenvectors in text-format ("hapla.chr1.eigenvec" and "hapla.chr1.eigenval")
```

**hapla regress** (Whole-genome regression and association testing)

The association testing in *hapla* is based on the *regenie* software, however it is haplotype cluster-based instead of SNP-based. The eigenvectors from the PCA are directly used as input to correct for population structure. Additional covariates can be provided as a simple text file only containing values (one line per individual). The provided phenotype file is expected to be a single column text file of values (one line per individual). Both files with covariates and phenotypes should have no header and no extra columns of unnessecary information, such that the ordering of individuals is expected to follow the VCF/BCF file used in the haplotype clustering.
```bash
hapla regress --filelist hapla.filelist --eigen hapla.eigenvec --pheno trait.pheno --threads 20 --out hapla.trait
# Saves association tests of haplotype clusters in text-format ("hapla.trait.assoc)
```

**hapla predict** (Predict haplotype clusters in new target file from pre-computed cluster medians)
```bash
# Cluster haplotypes in reference dataset and save cluster medians
for c in {1..22}
do
	hapla cluster --bcf ref.chr${c}.bcf --save_medians --threads 20 --out ref.chr${c}
	# Saves cluster medians in a binary NumPy file (ref.chr${c}.medians.npy) 
	realpath ref.chr${c}.z.npy >> ref.filelist
done

# Predict haplotype clusters in new target dataset (assumes same SNP set, missingness allowed)
for c in {1..22}
do
	hapla predict --bcf target.chr${c}.bcf --medians ref.chr${c}.medians.npy --threads 20 --out target.chr${c}
	realpath target.chr${c}.z.npy >> target.filelist
done
```

**hapla prs** (Polygenic risk score estimation using summary statistics)

Polygenic risk scores are estimated using the haplotype clustering and summary statistics of genome-wide association testing from a reference dataset.
```bash
hapla prs --filelist target.filelist --assoc reference.trait.assoc --threads 20 --out target.trait
# Saves polygenic risk scores in text-format ("target.trait.sumstats.prs")
```

**hapla split** (LD-based window generation for haplotype clustering)

An optimal LD splitting can be performed per chromosome to perform haplotype clustering in inferred LD blocks rather than a fixed window size. The method is a reimplementation of the algorithm in this article [Optimal linkage disequilibrium splitting](https://doi.org/10.1093/bioinformatics/btab519). Please cite the original article if this method has been used!
```bash
hapla split --bcf file.chr1.bcf --threads 20 --out hapla.chr1
# Saves the indices of the optimal splitting ("hapla.chr1.windows")

# The window indices can now directly be used as input to "hapla cluster"
hapla cluster --bcf file.chr1.bcf --windows hapla.chr1.windows --threads 20 --out hapla.chr1
```
