# hapla (v0.32.1)
***hapla*** is a framework for performing window-based haplotype clustering in phased genotype data. The inferred haplotype cluster alleles can be used to infer fine-scale population structure, perform polygenic prediction and haplotype cluster based association studies.

### Citation
Please cite our paper in [*Nature Communications*](https://doi.org/10.1038/s41467-024-55477-3)\
Preprint also available on [medRxiv](https://doi.org/10.1101/2024.04.30.24306654)

## Installation
```bash
# Option 1: Build and install via PyPI
pip install hapla

# Option 2: Download source and install via pip
git clone https://github.com/Rosemeis/hapla.git
cd hapla
pip install .

# Option 3: Download source and install in a new Conda environment
git clone https://github.com/Rosemeis/hapla.git
conda env create -f hapla/environment.yml
conda activate hapla
```
You can now run the `hapla` software and the subcommands. 

If you run into issues with your installation on a HPC system, it could be due to a mismatch of CPU architectures between login and compute nodes (illegal instruction). You can try and remove every instance of the `march=native` compiler flag in the [setup.py](./setup.py) file which optimizes `hapla` to your specific hardware setup. Another alternative is to use the [uv package manager](https://docs.astral.sh/uv/), where you can run `hapla` in a temporary and isolated environment by simply adding `uvx` in front of the `hapla` command.

## Quick start
***hapla*** contains the following subcommands at this moment:
- `hapla cluster`
- `hapla struct`
- `hapla predict`
- `hapla admix`
- `hapla fatash`


### Haplotype clustering
***hapla cluster***\
Window-based haplotype clustering in a phased VCF/BCF file (including index).
```bash
# Cluster haplotypes in a chromosome with fixed window size (8 SNPs)
hapla cluster --bcf data.chr1.bcf --size 8 --threads 8 --out hapla.chr1
# Saves inferred haplotype cluster assignments in binary hapla format
#	- hapla.chr1.bca
#	- hapla.chr1.ids
#	- hapla.chr1.win
```
`hapla cluster` outputs three files. A **.bca**-file (binary cluster assignments), which stores the cluster assignments as *unsigned char*s, a **.ids**-file with sample names and a **.win**-file with information about the genomic windows.

```bash
# Cluster haplotypes in a chromosome with fixed size and overlapping windows (step size 4)
hapla cluster --bcf data.chr1.bcf --size 8 --step 4 --threads 8 --out hapla.chr1

# Cluster haplotypes in all chromosomes and save output path in a filelist
for c in {1..22}
do
	hapla cluster --bcf data.chr${c}.bcf --size 8 --threads 8 --out hapla.chr${c}
	echo "hapla.chr${c}" >> hapla.filelist
done
```

Optionally, the haplotype cluster alleles can be saved in binary PLINK format (**.bed**, **.bim**, **.fam**) for ease of use with other software. Note that window information needs to be inferred from **.bim**-file for downstream analyses in this case.
```bash
hapla cluster --bcf data.chr1.bcf --threads 8 --out hapla.chr1 --plink
# Saves inferred haplotype cluster alleles in a binary PLINK format
#	- hapla.chr1.bed
#	- hapla.chr1.bim
#	- hapla.chr1.fam
```

The number of inferred haplotype clusters will depend on the chosen window size (`--size`), the number of allowed clusters per window (`--max-clusters`), as well as $\lambda$ (`--lmbda`) and the minimum haplotype cluster size. $\lambda$ represents the fraction of the specified window size in SNPs, which is required to create a new cluster based on Hamming distance, with a default setting of `--lmbda 0.125`.  The minimum haplotype cluster size can be adjusted using either `--min-freq` or `--min-mac`. The default setting is a minimum haplotype cluster frequency of at least 0.05 for the cluster to be retained (`--min-freq 0.05`), using `--min-mac` will override any setting for `--min-freq`. Smaller clusters will be iteratively removed. 


### Population structure inference and GRM estimation
***hapla struct***\
Infer population structure and estimate genome-wide relationship matrix (GRM) using haplotype cluster alleles.
```bash
# Perform PCA on all chromosomes (genome-wide) using filelist and extract top 20 eigenvectors
hapla struct --filelist hapla.filelist --threads 64 --pca 20 --out hapla
# Saves eigenvalues and eigenvectors in text-format
#	- hapla.eigenvecs
#	- hapla.eigenvals

# Or perform PCA on a single chromosome and extract top 20 eigenvectors
hapla struct --clusters hapla.chr1 --threads 64 --pca 20 --out hapla.chr1
# Saves eigenvalues and eigenvectors in text-format
#	- hapla.chr1.eigenvecs
#	- hapla.chr1.eigenvals

# Construct genome-wide relationship matrix (GRM)
hapla struct --filelist hapla.filelist --threads 64 --grm --out hapla
# Saves the GRM in binary GCTA format (float)
#	- hapla.grm.bin
#	- hapla.grm.N.bin
#	- hapla.grm.id
```

### Predict haplotype cluster assignments
***hapla predict***\
Predict haplotype cluster assignments using pre-computed cluster medians in a new set of haplotypes (VCF/BCF format). SNP sets must be overlapping.
```bash
# Cluster haplotypes in a chromosome with 'hapla cluster' and save cluster medians (--medians)
hapla cluster --bcf ref.chr1.bcf --size 8 --threads 64 --out ref.chr1 --medians
# Saves haplotype cluster medians (besides standard binary hapla format)
#	- ref.chr1.bcm
#	- ref.chr1.blk
#	- ref.chr1.wix

# Predict assignments in a set of new haplotypes using haplotype cluster medians
hapla predict --bcf new.chr1.bcf  --ref ref.chr1 --threads 64 --out new.chr1
# Saves predicted haplotype cluster assignments in binary hapla format
#	- new.chr1.bca
#	- new.chr1.ids
#	- new.chr1.win
```
Using `--medians` in `hapla cluster` outputs three extra files. A **.bcm**-file (binary cluster medians), which stores the cluster medians as *unsigned char*s, a **.blk**-file, which stores pairwise log-likelihoods between the cluster medians, a **.wix**-file with window index information. The files are needed to predict haplotype clusters in a new set of haplotypes.


### Ancestry estimation
***hapla admix***\
Estimate ancestry proportions and ancestral haplotype cluster frequencies with a pre-specified number of sources (K). Using a modified `fastmixture` model for use with our haplotype clusters. Projection and supervised modes are also available.
```bash
# Estimate ancestry proportions assuming K=3 ancestral sources for a single chromosome
hapla admix --clusters hapla.chr1 --K 3 --seed 1 --threads 64 --out hapla.chr1
# Saves Q matrix in text-format and P matrix as a binary file
#	- hapla.chr1.K3.s1.Q
#	- hapla.chr1.K3.s1.P.bin

# Estimate ancestry proportions assuming K=3 ancestral sources using filelist with all chromosomes
hapla admix --filelist hapla.filelist --K 3 --seed 1 --threads 64 --out hapla
# Saves Q matrix in text-format and separate binary files of P matrices, including a filelist of the P matrices
#	- hapla.K3.s1.Q
#	- hapla.K3.s1.chr{1..22}.P.bin
#	- hapla.K3.s1.pfilelist

# Estimate ancestry proportions in projection mode assuming K=3 ancestral sources using filelist with all chromosomes. Provide previously estimated ancestral haplotype cluster frequencies.
hapla admix --filelist hapla.proj.filelist --K 3 --seed 1 --threads 64 --projection hapla.K3.s1.pfilelist --out hapla.proj
# Saves Q matrix in text-format
#	- hapla.proj.K3.s1.Q

# Estimate ancestry proportions in supervised mode assuming K=3 ancestral sources using filelist with all chromosomes. Provide a single column text-file with population labels of the samples as integers, where 0 indicates no label.
hapla admix --filelist hapla.filelist --K 3 --seed 1 --threads 64 --supervised hapla.labels --out hapla.super
# Saves Q matrix in text-format and separate binary files of P matrices, including a filelist of the P matrices
#	- hapla.super.K3.s1.Q
#	- hapla.super.K3.s1.chr{1..22}.P.bin
#	- hapla.super.K3.s1.pfilelist
```


### Local ancestry inference
***hapla fatash***\
Infer local ancestry tracts using the admixture estimation from `hapla admix` in a hidden markov model. Based on a modified fastPHASE model for use with our haplotype clusters.
```bash
# Infer local ancestry tracts for a single chromosome (posterior decoding)
hapla fatash --clusters hapla.chr1 --qfile hapla.chr1.K3.s1.Q --pfile hapla.chr1.K3.s1.P.bin --threads 16 --out hapla.chr1
# Saves posterior decoding path in text-format
#	- hapla.chr1.path

# Infer local ancestry tracts using filelist with all chromosomes (Viterbi decoding)
hapla fatash --filelist hapla.filelist --qfile hapla.K3.s1.Q --pfilelist hapla.K3.s1.pfilelist --threads 16 --out hapla --viterbi
# Saves Viterbi decoding paths in text-files
#	- hapla.chr{1..22}.path
```
