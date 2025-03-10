# hapla (v0.21.0)
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

The number of inferred haplotype clusters will depend on the chosen window size but also on the minimum haplotype cluster size. The minimum haplotype cluster size can be adjusted using either `--min-freq` or `--min-mac`. The default setting is a minimum haplotype cluster frequency of at least 0.005 for the cluster to be retained (`--min-freq 0.005`). Smaller clusters will be iteratively removed.


### GRM estimation and population structure inference
***hapla struct***\
Estimate genome-wide relationship matrix (GRM) and infer population structure using the haplotype cluster alleles.
```bash
# Construct genome-wide relationship matrix (GRM)
hapla struct --filelist hapla.filelist --threads 64 --grm --out hapla
# Saves the GRM in binary GCTA format (float)
#	- hapla.grm.bin
#	- hapla.grm.N.bin
#	- hapla.grm.id

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
```

### Predict haplotype cluster assignments
***hapla predict***\
Predict haplotype cluster assignments using pre-computed cluster medians in a new set of haplotypes (VCF/BCF format). SNP sets must be overlapping.
```bash
# Cluster haplotypes in a chromosome with 'hapla cluster' and save cluster medians (--medians)
hapla cluster --bcf ref.chr1.bcf --size 8 --threads 64 --out ref.chr1 --medians
# Saves haplotype cluster medians (besides standard binary hapla format)
#	- ref.chr1.bcm
#	- ref.chr1.wix
#	- ref.chr1.hcc

# Predict assignments in a set of new haplotypes using haplotype cluster medians
hapla predict --bcf new.chr1.bcf  --ref ref.chr1 --threads 64 --out new.chr1
# Saves predicted haplotype cluster assignments in binary hapla format
#	- new.chr1.bca
#	- new.chr1.ids
#	- new.chr1.win
```
Using `--medians` in `hapla cluster` outputs three extra files. A **.bcm**-file (binary cluster medians), which stores the cluster medians as *unsigned char*s, a **.wix**-file with window index information and a **.hcc**-file with haplotype cluster counts. All files are needed to predict haplotype clusters in a new set of haplotypes.

### Admixture estimation
***hapla admix***\
Estimate ancestry proportions and ancestral haplotype cluster frequencies with a pre-specified number of sources (K). Using a modified ADMIXTURE model for our haplotype clusters.
```bash
# Estimate ancestry proportions assuming K=3 ancestral sources for a single chromosome
hapla admix --clusters hapla.chr1 --K 3 --seed 1 --threads 64 --out hapla.chr1
# Saves Q matrix in text-format and P matrix as a binary file
#	- hapla.chr1.K3.s1.Q
#	- hapla.chr1.K3.s1.P.bin

# Estimate ancestry proportions assuming K=3 ancestral sources using filelist with all chromosomes
hapla admix --filelist hapla.filelist --K 3 --seed 1 --threads 64 --out hapla
# Saves Q matrix in text-format and separate binary files of P matrices
#	- hapla.K3.s1.Q
#	- hapla.K3.s1.file{1..22}.P.bin
```

### Local ancestry inference
***hapla fatash***\
Infer local ancestry tracts using the admixture estimation in a hidden markov model. Using a modified fastPHASE model for our haplotype clusters.
```bash
# Infer local ancestry tracts for a single chromosome (posterior decoding)
hapla fatash --clusters hapla.chr1 --qfile hapla.chr1.K3.s1.Q --pfile hapla.chr1.K3.s1.P.bin --threads 16 --out hapla.chr1
# Saves posterior decoding path in text-format
#	- hapla.chr1.path

# Infer local ancestry tracts using filelist with all chromosomes (Viterbi decoding)
for c in {1..22}; do echo "hapla.chr1.K3.s1.file${c}.P.bin" >> hapla.K3.s1.pfilelist; done
hapla fatash --filelist hapla.filelist --qfile hapla.K3.s1.Q --pfilelist hapla.K3.s1.pfilelist --threads 16 --out hapla --viterbi
# Saves Viterbi decoding paths in text-files
#	- hapla.file{1..22}.path
```
