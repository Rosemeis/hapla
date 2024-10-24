##### Bash script for simulation study in hapla paper
#####
##### Requires:
##### 	- stdpopsim
##### 	- tskit
##### 	- tszip
##### 	- bcftools
##### 	- plink2
##### 	- gcta-1.94.1
#####	- hapla

# Create directories
mkdir data
mkdir clusters
mkdir pheno
mkdir grm
mkdir reml
mkdir reml/pheno1
mkdir reml/pheno2
mkdir reml/pheno3

# Simulation
stdpopsim HomSap -d 'OutOfAfrica_2T12' -L 1e8 -s 42 -o data/all.ts AFR:10000 EUR:10000
for s in {0..19999}
do
	echo "tsk_${s}" >> data/all.samples
done
head -n 10000 data/all.samples > data/afr.samples
tail -n 10000 data/all.samples > data/eur.samples

# Data file generation
for p in {afr,eur,all}
do
	# VCF
	tskit vcf data/all.ts | bcftools view -S data/${p}.samples -p -m 2 -M 2 -v snps -q 0.01 -Q 0.99 -Ou | bcftools norm -d all -Ob -o data/${p}.bcf
	bcftools index data/${p}.bcf

	# PLINK and LD pruning
	plink2 --bcf data/${p}.bcf --indep-pairwise 50 10 0.5 --freq --make-bed --out data/${p}
	plink2 --bfile data/${p} --extract data/${p}.prune.in --make-bed --out data/${p}.pruned

	# Downsampled
	Rscript downsampling.R
	bcftools view -i ID==@data/${p}.down.snp.id -Ob -o data/${p}.down.bcf data/${p}.bcf
	plink2 --bcf data/${p}.down.bcf --indep-pairwise 50 10 0.5 --make-bed --out data/${p}.down
	plink2 --bfile data/${p}.down --extract data/${p}.down.prune.in --make-bed --out data/${p}.down.pruned
done
tszip all.ts # Compress tree sequence

# Haplotype clustering
for p in {afr,eur,all}
do
	for w in {1,16,32}
	do 
		hapla cluster --bcf data/${p}.bcf --fixed $w --step $((($w+2-1)/2)) --threads 8 --out clusters/${p}.w${w}
		hapla cluster --bcf data/${p}.down.bcf --fixed $w --step $((($w+2-1)/2)) --threads 8 --out clusters/${p}.down.w${w}
	done
	hapla cluster --bcf data/${p}.bcf --fixed 8 --step $((($w+2-1)/2)) --threads 8 --out clusters/${p}.w8 --plink # For phenotype generation
	hapla cluster --bcf data/${p}.down.bcf --fixed 8 --step $((($w+2-1)/2)) --threads 8 --out clusters/${p}.down.w8
done

# Phenotype generation
for p in {afr,eur,all}
do
	python generatePheno.py --bfile data/${p} --causal 1000 --h2 0.8 --phenos 10 --out pheno/${p}.pheno1
	python generatePheno.py --bfile data/${p} --causal 1000 --h2 0.2 --phenos 10 --out pheno/${p}.pheno2
	python generatePheno.py --bfile clusters/${p}.w8 --causal 1000 --h2 0.8 --phenos 10 --out pheno/${p}.pheno3
done

# GRM estimation
for p in {afr,eur,all}
do
	for w in {1,8,16,32}
	do 
		hapla struct --clusters clusters/${p}.w${w}.z.npy --grm --threads 64 --iid data/${p}.samples --out grm/${p}.w${w}
		hapla struct --clusters clusters/${p}.down.w${w}.z.npy --grm --threads 64 --iid data/${p}.samples --out grm/${p}.down.w${w}
		realpath grm/${p}.w${w} >> grm/${p}.multi
		realpath grm/${p}.down.w${w} >> grm/${p}.down.multi
	done
	gcta-1.94.1 --bfile data/${p} --make-grm --threads 64 --out grm/${p}.gcta
	gcta-1.94.1 --bfile data/${p}.down --make-grm --threads 64 --out grm/${p}.down.gcta
	gcta-1.94.1 --bfile data/${p}.pruned --make-grm --threads 64 --out grm/${p}.pruned
	gcta-1.94.1 --bfile data/${p}.down.pruned --make-grm --threads 64 --out grm/${p}.down.pruned
done

# Estimate heritability and cvBLUPs
for y in {1..3}
do
	for p in {afr,eur,all}
	do
		for s in {1..10}
		do
			for w in {1,8,16,32}
			do
				gcta-1.94.1 --reml --reml-no-lrt --reml-no-constrain --cvblup --grm grm/${p}.w${w} --threads 64 --pheno pheno/${p}.pheno${y}.pheno --mpheno $s --out reml/pheno${y}/${p}.w${w}.${s}
				gcta-1.94.1 --reml --reml-no-lrt --reml-no-constrain --cvblup --grm grm/${p}.down.w${w} --threads 64 --pheno pheno/${p}.pheno${y}.pheno --mpheno $s --out reml/pheno${y}/${p}.down.w${w}.${s}
			done
			gcta-1.94.1 --reml --reml-no-lrt --reml-no-constrain --cvblup --mgrm grm/${p}.multi --threads 64 --pheno pheno/${p}.pheno${y}.pheno --mpheno $s --out reml/pheno${y}/${p}.multi.${s}
			gcta-1.94.1 --reml --reml-no-lrt --reml-no-constrain --cvblup --mgrm grm/${p}.down.multi --threads 64 --pheno pheno/${p}.pheno${y}.pheno --mpheno $s --out reml/pheno${y}/${p}.down.multi.${s}
			gcta-1.94.1 --reml --reml-no-lrt --reml-no-constrain --cvblup --grm grm/${p}.gcta --threads 64 --pheno pheno/${p}.pheno${y}.pheno --mpheno $s --out reml/pheno${y}/${p}.gcta.${s}
			gcta-1.94.1 --reml --reml-no-lrt --reml-no-constrain --cvblup --grm grm/${p}.down.gcta --threads 64 --pheno pheno/${p}.pheno${y}.pheno --mpheno $s --out reml/pheno${y}/${p}.down.gcta.${s}
			gcta-1.94.1 --reml --reml-no-lrt --reml-no-constrain --cvblup --grm grm/${p}.pruned --threads 64 --pheno pheno/${p}.pheno${y}.pheno --mpheno $s --out reml/pheno${y}/${p}.pruned.${s}
			gcta-1.94.1 --reml --reml-no-lrt --reml-no-constrain --cvblup --grm grm/${p}.down.pruned --threads 64 --pheno pheno/${p}.pheno${y}.pheno --mpheno $s --out reml/pheno${y}/${p}.down.pruned.${s}
		done
	done
done
