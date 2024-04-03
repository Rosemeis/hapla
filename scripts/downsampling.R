##### R-script for downsampling SNPs based on MAF bins

D <- 10 # Downsampling factor
M <- 10 # Number of MAF bins

### AFR
f <- read.table("data/afr.afreq")
m_l <- ceiling((dim(f)[1])/M/D)
f$V7 <- f$V5
f$V7[f$V7 > 0.5] <- 1 - f$V7[f$V7 > 0.5]
f$V8 <- cut(f$V7, M)

# First level
f_new <- f[f$V8 == levels(f$V8)[1],]
m_c <- dim(f_new)[1]
f_new <- f_new[sample(1:m_c)[1:m_l],]

# Loop
for (l in levels(f$V8)[2:M]) {
	f_tmp <- f[f$V8 == l,]
	m_c <- dim(f_tmp)[1]
	f_new <- rbind(f_new, f_tmp[sample(1:m_c)[1:m_l],])
}
f_new <- f_new[order(f_new$V2),2]
write.table(f_new, file="data/afr.down.snp.id", sep="\t", quote=F, row.names=F, col.names=F)


### EUR
f <- read.table("data/eur.afreq")
m_l <- ceiling((dim(f)[1])/M/D)
f$V7 <- f$V5
f$V7[f$V7 > 0.5] <- 1 - f$V7[f$V7 > 0.5]
f$V8 <- cut(f$V7, M)

# First level
f_new <- f[f$V8 == levels(f$V8)[1],]
m_c <- dim(f_new)[1]
f_new <- f_new[sample(1:m_c)[1:m_l],]

# Loop
for (l in levels(f$V8)[2:M]) {
	f_tmp <- f[f$V8 == l,]
	m_c <- dim(f_tmp)[1]
	f_new <- rbind(f_new, f_tmp[sample(1:m_c)[1:m_l],])
}
f_new <- f_new[order(f_new$V2),2]
write.table(f_new, file="data/eur.down.snp.id", sep="\t", quote=F, row.names=F, col.names=F)


### AFR
f <- read.table("data/all.afreq")
m_l <- ceiling((dim(f)[1])/M/D)
f$V7 <- f$V5
f$V7[f$V7 > 0.5] <- 1 - f$V7[f$V7 > 0.5]
f$V8 <- cut(f$V7, M)

# First level
f_new <- f[f$V8 == levels(f$V8)[1],]
m_c <- dim(f_new)[1]
f_new <- f_new[sample(1:m_c)[1:m_l],]

# Loop
for (l in levels(f$V8)[2:M]) {
	f_tmp <- f[f$V8 == l,]
	m_c <- dim(f_tmp)[1]
	f_new <- rbind(f_new, f_tmp[sample(1:m_c)[1:m_l],])
}
f_new <- f_new[order(f_new$V2),2]
write.table(f_new, file="data/all.down.snp.id", sep="\t", quote=F, row.names=F, col.names=F)
