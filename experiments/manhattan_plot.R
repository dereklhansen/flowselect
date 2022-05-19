library(tidyverse)
library(qqman)
trait = "oil"
df_cols = read_csv('./protein_columns.txt')  # SNP names (same for protein and oil)
names(df_cols) = c("idx", "snp_name") 
snp_names = df_cols %>% filter(idx > 3)  # extract SNP names 
snps = snp_names$snp_name
pv_name = paste("./pvalues-", trait, ".csv", sep="")
p_vals = read_csv(pv_name)
p_vals = p_vals$p_values  # get vector of p-values
thresh_name = paste("./thresholds-", trait, ".csv", sep="")
threshs = read_csv(thresh_name)  # load BH thresholds
snp_num = function(x){  # split SNP name to get position
  splits = str_split(x, pattern="_")
  return(splits[[1]][2])
}
# create data frame for manhattan plot
BP = as.vector(seq(1:length(p_vals)))  
CHR = as.numeric(as.vector(str_sub(snps, 3, 4)))
SNP = unname(sapply(snps, snp_num))
P = p_vals
x_man = data.frame(cbind(BP, CHR, P,SNP))
x_man$CHR = as.numeric(x_man$CHR)
x_man$BP = as.numeric(x_man$BP)
x_man$P = as.numeric(x_man$P)
nfdr = .2
bh_thresh = threshs$threshold[match(nfdr, threshs$n_fdrs)]
manhattan(x_man, 
          suggestiveline=-log10(bh_thresh+1e-6), 
          annotatePval=bh_thresh+1e-6, 
          col=c("skyblue", "navy"), 
          annotateTop=FALSE,
          cex.lab=0.9,
          cex.axis=0.9,
          cex=1.2)