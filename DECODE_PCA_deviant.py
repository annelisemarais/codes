#Temporal PCA on ERP data

#cd Z:\Bureau\data_analysis
########################
# LOADING DATA
########################

from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
import math

# load typical matlab file (as dictionnary)
ty_ML = loadmat('data/Typical.mat') #output is dict
#dict to np.array
ty_arrays = ty_ML['Typical'] #output is np of arrays (3,8)

#choose conditions
ty_std_somato = ty_arrays[0,5]
ty_dev_somato = ty_arrays[0,1]

#Check shape
ty_std_somato.shape #must be nb of participants * length time series
ty_dev_somato.shape





###########################
#KAISER ESTIMATION IN R
###########################

#From github.com/FlorianScharf/PCA_Tutorial

if(!require(MASS)) install.packages("MASS")
if(!require(R.matlab)) install.packages("R.matlab")
if(!require(psych)) install.packages("psych")
if(!require(psych)) install.packages("EFAtools")

library(MASS)
library(psych)
library(R.matlab)
library(EFAtools)

rm(list=ls()) #empty workspace
filename <- "Z:/Bureau/data_analysis/data/dataforR.mat"

dataforR <- readMat(filename)

colnames(dataforR$dataR) = paste0("erp_", 1:1000)
colnames(dataforR$dataidxR) = c("group", "cond", "topo", "subj")

erpdata = data.frame(cbind(dataforR$dataidxR, dataforR$dataR))
erpdata$group = factor(erpdata$group, labels = c("typical", "atypical"))
erpdata$cond = factor(erpdata$cond, labels = c("typical"))
erpdata$topo = factor(erpdata$topo, labels = c("som", "ftl"))
erpdata$subj = factor(erpdata$subj)

fs = 1000 # sampling rate
xmin = -0.1 # baseline
pnts = 1000 # epoch duration
lat = (1:pnts - 1) / fs + xmin

write.csv(erpdata, file = "Z:/Bureau/data_analysis/data/erpdata.csv")
save(erpdata, fs, xmin, pnts, lat, file = "Z:/Bureau/data_analysis/data/erpdata.Rdata")

group = "typical"

NDD_group_data = as.matrix(erpdata[erpdata$group == group, -c(1:4)])

R = cor(NDD_group_data)

##EKC

EKC <- function (R, N = NA, use = c("pairwise.complete.obs", "all.obs", 
                             "complete.obs", "everything", "na.or.complete"), cor_method = c("pearson", 
                                                                                             "spearman", "kendall")) 
{

  p <- ncol(R)
  lambda <- eigen(R, symmetric = TRUE, only.values = TRUE)$values
  refs <- vector("double", p)
  for (i in seq_len(p)) {
    refs[i] <- max(((1 + sqrt(p/N))^2) * (p - sum(refs))/(p - 
                                                            i + 1), 1)
  }
  out <- list(eigenvalues = lambda, n_factors = which(lambda <= 
                                                        refs)[1] - 1, references = refs, settings = list(use = use, 
                                                                                                         cor_method = cor_method, N = N))
  class(out) <- "EKC"
  return(out)
}
###

res_ekc = EKC(R, N = nrow(NDD_group_data))

nFac = res_ekc$n_factors

nFac #to show the number of factors

###Plot variance explained by factor###
plot(1:ncol(NDD_group_data), res_ekc$eigenvalues,
     xlab = "Factor", ylab = "Variance Explained",
     main = ifelse(group == "typical", "typical", "atypical"),
     xlim = c(0,40), pch = 16,
     col = (res_ekc$references <= res_ekc$eigenvalues) + 1)
lines(1:ncol(NDD_group_data), res_ekc$references, lty = 2, lwd = 3,
      col = "blue")

abline(v = nFac)
text(x = nFac, y = 100, pos = 2, cex = 0.8,  
     labels = paste0("Number of Factors\nto be extracted: ", nFac))





###########################
#PCA
###########################


#Run PCA for standard

pca_std = PCA(n_components=21)
n_components = 21
list_components = list(range(1,22))

#fit the data
pca_std.fit(ty_std_somato)
#how much variance is explained per component
print(pca_std.explained_variance_ratio_) 

cumsum_std = np.cumsum(pca_std.explained_variance_ratio_) * 100

plt.scatter(list_components,cumsum_std, c="b")
plt.axhline(y=100, c="r")
plt.xticks(range(math.floor(min(list_components)), math.ceil(max(list_components))+1))
plt.yticks(list(range(30,110,10)))
plt.xlabel("Number of components")
plt.ylabel("Explained variance (%)")
plt.title("Explained variance per component for standard condition")
plt.show()

#how much each time point contributes to the variance
print(pca_std.components_) #one array by component. Each array length is length of time series

#componentsplus = pca.components_[pca.components_>(pca.components_.mean() + pca.components_.std())]
#componentsmoins = pca.components_[pca.components_<(pca.components_.mean() - pca.components_.std())]

#plt.scatter(pca.components_[0, :], pca.components_[1, :], marker = '.')
#plt.show()













for comp in list_components:
  plt.plot(pca_std.components_[comp, :])


#COMPOSANT 1
meancomp1 = np.mean(pca_std.components_[0, :])
stdmeancomp1 = np.std(pca_std.components_[0, :])
stdposcomp1 = meancomp1 + stdmeancomp1
stdnegcomp1 = meancomp1 - stdmeancomp1

plt.plot(pca_std.components_[0, :])
plt.axhline(y=stdposcomp1)
plt.axhline(y=stdnegcomp1)
plt.show()







#COMPOSANT 2 

meancomp2 = np.mean(pca.components_[1, :])
stdmeancomp2 = np.std(pca.components_[1, :])
stdposcomp2 = meancomp2 + stdmeancomp2
stdnegcomp2 = meancomp2 - stdmeancomp2

plt.plot(pca.components_[1, :])
plt.axhline(y=stdposcomp2)
plt.axhline(y=stdnegcomp2)
plt.show()

#COMPOSANT 3 

meancomp3 = np.mean(pca.components_[2, :])
stdmeancomp3 = np.std(pca.components_[2, :])
stdposcomp3 = meancomp3 + stdmeancomp3
stdnegcomp3 = meancomp3 - stdmeancomp3

plt.plot(pca.components_[2, :])
plt.axhline(y=stdposcomp3)
plt.axhline(y=stdnegcomp3)
plt.show()

mean_ty_fam_somato = np.mean(ty_fam_somato, axis = 0)
plt.plot(mean_ty_fam_somato)
plt.show()





#Run PCA2

pca = PCA(n_components=5)

pca.fit(ty_con_somato)
print(pca.explained_variance_ratio_)
print(pca.components_)

componentsplus = pca.components_[pca.components_>(pca.components_.mean() + pca.components_.std())]
componentsmoins = pca.components_[pca.components_<(pca.components_.mean() - pca.components_.std())]

plt.scatter(pca.components_[0, :], pca.components_[1, :], marker = '.')
plt.show()

#COMPOSANT 1
meancomp1 = np.mean(pca.components_[0, :])
stdmeancomp1 = np.std(pca.components_[0, :])
stdposcomp1 = meancomp1 + stdmeancomp1
stdnegcomp1 = meancomp1 - stdmeancomp1

plt.plot(pca.components_[0, :])
plt.axhline(y=stdposcomp1)
plt.axhline(y=stdnegcomp1)
plt.show()

#COMPOSANT 2 

meancomp2 = np.mean(pca.components_[1, :])
stdmeancomp2 = np.std(pca.components_[1, :])
stdposcomp2 = meancomp2 + stdmeancomp2
stdnegcomp2 = meancomp2 - stdmeancomp2

plt.plot(pca.components_[1, :])
plt.axhline(y=stdposcomp2)
plt.axhline(y=stdnegcomp2)
plt.show()

#COMPOSANT 3 

meancomp3 = np.mean(pca.components_[2, :])
stdmeancomp3 = np.std(pca.components_[2, :])
stdposcomp3 = meancomp3 + stdmeancomp3
stdnegcomp3 = meancomp3 - stdmeancomp3

plt.plot(pca.components_[2, :])
plt.axhline(y=stdposcomp3)
plt.axhline(y=stdnegcomp3)
plt.show()

mean_ty_con_somato = np.mean(ty_con_somato, axis = 0)
plt.plot(mean_ty_con_somato)
plt.show()




#####
#Next Step

#Run PCA on all cchildren (typ and atyp confondu)
#ensuite PCA fit transform uniquement sur typique puis uniquement sur atypique et regarder difference