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
import statsmodels.api as sm

# load typical matlab file (as dictionnary)
ty_ML = loadmat('data/Typical.mat') #output is dict
#dict to np.array
ty_arrays = ty_ML['Typical'] #output is np of arrays (3,8)

#choose conditions
ty_std_somato = ty_arrays[0,5]
ty_dev_somato = ty_arrays[0,1]

ty_std_mean_somato = np.mean(ty_std_somato)
ty_dev_mean_somato = np.mean(ty_dev_somato)

plt.plot(np.mean(ty_std_somato, axis=0), c="k")
plt.plot(np.mean(ty_dev_somato, axis=0),c="b")
plt.gca().invert_yaxis()
plt.xlabel("Time series (ms)")
plt.ylabel("µV")
plt.title("deviance in somatosensory cortex of children 2-6 yo")
plt.legend()
plt.show()

#Check shape
ty_std_somato.shape #must be nb of participants * length time series
ty_dev_somato.shape





###########################
#KAISER ESTIMATION IN R
###########################

#see code Kaiser estimation

#nb of componenents Jan 17 2023 = 21

###########################
#PCA
###########################

###########################
#Run PCA for standard


pca_std_somato = PCA(n_components=21)
n_components = 21
list_components = list(range(1,n_components+1))

#covariance matrix 
cov_std_somato = np.cov(ty_std_somato.T) #transpose to have covar on time series

#fit the data
pca_std_somato = pca_std_somato.fit(cov_std_somato)
#how much variance is explained per component
print(pca_std_somato.explained_variance_ratio_) 

cumsum_std_somato = np.cumsum(pca_std_somato.explained_variance_ratio_) * 100

plt.scatter(list_components,cumsum_std_somato, c="b")
plt.axhline(y=100, c="r")
plt.xticks(range(math.floor(min(list_components)), math.ceil(max(list_components))+1))
plt.yticks(list(range(30,110,10)))
plt.xlabel("Number of components")
plt.ylabel("Explained variance (%)")
plt.title("Explained variance per component for somatosensory standard condition")
plt.show()

#how much each time point contributes to the variance
print(pca_std_somato.components_) #one array by component. Each array length is length of time series

for comp in list_components:
  plt.plot(pca_std_somato.components_[comp-1, :])
plt.legend(list_components, title="components")
plt.xlabel("Time series (ms)")
plt.ylabel("Components loadings in somattosensory standard condition")
plt.title("Component loadings")
plt.show()



#FACTOR SCORES
factor_scores = pca_std_somato.fit_transform(cov_std_somato)
for comp in list_components:
  plt.plot(factor_scores[:, comp-1])
plt.legend(list_components, title="components")
plt.xlabel("Time series (ms)")
plt.ylabel("Components loadings in somattosensory standard condition")
plt.title("Component loadings")
plt.show()


###########################
#Run PCA for deviant

pca_dev_somato = PCA(n_components=21)


#covariance matrix 
cov_dev_somato = np.cov(ty_dev_somato.T) #transpose to have covar on time series

#fit the data
pca_dev_somato = pca_dev_somato.fit(cov_dev_somato)
#how much variance is explained per component
print(pca_dev_somato.explained_variance_ratio_) 

cumsum_dev_somato = np.cumsum(pca_dev_somato.explained_variance_ratio_) * 100

plt.scatter(list_components,cumsum_dev_somato, c="b")
plt.axhline(y=100, c="r")
plt.xticks(range(math.floor(min(list_components)), math.ceil(max(list_components))+1))
plt.yticks(list(range(30,110,10)))
plt.xlabel("Number of components")
plt.ylabel("Explained variance (%)")
plt.title("Explained variance per component for somatosensory deviant condition")
plt.show()

#how much each time point contributes to the variance
print(pca_dev_somato.components_) #one array by component. Each array length is length of time series

for comp in list_components:
  plt.plot(pca_dev_somato.components_[comp-1, :])
plt.legend(list_components, title="components")
plt.xlabel("Time series (ms)")
plt.ylabel("Components loadings in somatosensory deviant condition")
plt.title("Component loadings")
plt.show()


###########################
#plot all components loadings from both conditions

f, ax = plt.subplots(1,2, sharey=True)
for comp in list_components:
  ax[0].plot(pca_std_somato.components_[comp-1, :])
  ax[1].plot(pca_dev_somato.components_[comp-1, :])
plt.legend(list_components, title="components")
ax[0].set_xlabel("Time series (ms)")
ax[1].set_xlabel("Time series (ms)")
ax[0].set_ylabel("Components loadings in standard condition")
ax[1].set_ylabel("Components loadings in deviant condition")
plt.suptitle("Component loadings in somatosensory cortex")
plt.show()









### loadings * sqrt

loadingsqrt_std_somato = pca_std_somato.components_.T * np.sqrt(pca_std_somato.explained_variance_)
loadingsqrt_dev_somato = pca_dev_somato.components_.T * np.sqrt(pca_dev_somato.explained_variance_)

#plot all components loadings * sqrt ?
f, ax = plt.subplots(1,2, sharey=True)
for comp in list_components:
  ax[0].plot(loadingsqrt_std_somato[:, comp-1])
  ax[1].plot(loadingsqrt_dev_somato[:, comp-1])
plt.legend(list_components, title="components")
ax[0].set_xlabel("Time series (ms)")
ax[1].set_xlabel("Time series (ms)")
ax[0].set_ylabel("Loadings * explained variance in standard condition")
ax[1].set_ylabel("Loadings * explained variance in deviant condition")
plt.suptitle("Loadings * explained variance in somatosensory cortex")
plt.show()











###Trying Varimax rotation on std data


pca_try = PCA(n_components=21)
cov_try = np.cov(ty_std_somato.T) #transpose to have covar on time series
pca_try = pca_try.fit(cov_try)

#Standardize factor loadings
standardized_components = pca_try.components_ / np.std(pca_try.components_)

##Define Varimax### from wikipedia
from numpy import eye, asarray, dot, sum, diag
from numpy.linalg import svd
def varimax(Phi, gamma = 1.0, q = 20, tol = 1e-6):
    p,k = Phi.shape
    R = eye(k)
    d=0
    for i in range(q):
        d_old = d
        Lambda = dot(Phi, R)
        u,s,vh = svd(dot(Phi.T,asarray(Lambda)**3 - (gamma/p) * dot(Lambda, diag(diag(dot(Lambda.T,Lambda))))))
        R = dot(u,vh)
        d = sum(s)
        if d_old!=0 and d/d_old < 1 + tol: break
    return dot(Phi, R)
###################


my_varimax = varimax(standardized_components)

for comp in list_components:
  plt.plot(my_varimax[comp-1, :])
plt.legend(list_components, title="components")
plt.xlabel("Time series (ms)")
plt.ylabel("Components loadings in somatosensory deviant condition")
plt.title("Component loadings")
plt.show()












#componentsplus = pca.components_[pca.components_>(pca.components_.mean() + pca.components_.std())]
#componentsmoins = pca.components_[pca.components_<(pca.components_.mean() - pca.components_.std())]

#plt.scatter(pca.components_[0, :], pca.components_[1, :], marker = '.')
#plt.show()


#####
#Next Step

#Run PCA on all cchildren (typ and atyp confondu)
#ensuite PCA fit transform uniquement sur typique puis uniquement sur atypique et regarder difference