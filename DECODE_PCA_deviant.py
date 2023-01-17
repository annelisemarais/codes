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

#see code Kaiser estimation

#nb of componenents Jan 17 2023 = 21

###########################
#PCA
###########################

###########################
#Run PCA for standard


pca_std = PCA(n_components=5)
n_components = 5
list_components = list(range(1,n_components+1))

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


###########################
#Run PCA for deviant

pca_dev = PCA(n_components=5)


#fit the data
pca_dev.fit(ty_dev_somato)
#how much variance is explained per component
print(pca_dev.explained_variance_ratio_) 

cumsum_dev = np.cumsum(pca_std.explained_variance_ratio_) * 100

plt.scatter(list_components,cumsum_dev, c="b")
plt.axhline(y=100, c="r")
plt.xticks(range(math.floor(min(list_components)), math.ceil(max(list_components))+1))
plt.yticks(list(range(30,110,10)))
plt.xlabel("Number of components")
plt.ylabel("Explained variance (%)")
plt.title("Explained variance per component for deviant condition")
plt.show()

#how much each time point contributes to the variance
print(pca_dev.components_) #one array by component. Each array length is length of time series


###########################
#plot all components from both conditions

f, ax = plt.subplots(1,2, sharey=True)

#plot all components loadings ?
for comp in list_components:
  ax[0].plot(pca_std.components_[comp-1, :])
  ax[1].plot(pca_dev.components_[comp-1, :])
plt.legend(list_components)
plt.xlabel("Time series (ms)")
ax[0].set_ylabel("Loadings ? condition standard")
ax[1].set_ylabel("Loadings ? condition deviant")
plt.title("Loadings ?")
plt.show()


### ??


loadings_std = pca_std.components_.T * np.sqrt(pca_std.explained_variance_)
loadings_dev = pca_dev.components_.T * np.sqrt(pca_dev.explained_variance_)

f, ax = plt.subplots(1,2, sharey=True)

#plot all components loadings * sqrt ?
for comp in list_components:
  ax[0].plot(loadings_std[:, comp-1])
  ax[1].plot(loadings_dev[:, comp-1])
plt.legend(list_components)
plt.xlabel("Time series (ms)")
ax[0].set_ylabel("Loadings * explained variance ? condition standard")
ax[1].set_ylabel("Loadings * explained variance ? condition deviant")
plt.title("Loadings * explained variance ?")
plt.show()

#componentsplus = pca.components_[pca.components_>(pca.components_.mean() + pca.components_.std())]
#componentsmoins = pca.components_[pca.components_<(pca.components_.mean() - pca.components_.std())]

#plt.scatter(pca.components_[0, :], pca.components_[1, :], marker = '.')
#plt.show()


#####
#Next Step

#Run PCA on all cchildren (typ and atyp confondu)
#ensuite PCA fit transform uniquement sur typique puis uniquement sur atypique et regarder difference