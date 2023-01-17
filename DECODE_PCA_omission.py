########################
# LOADING DATA
########################

from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

# load atypical and typical matlab file (as dictionnary)
aty_ML = loadmat('data/Atypical.mat') #output is dict
ty_ML = loadmat('data/Typical.mat') #output is dict
#dict to np.array
aty_arrays = aty_ML['Atypical'] #output is np of arrays (3,8)
ty_arrays = ty_ML['Typical'] #output is np of arrays (3,8)

#choose omission condition
ty_omi_somato = ty_arrays[0,3]
ty_omi_frtl = ty_arrays[1,3]
aty_omi_somato = aty_arrays[0,3]
aty_omi_frtl = aty_arrays[1,3]

#normalize
ty_omi_somato_norm = normalize(ty_omi_somato)
aty_omi_somato_norm = normalize(aty_omi_somato)

#Run PCA1

pca = PCA(n_components=3)

pca.fit(ty_omi_somato_norm)
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

mean_ty_omi_somato = np.mean(ty_omi_somato, axis = 0)
plt.plot(mean_ty_omi_somato)
plt.show()





#Run PCA2

pca = PCA(n_components=3)

pca.fit(aty_omi_somato_norm)
print(pca.explained_variance_ratio_)
print(pca.components_)

componentsplus = pca.components_[pca.components_>(pca.components_.mean() + pca.components_.std())]
componentsmoins = pca.components_[pca.components_<(pca.components_.mean() - pca.components_.std())]

plt.scatter(pca.components_[0, :], pca.components_[1, :], marker = '.')
plt.show()

#COMPOSANT1

meancomp1 = np.mean(pca.components_[0, :])
stdmeancomp1 = np.std(pca.components_[0, :])
stdposcomp1 = meancomp1 + stdmeancomp1
stdnegcomp1 = meancomp1 - stdmeancomp1

plt.plot(pca.components_[0, :])
plt.axhline(y=stdposcomp1)
plt.axhline(y=stdnegcomp1)
plt.show()

mean_aty_omi_somato = np.mean(aty_omi_somato, axis = 0)
plt.plot(mean_aty_omi_somato)
plt.show()

#COMPOSANT2

meancomp2 = np.mean(pca.components_[1, :])
stdmeancomp2 = np.std(pca.components_[1, :])
stdposcomp2 = meancomp2 + stdmeancomp2
stdnegcomp2 = meancomp2 - stdmeancomp2

plt.plot(pca.components_[1, :])
plt.axhline(y=stdposcomp2)
plt.axhline(y=stdnegcomp2)
plt.show()

mean_ty_omi_somato = np.mean(ty_omi_somato, axis = 0)
plt.plot(mean_ty_omi_somato)
plt.show()




#####
#Next Step

#Run PCA on all cchildren (typ and atyp confondu)
#ensuite PCA fit transform uniquement sur typique puis uniquement sur atypique et regarder difference