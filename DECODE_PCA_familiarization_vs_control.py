#Temporal PCA on ERP data


########################
# LOADING DATA
########################

from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

# load typical matlab file (as dictionnary)
ty_ML = loadmat('data/Typical.mat') #output is dict
#dict to np.array
ty_arrays = ty_ML['Typical'] #output is np of arrays (3,8)

#choose conditions
ty_fam_somato = ty_arrays[0,2]
ty_con_somato = ty_arrays[0,0]

#Run PCA1

pca = PCA(n_components=3)

pca.fit(ty_fam_somato)
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

mean_ty_fam_somato = np.mean(ty_fam_somato, axis = 0)
plt.plot(mean_ty_fam_somato)
plt.show()





#Run PCA2

pca = PCA(n_components=3)

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