########################
# LOADING DATA
########################
#cd Z:\Bureau\data_analysis
from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import normalize
import math

# load typical matlab file (as dictionnary)
data_test = loadmat('data/avrdata.mat')['data'] 

 

########
## first step : compute full PCA to get needed nb of components (Kaiser method)
#######

kaiser_model_test = FactorAnalysis()
kaiser_model_test.fit(data_test)

# get how many components needed to reach 99% of explained variance
perc_var_explained = 0
n_components_test=0
components_variance_test = []
matrix = np.sum(kaiser_model.components_**2, axis=1)

for i in range(len(matrix)):
  variance_comp = (100*matrix[i])/np.sum(matrix)
  components_variance_test.append(variance_comp)
  perc_var_explained += variance_comp
  n_components_test = i
  if perc_var_explained>99:
    print(n_components_test, " Components needed")
    break

components_variance_test = np.array(components_variance_test)
########
## first step : compute full PCA to get needed nb of components (Kaiser method)
#######

model_test = FactorAnalysis(n_components=n_components_test, rotation='varimax')
model_test.fit(data_test)
components_test = model_test.components_

#######
##plot components
#######

legend_test = ['comp1','comp2','comp3','comp4','comp5','comp6','comp7','comp8','comp9']

# plot the loadings:
def plot_ts(loadings):
  plt.close('all')
  for ind, comp in enumerate(loadings):
    plt.plot(range(0, 500), comp, linewidth=3)
  plt.xlabel("Time series (ms)")
  plt.ylabel("Loadings")
  plt.legend(legend)

plot_ts(abs(components_test))
plt.show()


##############
##save componenent 1 by 1
##############

#normalize by component
norm_components_test = []
for comp in components_test:
  normalize = (comp - np.mean(comp))/np.std(comp)
  norm_components_test.append(normalize)

norm_components_test=np.array(norm_components_test)



############
##find max
############
