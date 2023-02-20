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
data = loadmat('data/Typical_PCA.mat')['Typical_PCA'] # 23478*1000

 
########
## Third step : construct pandas dataframe with label of each row 
#######

sub = []
for i in range(1,27):
    new_sub = [i] * 129* 7
    sub += new_sub

condition = []
for j in range(1, 27):
  for i in range(1,8):
    new_condition = [i] * 129
    condition += new_condition

electrode = list(range(1, 130)) * 26 *7

df_data = pd.DataFrame(data)
df_data['condition'] = condition
df_data['electrode'] = electrode
df_data['sub'] = sub

########
## Third step : separate data for PCA1 and PCA2 (omission only) 
#######

data_omission = df_data[df_data["condition"]==4]
data_omission2 = data_omission.drop(['condition', 'electrode','sub'], axis=1)
data_omission3 = data_omission2.to_numpy()

data1 = df_data.drop(df_data[df_data["condition"]==4].index)
data2 = data1.drop(['condition', 'electrode','sub'], axis=1)
data3 = data2.to_numpy()


########
## first step : compute full PCA to get needed nb of components (Kaiser method)
#######

kaiser_model = PCA()
kaiser_model.fit(data3)

# get how many components needed to reach 99% of explained variance
perc_var_explained = 0

for ind, i in enumerate(kaiser_model.explained_variance_ratio_):
  perc_var_explained += i
  if perc_var_explained>0.99:
    print(ind, " Components needed")
    n_component_to_keep = ind
    break

########
## Second step : keep only the needed nb of components 
#######

model = FactorAnalysis(n_components=n_component_to_keep)
model.fit(data3)
loadings = model.components_


# plot the loadings:
def plot_ts(loadings):
  plt.close('all')
  for ind, comp in enumerate(loadings):
    plt.plot(range(0, 500), comp, linewidth=3, label='comp{}'.format(ind))
  plt.xlabel("Time series (ms)")
  plt.ylabel("Loadings")
  #plt.set_xticks(range(0,999,99))
  #plt.set_xticklabels(['-100','0','100','200','300','400','500','600','700','800','900'])
  #plt.savefig("Components.png")
plot_ts(loadings)
plt.legend()
plt.show()


########
## fourth step : build factor scores
####### 

factor_scores = model.fit_transform(data)

########
## multiply factor score times loading peak abs
####### 
peak_loadings = np.zeros(loadings.shape)
loadings_abs = abs(loadings)
for row in range(6):
  peak_loading_index = np.where(loadings_abs[row] == loadings_abs[row].max())
  peak_loadings[row][peak_loading_index] = loadings_abs[row][peak_loading_index]

dependant_variables = factor_scores * np.max(abs(loadings), axis=1)


########
## visualisation condition 1 vs condition 3 on these dependant variables
####### 

# pick the indexes of the particular group under test
c1_frontal = df_data[df_data['condition']==1][df_data['electrode'].isin([5, 6, 7, 12, 13, 106, 112])][df_data.columns[0:1000]]
c3_frontal = df_data[df_data['condition']==3][df_data['electrode'].isin([5, 6, 7, 12, 13, 106, 112])][df_data.columns[0:1000]]

# get the factor at specific index
c1_frontal_stat = factor_scores[c1_frontal.index]
c3_frontal_stat = factor_scores[c3_frontal.index]

# extract scores for a specific component
comp_under_test = 1 # up to 6
var_ind_c3 = c3_frontal_stat[:, comp_under_test]
var_ind_c1 = c1_frontal_stat[:, comp_under_test]

# perfom t-test
from scipy import stats
stats.ttest_rel(var_ind_c1, var_ind_c3)