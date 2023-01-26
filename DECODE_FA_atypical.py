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

df_data = pd.DataFrame(data)

#########
##First step : Detect bad channels that will be removed from analysis
#########

###create a standardized df
df_standardized = (df_data - np.mean(df_data)) / np.std(df_data)
#find channel with data > 3 std
df_badchan = df_standardized[df_standardized>3]
df_badchan = df_badchan.replace(np.nan,0)
#cumsum to easily find those channels in the last column
df_badchan = df_badchan.cumsum(axis=1)
df_badchan.to_excel("df_badchan_typical.xlsx") #for manual checking
#get the last column
df_mybadchan = df_badchan[999]
#find index of non zero (bad channel) rwos
df_mybadchan = df_mybadchan.index[df_mybadchan>0]

########
## Second step : construct pandas dataframe with label of each row 
#######

sub = []
for i in range(1,27): #26 subjects
    new_sub = [i] * 129* 7 #129 channels and 7 conditions
    sub += new_sub

condition = []
for j in range(1, 27): #26 subjects
  for i in range(1,8): #7conditions
    new_condition = [i] * 129 #channels
    condition += new_condition

electrode = list(range(1, 130)) * 26 *7


df_data['condition'] = condition
df_data['electrode'] = electrode
df_data['sub'] = sub

######
##Third step Discard eye channels (125:128) and bad channels  
######
#check difference between length must be equal to number of dropped index

#delete bad channels
df_nobadchannel = df_data.drop(index=df_mybadchan)
df_nobadchannel.to_excel("df_nobadchannel_typical.xlsx")
#delete eye channels
df_clean = df_nobadchannel.drop(df_nobadchannel[df_nobadchannel['electrode'] ==125].index)
df_clean = df_clean.drop(df_clean[df_clean['electrode'] ==126].index)
df_clean = df_clean.drop(df_clean[df_clean['electrode'] ==127].index)
df_clean = df_clean.drop(df_clean[df_clean['electrode'] ==128].index)

df_clean.to_excel("df_clean_typical.xlsx")

########
## Third step : separate data for PCA1 (data without omission) and PCA2 (omission only) 
#######

df_omission = df_clean[df_clean["condition"]==4]
data_omission = df_omission.drop(['condition', 'electrode','sub'], axis=1)
data_omission = data_omission.to_numpy()

df_ERPdata = df_clean.drop(df_clean[df_clean["condition"]==4].index)
#to replace moy by std data['condition'] = data['condition'].replace(7, 6)
data_ERP = df_ERPdata.drop(['condition', 'electrode','sub'], axis=1)
data_ERP = data_ERP.to_numpy()



####
##visualization of conditions
###

familiarization = df_ERPdata[df_ERPdata['condition']==3]
familiarization =familiarization.groupby('electrode', as_index=False).mean()
familiarization = familiarization.drop(['condition', 'sub','electrode'], axis=1)

deviant = df_ERPdata[df_ERPdata['condition']==2]
deviant =deviant.groupby('electrode', as_index=False).mean()
deviant = deviant.drop(['condition', 'sub','electrode'], axis=1)

postom = df_ERPdata[df_ERPdata['condition']==5]
postom =postom.groupby('electrode', as_index=False).mean()
postom = postom.drop(['condition', 'sub','electrode'], axis=1)

standard = df_ERPdata[df_ERPdata['condition']==6]
standard =standard.groupby('electrode', as_index=False).mean()
standard = standard.drop(['condition', 'sub','electrode'], axis=1)

plt.plot(np.mean(standard.iloc[[28,35]][:]))
plt.show()

########
## first step : compute full PCA to get needed nb of components (Kaiser method)
#######

kaiser_model = FactorAnalysis()
kaiser_model.fit(data_ERP)

# get how many components needed to reach 99% of explained variance
perc_var_explained = 0
n_components=0
components_variance = []
matrix = np.sum(kaiser_model.components_**2, axis=1)

for i in range(len(matrix)):
  variance_comp = (100*matrix[i])/np.sum(matrix)
  components_variance.append(variance_comp)
  perc_var_explained += variance_comp
  n_components = i
  if perc_var_explained>99:
    print(n_components, " Components needed")
    break

components_variance = np.array(components_variance)
########
## first step : compute full PCA to get needed nb of components (Kaiser method)
#######

model = FactorAnalysis(n_components=n_components, rotation='varimax')
model.fit(data_ERP)
components = model.components_

#######
##plot components
#######

#if components[n] as a majority of negative loadings, invert
def count_signs(loadings):
	negative_loadings= 0
	for ts in loadings:
		if ts < 0:
		    negative_loadings += 1
	return negative_loadings

pos_loadings = model.components_
for comp in range(0,len(components)):
	negative_loadings = count_signs(components[comp])
	if negative_loadings> 500:
    	pos_loadings[comp] = np.negative(components[comp])


legend = ['comp1','comp2','comp3','comp4','comp5','comp6','comp7','comp8','comp9']

# plot the loadings:
def plot_ts(loadings):
  plt.close('all')
  for ind, comp in enumerate(loadings):
    plt.plot(range(0, 1000), comp, linewidth=3)
  plt.xlabel("Time series (ms)")
  plt.ylabel("Loadings")
  plt.xticks(ticks=range(0,999,99), labels =['-100','0','100','200','300','400','500','600','700','800','900'])
  plt.legend(legend)
  plt.savefig("figures/FA_loadings_typiques.png")

plot_ts(pos_loadings)
plt.show()


##############
##save componenent 1 by 1
##############

#save components
for ind, comp in enumerate(pos_loadings):
  plt.close('all')
  plt.plot(comp)
  plt.xticks(ticks=range(0,999,99), labels =['-100','0','100','200','300','400','500','600','700','800','900'])
  plt.yticks(ticks=range(-3,5,1), labels =['','','','0','','','',''])
  plt.ylim([-2,7])
  plt.ylabel("arbitrary unit")
  plt.xlabel("Time series (ms)")
  plt.title("comp_" + str(ind) + " explained variance = " + str(round(components_variance[ind],3)))
  plt.savefig("figures/FA_typique_pos_comp_" + str(ind) + ".png".format("PNG"))