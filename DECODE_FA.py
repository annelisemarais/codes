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

##############
##First step : Detect bad channels that will be removed from analysis
##############

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

##############
## Second step : construct pandas dataframe with label of each row 
##############

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

##############
##Third step Discard eye channels (125:128) and bad channels  
##############
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

##############
## Fourth step : separate data for PCA1 (data without omission) and PCA2 (omission only) 
##############

df_omission = df_clean[df_clean["condition"]==4]
df_omission = df_omission.reset_index(drop=True)
data_omission = df_omission.drop(['condition', 'electrode','sub'], axis=1)
data_omission = data_omission.to_numpy()

df_ERPdata = df_clean.drop(df_clean[df_clean["condition"]==4].index)
df_ERPdata = df_ERPdata.reset_index(drop=True)
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

##############
## Fifth step : compute full PCA to get needed nb of components for 99% of variance
##############

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


##############
## Sixth step : compute PCA with number of components
##############

model = FactorAnalysis(n_components=n_components, rotation='varimax')
model.fit(data_ERP)
components = model.components_

##############
##Seventh step : plot components
##############

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
## Eigth step : save componenent 1 by 1
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


##############
## Nineth step : find max of each component, used to label them
##############

components_max_ind = np.argmax(pos_loadings, axis=1)
components_max = np.amax(pos_loadings, axis=1)

max_loadings = pos_loadings
for ind1, comp in enumerate(max_loadings):
	for ind2, ts in enumerate(comp):
		if ind2 != int(components_max_ind[ind1]):
			max_loadings[ind1,ind2] =0


plt.close('all')
plt.plot(max_loadings.T)
plt.xlabel("Time series (ms)")
plt.ylabel("Max loadings")
plt.xticks(ticks=range(0,999,99), labels =['-100','0','100','200','300','400','500','600','700','800','900'])
plt.legend(labels="123456789")
plt.savefig("figures/FA_max_loadings_typiques.png")
plt.show()

##############
## Tenth step : Match peak loadings with ERP peaks to label them
##either use components or max
##############

#In somatosensory areas /early components

#plt.plot(components[1].T)
#plt.plot(components[7].T)
#plt.plot(components[8].T)
plt.plot(components[10].T)
#plt.plot(components[11].T)
#plt.plot(components[13].T)
plt.plot(components[19].T)
#plt.plot(components[21].T)
#plt.plot(components[22].T)
plt.plot(components[26].T)
plt.plot(np.negative(np.mean(standard.iloc[[27,28,34,35,41,42,47]][:])),linewidth=2, color='k')
plt.plot(np.negative(np.mean(familiarization.iloc[[27,28,34,35,41,42,47]][:])),linewidth=2, color='k')
plt.plot(np.negative(np.mean(deviant.iloc[[27,28,34,35,41,42,47]][:])),linewidth=2, color='k')
plt.plot(np.negative(np.mean(postom.iloc[[27,28,34,35,41,42,47]][:])),linewidth=2, color='k')
plt.xticks(ticks=range(0,999,99), labels =['-100','0','100','200','300','400','500','600','700','800','900'])
plt.legend(["comp 10 N140", "comp 19 P100","comp 26 P50", "ERPs"])
plt.title('Labelling components for somatosensory activation')
plt.show()

#In frontal areas / late components

#plt.plot(components[0].T)
plt.plot(components[1].T)
#plt.plot(components[2].T)
#plt.plot(components[3].T)
#plt.plot(components[4].T)
#plt.plot(components[5].T)
#plt.plot(components[6].T)
#plt.plot(components[9].T)
#plt.plot(components[12].T)
#plt.plot(components[14].T)
plt.plot(components[20].T)
#plt.plot(components[25].T)
#plt.plot(components[27].T)
plt.plot(np.negative(np.mean(standard.iloc[[4,5,6,11,12,105,111]][:])),linewidth=2, color='k')
plt.plot(np.negative(np.mean(familiarization.iloc[[4,5,6,11,12,105,111]][:])),linewidth=2, color='k')
plt.plot(np.negative(np.mean(deviant.iloc[[4,5,6,11,12,105,111]][:])),linewidth=2, color='k')
plt.plot(np.negative(np.mean(postom.iloc[[4,5,6,11,12,105,111]][:])),linewidth=2, color='k')
plt.xticks(ticks=range(0,999,99), labels =['-100','0','100','200','300','400','500','600','700','800','900'])
plt.legend(["comp1 P300 ?", "comp 20 LPC", "ERPs"])
plt.title('Labelling components for frontal activation')
plt.show()


###########
##Not in the code
### cumsum of factor scores to understand how it works
##########

#df_scores = pd.DataFrame(factor_scores)

#frontal_comp_index = df_ERPdata[df_ERPdata['electrode'] == 7]
#frontal_comp_index_standard = frontal_comp_index.index[frontal_comp_index['condition']==2]

#frontal_comp_standard = df_scores.iloc[frontal_comp_index_standard]

#cs = np.cumsum(abs(frontal_comp_standard), axis=0)
#print(cs.iloc[-1])


##############
## Eleventh step : build factor scores
##############

factor_scores = model.fit_transform(data_ERP)


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


##############
##Twelfth step : Statistics
##############

# perfom t-test
from scipy import stats
stats.ttest_rel(var_ind_c1, var_ind_c3)