########################
# LOADING DATA
########################
#cd Z:\Bureau\data_analysis
from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import FactorAnalysis
from scipy import stats
from statsmodels.stats.multitest import multipletests
import random

# load typical matlab file 
data = loadmat('data/typical/Typical_PCA.mat')['Typical_PCA'] # 27090*1000
df_rawdata = pd.DataFrame(data)

##############
## First step : construct pandas dataframe with label of each row 
##############
myages = list()
two_yo = [2]*12 #12 participants of two years old
myages.extend(two_yo)
four_yo = [4]*12 #12 participants of four years old
myages.extend(four_yo)
six_yo = [6]*6 #6 participants of six years old
myages.extend(six_yo)

nb_subjects = 30 #total subjects

age = []
sub = []
for i in range(nb_subjects): 
    new_sub = [i+1] * 129* 7 #129 EEG channels and 7 conditions
    new_age = [myages[i]] * 129 * 7
    sub += new_sub
    age += new_age

condition = []
for j in range(nb_subjects): 
  for i in range(1,8): #7 conditions
    new_condition = [i] * 129 #EEG channels
    condition += new_condition

electrode = list(range(1, 130)) * nb_subjects *7


df_rawdata['condition'] = condition
df_rawdata['electrode'] = electrode
df_rawdata['sub'] = sub
df_rawdata['age'] = age

df_rawdata #check data


########
###Resolve class imbalance  by upsampling six years old
########
df_data =df_rawdata.drop(df_rawdata[df_rawdata['age']==6].index)

#Random upsampling of subjects

mylist = list(range(25,31))
random.seed(1)
upsample = random.choices(mylist,k=12)
#output = [25, 30, 29, 26, 27, 27, 28, 29, 25, 25, 30, 27]

sixyo_resampled = pd.DataFrame(columns=df_rawdata.columns)
for sub in upsample:
  sixyo_resampled = pd.concat([sixyo_resampled,df_rawdata[df_rawdata['sub']==sub]])
  
#change subject number
sub = []
for i in range(25,37): 
    new_sub = [i] * 129* 7 #129 EEG channels and 7 conditions
    sub += new_sub

sixyo_resampled['sub']=sub

df_data = df_data.append(sixyo_resampled)

##############
##Second step : Remove bad channels and eye channels
##############

###create a standardized df
df_standardized = (df_data - np.mean(df_data)) / np.std(df_data)

#find channel with data > 3 std
df_badchan = df_standardized[df_standardized>3]
df_badchan = df_badchan.replace(np.nan,0)

#cumsum to easily find those channels in the last column
df_badchan = df_badchan.cumsum(axis=1)

#save
#df_badchan.to_excel("data/typical/df_typ_badchan.xlsx") #for manual checking

#get the last column
df_mybadchan = df_badchan[999]

#find index of non zero (bad channel) rwos
df_mybadchan = df_mybadchan.index[df_mybadchan>0]

#delete bad channels
df_nobadchannel = df_data.drop(index=df_mybadchan)

#save
#df_nobadchannel.to_excel("data/typical/df_typ_nobadchannel.xlsx")

#delete eye channels
df_clean = df_nobadchannel.drop(df_nobadchannel[df_nobadchannel['electrode'] ==125].index)
df_clean = df_clean.drop(df_clean[df_clean['electrode'] ==126].index)
df_clean = df_clean.drop(df_clean[df_clean['electrode'] ==127].index)
df_clean = df_clean.drop(df_clean[df_clean['electrode'] ==128].index)

#save
#df_clean.to_excel("data/typical/df_typ_clean.xlsx")

##############
## Third step : separate data for PCA1 (data without omission) and PCA2 (omission only) 
##############

#discard omission data
df_ERPdata = df_clean.drop(df_clean[df_clean["condition"]==4].index)
df_ERPdata = df_ERPdata.reset_index(drop=True)

df_omi = df_clean[df_clean['condition']==4]
df_omi = df_omi.reset_index(drop=True)

#save new dfs
#df_ERPdata.to_excel("data/typical/ERPdata/df_ERP_typ.xlsx")
#df_omi.to_excel("data/typical/omission/df_omi_typ.xlsx")

#df to numpy to do the PCA
ERPdata = df_ERPdata.drop(['condition', 'electrode','sub','age'], axis=1)
ERPdata = ERPdata.to_numpy()

##########
##visualization of conditions
##########

familiarization = df_ERPdata[df_ERPdata['condition']==3]
familiarization = familiarization.drop(['condition', 'sub','age'], axis=1)
familiarization =familiarization.groupby('electrode', as_index=False).mean()
familiarization = familiarization.drop(['electrode'], axis=1)

deviant = df_ERPdata[df_ERPdata['condition']==2]
deviant = deviant.drop(['condition', 'sub','age'], axis=1)
deviant =deviant.groupby('electrode', as_index=False).mean()
deviant = deviant.drop(['electrode'], axis=1)

postom = df_ERPdata[df_ERPdata['condition']==5]
postom = postom.drop(['condition', 'sub','age'], axis=1)
postom =postom.groupby('electrode', as_index=False).mean()
postom = postom.drop(['electrode'], axis=1)

standard = df_ERPdata[df_ERPdata['condition']==7]
standard = standard.drop(['condition', 'sub','age'], axis=1)
standard =standard.groupby('electrode', as_index=False).mean()
standard = standard.drop(['electrode'], axis=1)

plt.plot(np.mean(deviant.iloc[[4,5,6,11,12,105,111,124]][:]))
plt.plot(np.mean(standard.iloc[[4,5,6,11,12,105,111,124]][:]))
plt.plot(np.mean(familiarization.iloc[[4,5,6,11,12,105,111,124]][:]))
plt.xticks(range(0,999,99),('-100','0','100','200','300','400','500','600','700','800','900'), fontsize=12)
plt.ylim(-3,2)
plt.gca().invert_yaxis()
plt.legend(['dev', 'std', 'fam'])
plt.show()

plt.plot(np.mean(deviant.iloc[[35,40,41,45,46,50,51,52]][:]))
plt.plot(np.mean(standard.iloc[[35,40,41,45,46,50,51,52]][:]))
plt.plot(np.mean(familiarization.iloc[[35,40,41,45,46,50,51,52]][:]))
plt.xticks(range(0,999,99),('-100','0','100','200','300','400','500','600','700','800','900'), fontsize=12)
plt.ylim(-3,2)
plt.gca().invert_yaxis()
plt.legend(['dev', 'std','fam'])
plt.show()

##############
## Forth step : compute full PCA to get needed nb of components for 99% of variance
##############

kaiser_model = FactorAnalysis()
kaiser_model.fit(ERPdata)

# get n components needed to reach 99% of explained variance
perc_var_explained = 0
n_components=0
components_variance = []
matrix = np.sum(kaiser_model.components_**2, axis=1)

for i in range(len(matrix)):
  variance_comp = (100*matrix[i])/np.sum(matrix)
  perc_var_explained += variance_comp
  n_components = i
  if perc_var_explained>99:
    print(n_components, " Components needed")
    break
  else:
      components_variance.append(variance_comp)

components_variance = np.round(np.array(components_variance),decimals=2)

#scree plot
cum_variance = np.cumsum(components_variance)
list_components = list(range(1,n_components+1))

plt.scatter(list_components, cum_variance, color='k')
plt.axhline(y=99, c='r')
plt.yticks(list(range(50,110,10)))
plt.xlabel("Components")
plt.ylabel("Explained variance (%)")
plt.title("Explained variance per component")
plt.savefig("figures/typical/ERPdata/ERP_typ_screeplot.png")
plt.show()


df_components_variance = pd.DataFrame(components_variance)

#save
#df_components_variance.to_excel("data/typical/ERPdata/df_ERP_typ_components_variance.xlsx")


##############
## Fifth step : compute PCA1
##############

model = FactorAnalysis(n_components=n_components, rotation='varimax')
model.fit(ERPdata)
components = model.components_

df_components = pd.DataFrame(components) #nb_components * 1000 time series

#save
#df_components.to_excel("data/typical/dataERP/df_ERP_components_typical.xlsx")

##############
##Sixth step : plot and save components loadings
##############

#if components[n] peak is negative, invert
pos_loadings = model.components_
for ind, comp in enumerate(pos_loadings):
  peakmax = max(comp)
  peakmin = abs(min(comp))
  if peakmax < peakmin:
    pos_loadings[ind] = np.negative(comp)


# plot the loadings:
def plot_ts(loadings):
  plt.close('all')
  for ind, comp in enumerate(loadings):
    plt.plot(range(0, 1000), comp, linewidth=3)
  plt.xlabel("Time series (ms)")
  plt.ylabel("Loadings")
  plt.xticks(ticks=range(0,999,99), labels =['-100','0','100','200','300','400','500','600','700','800','900'])
  plt.savefig("figures/typical/FA_loadings_ERP_typ.png")

plot_ts(pos_loadings)
plt.show()

##components on by one
#for ind, comp in enumerate(pos_loadings):
#  plt.close('all')
#  plt.plot(comp)
#  plt.xticks(ticks=range(0,999,99), labels =['-100','0','100','200','300','400','500','600','700','800','900'])
#  plt.ylim([round(components.min().min())-0.5,round(components.max().max())+0.5])
#  plt.ylabel("arbitrary unit")
#  plt.xlabel("Time series (ms)")
#  plt.title("comp_" + str(ind) + " explained variance = " + str(round(components_variance[ind],3)))
#  plt.savefig("figures/typical/FA_loadings_typ_" + str(ind) + ".png".format("PNG"))

##Get max loadings to determine their latencies
loadings_max = {}
for ind, comp in enumerate(pos_loadings):
  load_max = np.argmax(pos_loadings[ind]) - 100
  loadings_max[ind+1] = load_max

sorted_max = dict(sorted(loadings_max.items(), key=lambda item: item[1]))
df_max = pd.DataFrame(data=sorted_max, index=[0]).T
df_max.to_excel('data/typical/ERPdata/df_ERP_max_loadings.xlsx')

##############
## Seventh step : build factor scores and get components' scores topographies
##############

factor_scores = model.fit_transform(ERPdata)
df_factor_scores = pd.DataFrame(factor_scores) #22385 data * 27 components

#save
#df_factor_scores.to_excel("data/typical/ERPdata/df_ERP_factor_scores_typical.xlsx")

df_scores = df_factor_scores

myelec = df_ERPdata['electrode'] #get clean electrodes
df_scores['electrode'] = myelec #add to df scores

df_scores =df_scores.groupby('electrode', as_index=False).mean()
df_scores = df_scores.drop(['electrode'],axis=1)

df_scores_min = df_scores.min().min()
df_scores_max = df_scores.max().max()

coordinates = pd.read_excel('EEG_coordinates.xlsx')
coordinates = coordinates.drop(coordinates[coordinates['electrode'] == 'E125'].index)
coordinates = coordinates.drop(coordinates[coordinates['electrode'] == 'E126'].index)
coordinates = coordinates.drop(coordinates[coordinates['electrode'] == 'E127'].index)
coordinates = coordinates.drop(coordinates[coordinates['electrode'] == 'E128'].index)
coordinates = coordinates.reset_index(drop=True)
coordinates = coordinates.drop(['electrode'],axis=1)

##score topographies
#for ind, comp in enumerate(df_scores):
#  circle = plt.Circle((0, 0.05), coordinates.y_coor.max()-0.05, color='k', fill=False, linewidth=2)
#  nose = plt.Polygon([(-0.3,coordinates.y_coor.max()), (0,coordinates.y_coor.max()*1.1), (0.3,coordinates.y_coor.max())], color='k', fill=False, linewidth=2)
#  plt.gca().add_patch(circle)
#  plt.gca().add_patch(nose)
#  plt.tricontourf(coordinates.x_coor,coordinates.y_coor, df_scores[comp], cmap='seismic',  levels=125, vmin=df_scores_min, vmax=df_scores_max)
#  plt.plot(coordinates.x_coor,coordinates.y_coor, 'k.', markersize=7)
#  plt.gca().set_frame_on(False)
#  plt.gca().set_xticks([])
#  plt.gca().set_yticks([])
#  plt.colorbar()
#  plt.title(ind)
#  #plt.savefig("figures/FA_ERP_typ_scores_topo_" + str(ind) + ".png".format("PNG"))
#  plt.show() 


##plot and save score topographies and latencies to choose components
def plot_scores(loadings, scores):
  fig,ax = plt.subplots(2,1,figsize=(4, 6),gridspec_kw={'height_ratios': [1, 2]})
  fig.suptitle('Component '+str(i+1)+ ', explained variance = '+str(components_variance[i])+'%')
  ax[0].plot(loadings)
  ax[0].set_ylim([loadings.min().min()-0.5, loadings.max().max()+0.5])
  ax[0].set_xlabel("Time series (ms)")
  ax[0].set_xticks(ticks=range(0,999,99))
  ax[0].set_xticklabels(['-100','0','100','200','300','400','500','600','700','800','900'])
  ax[1].tricontourf(coordinates.x_coor,coordinates.y_coor, scores, cmap='seismic', levels=125, alpha=0.9, vmin=df_scores_min, vmax=df_scores_max)
  ax[1].plot(coordinates.x_coor,coordinates.y_coor, 'k.', markersize=7)
  ax[1].set_axis_off()
  circle = plt.Circle((0, 0), coordinates.y_coor.max(), color='k', fill=False, linewidth=2)
  ax[1].add_patch(circle)
  ax[1].plot([-0.3,0,0.3,-0.3], [coordinates.y_coor.max(),coordinates.y_coor.max()*1.1,coordinates.y_coor.max(),coordinates.y_coor.max()], color='k')
  plt.tight_layout()

for i in range(len(components)):
  plot_scores(pos_loadings[i],df_scores[i])
  plt.savefig("figures/typical/ERPdata/FA_ERP_typ_comp_lat_topo_" + str(i+1) + ".png".format("PNG"))

#my components are 15 for N140 and 11 for P300

###############
##Plot scores per condition and per age
###############

##Plot components latencies and topographies per condition

df_scores_cond = df_factor_scores

myelectrode = df_ERPdata['electrode'] #get clean electrodes
df_scores_cond['electrode'] = myelectrode #add to df scores
mycond = df_ERPdata['condition']
df_scores_cond['condition'] = mycond

def scores_percond(df, cond):
  df_cond = df[df['condition']==cond]
  df_cond = df_cond.drop(['condition'],axis=1)
  df_cond = df_cond.groupby(['electrode'], as_index=False).mean()
  df_cond = df_cond.drop(['electrode'],axis=1)
  return df_cond

df_scores_std = scores_percond(df_scores_cond,7)
df_scores_dev = scores_percond(df_scores_cond,2)
df_scores_MMN = df_scores_dev - df_scores_std
df_scores_fam = scores_percond(df_scores_cond,3)
df_scores_con = scores_percond(df_scores_cond,1)
df_scores_pom = scores_percond(df_scores_cond,5)

plt.close('all')
for i in range(27):
  plot_scores(pos_loadings[i],df_scores_std[i])
  plt.savefig("figures/typical/ERPdata/percond/FA_typ_std_lat_topo_" + str(i+1) + ".png".format("PNG"))

plt.close('all')
for i in range(27):
  plot_scores(pos_loadings[i],df_scores_dev[i])
  plt.savefig("figures/typical/ERPdata/percond/FA_typ_dev_lat_topo_" + str(i+1) + ".png".format("PNG"))

plt.close('all')
for i in range(27):
  plot_scores(pos_loadings[i],df_scores_MMN[i])
  plt.savefig("figures/typical/ERPdata/percond/FA_typ_MMN_lat_topo_" + str(i+1) + ".png".format("PNG"))

plt.close('all')
for i in range(27):
  plot_scores(pos_loadings[i],df_scores_fam[i])
  plt.savefig("figures/typical/ERPdata/percond/FA_typ_fam_lat_topo_" + str(i+1) + ".png".format("PNG"))

plt.close('all')
for i in range(27):
  plot_scores(pos_loadings[i],df_scores_con[i])
  plt.savefig("figures/typical/ERPdata/percond/FA_typ_con_lat_topo_" + str(i+1) + ".png".format("PNG"))


plt.close('all')
for i in range(27):
  plot_scores(pos_loadings[i],df_scores_pom[i])
  plt.savefig("figures/typical/ERPdata/percond/FA_typ_pom_lat_topo_" + str(i+1) + ".png".format("PNG"))

plt.close('all')

##plot components latencies and topographies per age

df_scores_age = pd.DataFrame(factor_scores)

#get topographies per age
myelectrode = df_ERPdata['electrode'] #get clean electrodes
df_scores_age['electrode'] = myelectrode #add to df scores
myage = df_ERPdata['age'] #get clean age
df_scores_age['age'] = myage

def scores_perage(df, age):
  df_age = df[df['age']==age]
  df_age = df_age.drop(['age'],axis=1)
  df_age = df_age.groupby(['electrode'], as_index=False).mean()
  df_age = df_age.drop(['electrode'],axis=1)
  return df_age

df_scores_two = scores_perage(df_scores_age,2)
df_scores_four = scores_perage(df_scores_age,4)
df_scores_six = scores_perage(df_scores_age,6)


for i in range(len(components)):
  plot_scores(pos_loadings[i],df_scores_two[i])
  plt.savefig("figures/typical/ERPdata/perage/comp_typ_two_lat_topo_" + str(i+1) + ".png".format("PNG"))

for i in range(len(components)):
  plot_scores(pos_loadings[i],df_scores_four[i])
  plt.savefig("figures/typical/ERPdata/perage/comp_typ_four_lat_topo_" + str(i+1) + ".png".format("PNG"))

for i in range(len(components)):
  plot_scores(pos_loadings[i],df_scores_six[i])
  plt.savefig("figures/typical/ERPdata/perage/comp_typ_six_lat_topo_" + str(i+1) + ".png".format("PNG"))



######################
##Tenth step : Extract data for statistics
######################


##All subjects analysis

#define two topographies first
somato = [36,41,42,46,47,51,52,53]
frontal = [5, 6, 7, 12, 13, 106, 112,129]
mycomponents = ([14,10]) #my components are 15 for N140 and 11 for P300

def scores2stat(df,condition,electrode,factor_scores,component):
  mystat = df[df['condition']==condition][df['electrode'].isin(electrode)][df.columns[0:1000]].index
  mystat = factor_scores[mystat]
  mystat = mystat.T
  mystat = mystat[component]
  return mystat

control_somato = scores2stat(df_ERPdata,1,somato,factor_scores,14)
deviant_somato = scores2stat(df_ERPdata,2,somato,factor_scores,14)
familiarization_somato = scores2stat(df_ERPdata,3,somato,factor_scores,14)
standard_somato = scores2stat(df_ERPdata,7,somato,factor_scores,14)

control_frontal = scores2stat(df_ERPdata,1,frontal,factor_scores,10)
deviant_frontal = scores2stat(df_ERPdata,2,frontal,factor_scores,10)
familiarization_frontal = scores2stat(df_ERPdata,3,frontal,factor_scores,10)
standard_frontal = scores2stat(df_ERPdata,7,frontal,factor_scores,10)


##Compare components per age

df_scores_age = pd.DataFrame(factor_scores)

def comp_perage(df,age,electrode,mycomponents):
  myindex = df[df['age']==age][df['electrode'].isin(electrode)][df.columns[0:1000]].index
  mystat = df_scores_age.iloc[myindex]
  mystat = mystat[mycomponents]
  return mystat


two_scores_mycomp = comp_perage(df_ERPdata,2,somato,mycomponents)
four_scores_mycomp = comp_perage(df_ERPdata,4,somato,mycomponents)
six_scores_mycomp = comp_perage(df_ERPdata,6,somato,mycomponents)


##Compare condition per age










######################
##Tenth step : Statistics
######################

tests = [(familiarization_somato, control_somato),(familiarization_frontal, control_frontal), (deviant_somato, standard_somato),(deviant_frontal, standard_frontal),(two_scores_mycomp[14],four_scores_mycomp[14]),(two_scores_mycomp[14],six_scores_mycomp[14]),(four_scores_mycomp[14],six_scores_mycomp[14]),(two_scores_mycomp[10],four_scores_mycomp[10]),(two_scores_mycomp[10],six_scores_mycomp[10]),(four_scores_mycomp[10],six_scores_mycomp[10])]
n_tests = len(tests)

df_p_value = pd.DataFrame(columns=['p_values'], index=['RS_somato','RS_frtl','MMN_somato','MMN_frtl','twofour_N140','twosix_N140','foursix_N140','twofour_P300','twosix_P300','foursix_P300'])
p_values = []
for ind, samples in enumerate(tests):
  sample1 = samples[0]
  sample2 = samples[1]
  stat, p = stats.wilcoxon(sample1, sample2)
  df_p_value.iloc[ind] = p
  p_values.append(p)

p_values = np.array(p_values)

significant, corrected_p_values, _, _ = multipletests(p_values, method='holm')

df_p_value['significant'] = significant
df_p_value['corrected'] = corrected_p_values























myindex = df_ERPdata[df_ERPdata['age']==2][df_ERPdata['electrode'].isin([36,41,42,46,47,51,52,53])][df_ERPdata.columns[0:1000]].index





df_scores_age = pd.DataFrame(factor_scores)

def scores_age2stat(df,condition,age,electrode,factor_scores):
  mystat = df[df['condition']==condition][df['age']==age][df['electrode'].isin(electrode)][df.columns[0:1000]].index
  mystat = factor_scores.iloc[mystat]
  mystat = mystat[mycomponents]
  return mystat


two_control_somato = scores_age2stat(df_ERPdata,1,2,somato,df_scores_age)
two_deviant_somato = scores_age2stat(df_ERPdata,2,2,somato,df_scores_age)
two_familiarization_somato = scores_age2stat(df_ERPdata,3,2,somato,df_scores_age)
two_standard_somato = scores_age2stat(df_ERPdata,7,2,somato,df_scores_age)
two_control_frontal = scores_age2stat(df_ERPdata,1,2,frontal,df_scores_age)
two_deviant_frontal = scores_age2stat(df_ERPdata,2,2,frontal,df_scores_age)
two_familiarization_frontal = scores_age2stat(df_ERPdata,3,2,frontal,df_scores_age)
two_standard_frontal = scores_age2stat(df_ERPdata,7,2,frontal,df_scores_age)

four_control_somato = scores_age2stat(df_ERPdata,1,4,somato,df_scores_age)
four_deviant_somato = scores_age2stat(df_ERPdata,2,4,somato,df_scores_age)
four_familiarization_somato = scores_age2stat(df_ERPdata,3,4,somato,df_scores_age)
four_standard_somato = scores_age2stat(df_ERPdata,7,4,somato,df_scores_age)
four_control_frontal = scores_age2stat(df_ERPdata,1,4,frontal,df_scores_age)
four_deviant_frontal = scores_age2stat(df_ERPdata,2,4,frontal,df_scores_age)
four_familiarization_frontal = scores_age2stat(df_ERPdata,3,4,frontal,df_scores_age)
four_standard_frontal = scores_age2stat(df_ERPdata,7,4,frontal,df_scores_age)

six_control_somato = scores_age2stat(df_ERPdata,1,6,somato,df_scores_age)
six_deviant_somato = scores_age2stat(df_ERPdata,2,6,somato,df_scores_age)
six_familiarization_somato = scores_age2stat(df_ERPdata,3,6,somato,df_scores_age)
six_standard_somato = scores_age2stat(df_ERPdata,7,6,somato,df_scores_age)
six_control_frontal = scores_age2stat(df_ERPdata,1,6,frontal,df_scores_age)
six_deviant_frontal = scores_age2stat(df_ERPdata,2,6,frontal,df_scores_age)
six_familiarization_frontal = scores_age2stat(df_ERPdata,3,6,frontal,df_scores_age)
six_standard_frontal = scores_age2stat(df_ERPdata,7,6,frontal,df_scores_age)

#############
##Statistics per age
############














################
##PER AGE
################


for i in range(27): 
  fig,ax = plt.subplots(figsize=(12, 6))
  grid = plt.GridSpec(3, 3, wspace =0.3, hspace = 0.3)
  ax0 = plt.subplot(grid[0, 0:2])
  ax1 = plt.subplot(grid[1:, 0])
  ax2 = plt.subplot(grid[1:, 1])
  ax3 = plt.subplot(grid[1:, 2])
  fig.suptitle('Component '+str(i+1)+ ', explained variance = '+str(components_variance[i])+'%')
  ax0.plot(pos_loadings[i])
  ax0.set_ylim([pos_loadings.min().min()-0.5, pos_loadings.max().max()+0.5])
  ax0.set_xlabel("Time series (ms)")
  ax0.set_xticks(ticks=range(0,999,99))
  ax0.set_xticklabels(['-100','0','100','200','300','400','500','600','700','800','900'])
  ax1.tricontourf(coordinates.x_coor,coordinates.y_coor, df_scores_two[i], cmap='seismic', levels=125, alpha=0.9, vmin=df_scores_min, vmax=df_scores_max)
  ax2.tricontourf(coordinates.x_coor,coordinates.y_coor, df_scores_four[i], cmap='seismic', levels=125, alpha=0.9, vmin=df_scores_min, vmax=df_scores_max)
  ax3.tricontourf(coordinates.x_coor,coordinates.y_coor, df_scores_six[i], cmap='seismic', levels=125, alpha=0.9, vmin=df_scores_min, vmax=df_scores_max)
  ax1.plot(coordinates.x_coor,coordinates.y_coor, 'k.', markersize=7)
  ax2.plot(coordinates.x_coor,coordinates.y_coor, 'k.', markersize=7)
  ax3.plot(coordinates.x_coor,coordinates.y_coor, 'k.', markersize=7)
  ax1.set_axis_off()
  ax2.set_axis_off()
  ax3.set_axis_off()
  circle1 = plt.Circle((0, 0), coordinates.y_coor.max(), color='k', fill=False, linewidth=2)
  circle2 = plt.Circle((0, 0), coordinates.y_coor.max(), color='k', fill=False, linewidth=2)
  circle3 = plt.Circle((0, 0), coordinates.y_coor.max(), color='k', fill=False, linewidth=2)
  ax1.add_patch(circle1)
  ax2.add_patch(circle2)
  ax3.add_patch(circle3)
  ax1.plot([-0.3,0,0.3,-0.3], [coordinates.y_coor.max(),coordinates.y_coor.max()*1.1,coordinates.y_coor.max(),coordinates.y_coor.max()], color='k')
  ax2.plot([-0.3,0,0.3,-0.3], [coordinates.y_coor.max(),coordinates.y_coor.max()*1.1,coordinates.y_coor.max(),coordinates.y_coor.max()], color='k')
  ax3.plot([-0.3,0,0.3,-0.3], [coordinates.y_coor.max(),coordinates.y_coor.max()*1.1,coordinates.y_coor.max(),coordinates.y_coor.max()], color='k')
  ax1.text(-1,-3.5,'two years old')
  ax2.text(-1,-3.5,'four years old')
  ax3.text(-1,-3.5,'six years old')
  plt.tight_layout()
  plt.savefig("figures/FA_typique_per_age_lat_topo_" + str(i) + ".png".format("PNG"))


def df_perage_percond(df, age, condition):
  df_age = df[df['age']==age]
  df_age = df_age.drop(['age'],axis=1)
  df_cond = df_age[df_age['condition']==condition]
  df_cond = df_cond.drop(['condition'],axis=1)
  df_cond = df_cond.groupby(['electrode'], as_index=False).mean()
  df_cond = df_cond.drop(['electrode'],axis=1)
  return df_cond

mycond = df_ERPdata['condition'] #get clean age
df_scores_age['condition'] = mycond


df_scores_two_fam = df_perage_percond(df_scores_age, 2, 3)
df_scores_two_con = df_perage_percond(df_scores_age, 2, 1)
df_scores_two_std = df_perage_percond(df_scores_age, 2, 7)
df_scores_two_dev = df_perage_percond(df_scores_age, 2, 2)
df_scores_four_fam = df_perage_percond(df_scores_age, 4, 3)
df_scores_four_con = df_perage_percond(df_scores_age, 4, 1)
df_scores_four_std = df_perage_percond(df_scores_age, 4, 7)
df_scores_four_dev = df_perage_percond(df_scores_age, 4, 2)
df_scores_six_fam = df_perage_percond(df_scores_age, 6, 3)
df_scores_six_con = df_perage_percond(df_scores_age, 6, 1)
df_scores_six_std = df_perage_percond(df_scores_age, 6, 7)
df_scores_six_dev = df_perage_percond(df_scores_age, 6, 2)


#figure 4 brain, comp scores at two yo, std, dev, fam, con

for i in range(27):
  fig,ax = plt.subplots(1,4, figsize=(18,6))
  fig.suptitle('component for 2 yo in P1')
  ax[0].tricontourf(coordinates.x_coor,coordinates.y_coor, df_scores_two_std[i], cmap='seismic', levels=125, alpha=0.9, vmin=df_scores_min, vmax=df_scores_max)
  ax[1].tricontourf(coordinates.x_coor,coordinates.y_coor, df_scores_two_dev[i], cmap='seismic', levels=125, alpha=0.9, vmin=df_scores_min, vmax=df_scores_max)
  ax[2].tricontourf(coordinates.x_coor,coordinates.y_coor, df_scores_two_fam[i], cmap='seismic', levels=125, alpha=0.9, vmin=df_scores_min, vmax=df_scores_max)
  ax[3].tricontourf(coordinates.x_coor,coordinates.y_coor, df_scores_two_con[i], cmap='seismic', levels=125, alpha=0.9, vmin=df_scores_min, vmax=df_scores_max)
  ax[0].plot(coordinates.x_coor,coordinates.y_coor, 'k.', markersize=7)
  ax[1].plot(coordinates.x_coor,coordinates.y_coor, 'k.', markersize=7)
  ax[2].plot(coordinates.x_coor,coordinates.y_coor, 'k.', markersize=7)
  ax[3].plot(coordinates.x_coor,coordinates.y_coor, 'k.', markersize=7)
  ax[0].set_axis_off()
  ax[1].set_axis_off()
  ax[2].set_axis_off()
  ax[3].set_axis_off()
  circle1 = plt.Circle((0, 0), coordinates.y_coor.max(), color='k', fill=False, linewidth=2)
  circle2 = plt.Circle((0, 0), coordinates.y_coor.max(), color='k', fill=False, linewidth=2)
  circle3 = plt.Circle((0, 0), coordinates.y_coor.max(), color='k', fill=False, linewidth=2)
  circle4 = plt.Circle((0, 0), coordinates.y_coor.max(), color='k', fill=False, linewidth=2)
  ax[0].add_patch(circle1)
  ax[1].add_patch(circle2)
  ax[2].add_patch(circle3)
  ax[3].add_patch(circle4)
  ax[0].plot([-0.3,0,0.3,-0.3], [coordinates.y_coor.max(),coordinates.y_coor.max()*1.1,coordinates.y_coor.max(),coordinates.y_coor.max()], color='k')
  ax[1].plot([-0.3,0,0.3,-0.3], [coordinates.y_coor.max(),coordinates.y_coor.max()*1.1,coordinates.y_coor.max(),coordinates.y_coor.max()], color='k')
  ax[2].plot([-0.3,0,0.3,-0.3], [coordinates.y_coor.max(),coordinates.y_coor.max()*1.1,coordinates.y_coor.max(),coordinates.y_coor.max()], color='k')
  ax[3].plot([-0.3,0,0.3,-0.3], [coordinates.y_coor.max(),coordinates.y_coor.max()*1.1,coordinates.y_coor.max(),coordinates.y_coor.max()], color='k')
  plt.tight_layout()
  plt.show()
  #plt.savefig("figures/FA_typique_per_age_lat_topo_" + str(i) + ".png".format("PNG"))













































































#####################################################
##OMISSION ANALYSIS
#####################################################

df_omission = df_clean[df_clean["condition"]==4]
df_omission = df_omission.reset_index(drop=True)
#save#df_omission.to_excel("df_omission_typical.xlsx")
data_omission = df_omission.drop(['condition', 'electrode','sub','age'], axis=1)
data_omission = data_omission.to_numpy()


##############
## Fifth step : compute full PCA to get needed nb of components for 99% of variance
##############

kaiser_model_omi = FactorAnalysis()
kaiser_model_omi.fit(data_omission)

# get how many components needed to reach 99% of explained variance
perc_var_explained_omi = 0
n_components_omi=0
components_variance_omi = []
matrix_omi = np.sum(kaiser_model_omi.components_**2, axis=1)

for i in range(len(matrix)):
  variance_comp_omi = (100*matrix_omi[i])/np.sum(matrix_omi)
  perc_var_explained_omi += variance_comp_omi
  n_components_omi = i
  if perc_var_explained_omi>99:
    print(n_components_omi, " Components needed")
    break
  else:
      components_variance_omi.append(variance_comp_omi)

components_variance_omi = np.round(np.array(components_variance_omi),decimals=2)

#scree plot
cum_variance_omi = np.cumsum(components_variance_omi)
list_components_omi = list(range(1,n_components_omi+1))

plt.scatter(list_components_omi, cum_variance_omi, color='k')
plt.axhline(y=99, c='r')
plt.yticks(list(range(50,110,10)))
plt.xlabel("Components")
plt.ylabel("Explained variance (%)")
plt.title("Explained variance per component")
plt.savefig("figures/scree_plot_omi_typiques.png")
plt.show()


df_components_variance_omi = pd.DataFrame(components_variance_omi)
#save#df_components_variance.to_excel("components_variance_typical.xlsx")
##############
## Sixth step : compute PCA with number of components
##############

model_omi = FactorAnalysis(n_components=n_components_omi, rotation='varimax')
model_omi.fit(data_omission)
components_omi = model_omi.components_

df_components_omi = pd.DataFrame(components_omi)
#save#df_components_omi.to_excel("components_omi_typical.xlsx")
##############
##Seventh step : plot components
##############

#if components[n] peak is negative, invert

pos_loadings_omi = model_omi.components_
for ind, comp in enumerate(pos_loadings_omi):
  peakmax = max(comp)
  peakmin = abs(min(comp))
  if peakmax < peakmin:
    pos_loadings_omi[ind] = np.negative(comp)


# plot the loadings:
def plot_ts_omi(loadings):
  plt.close('all')
  for ind, comp in enumerate(loadings):
    plt.plot(range(0, 1000), comp, linewidth=3)
  plt.xlabel("Time series (ms)")
  plt.ylabel("Loadings")
  plt.xticks(ticks=range(0,999,99), labels =['-500','-400','-300','-200','-100','0','100','200','300','400','500'])
  plt.savefig("figures/FA_loadings_omi_typiques.png")

plot_ts_omi(pos_loadings_omi)
plt.show()


##############
## Eigth step : save componenent 1 by 1
##############

#save components
for ind, comp in enumerate(pos_loadings_omi):
  plt.close('all')
  plt.plot(comp)
  plt.xticks(ticks=range(0,999,99), labels =['-500','-400','-300','-200','-100','0','100','200','300','400','500'])
  plt.ylim([round(components.min().min())-0.5,round(components.max().max())+0.5])
  plt.ylabel("arbitrary unit")
  plt.xlabel("Time series (ms)")
  plt.title("comp_" + str(ind) + " explained variance = " + str(round(components_variance_omi[ind],3)))
  plt.savefig("figures/FA_typique_omi_pos_comp_" + str(ind) + ".png".format("PNG"))


############
##Get max loadings
############
loadings_max_omi = {}
for ind, comp in enumerate(pos_loadings_omi):
  load_max = np.argmax(pos_loadings_omi[ind]) - 500
  loadings_max_omi[ind+1] = load_max

sorted_max_omi = dict(sorted(loadings_max_omi.items(), key=lambda item: item[1]))
df_max_omi = pd.DataFrame(data=sorted_max_omi, index=[0]).T
df_max_omi.to_excel('df_max_omi_loadings.xlsx')

##############
## Eleventh step : build factor scores and get components' scores topographies
##############

factor_scores_omi = model_omi.fit_transform(data_omission)

df_scores_omi = pd.DataFrame(factor_scores_omi)
#save#df_scores.to_excel("scores_typical.xlsx")

myelectrode = df_omission['electrode'] #get clean electrodes
df_scores_omi['electrode'] = myelectrode #add to df scores

df_scores_age_omi = df_scores_omi #keep for later

df_scores_omi =df_scores_omi.groupby('electrode', as_index=False).mean()
df_scores_omi = df_scores_omi.drop(['electrode'],axis=1)

df_scores_omi_min = df_scores_omi.min().min()
df_scores_omi_max = df_scores_omi.max().max()

coordinates = pd.read_excel('EEG_coordinates.xlsx')
coordinates = coordinates.drop(coordinates[coordinates['electrode'] == 'E125'].index)
coordinates = coordinates.drop(coordinates[coordinates['electrode'] == 'E126'].index)
coordinates = coordinates.drop(coordinates[coordinates['electrode'] == 'E127'].index)
coordinates = coordinates.drop(coordinates[coordinates['electrode'] == 'E128'].index)
coordinates = coordinates.reset_index(drop=True)
coordinates = coordinates.drop(['electrode'],axis=1)

#save score topographies
for ind, comp in enumerate(df_scores_omi):
  circle = plt.Circle((0, 0.05), coordinates.y_coor.max()-0.05, color='k', fill=False, linewidth=2)
  nose = plt.Polygon([(-0.3,coordinates.y_coor.max()), (0,coordinates.y_coor.max()*1.1), (0.3,coordinates.y_coor.max())], color='k', fill=False, linewidth=2)
  plt.gca().add_patch(circle)
  plt.gca().add_patch(nose)
  plt.tricontourf(coordinates.x_coor,coordinates.y_coor, df_scores_omi[comp], cmap='seismic',  levels=125, vmin=df_scores_omi_min, vmax=df_scores_omi_max)
  plt.plot(coordinates.x_coor,coordinates.y_coor, 'k.', markersize=7)
  plt.gca().set_frame_on(False)
  plt.gca().set_xticks([])
  plt.gca().set_yticks([])
  plt.colorbar()
  plt.title(ind)
  #plt.savefig("figures/FA_typique_scores_topo_" + str(ind) + ".png".format("PNG"))
  plt.show() 

########
##plot score topographies and latencies to choose components
#######

for i in range(26):
  fig,ax = plt.subplots(2,1,figsize=(4, 6),gridspec_kw={'height_ratios': [1, 2]})
  fig.suptitle('Component '+str(i+1)+ ', explained variance = '+str(components_variance_omi[i])+'%')
  ax[0].plot(pos_loadings_omi[i])
  ax[0].set_ylim([pos_loadings_omi.min().min()-0.5, pos_loadings_omi.max().max()+0.5])
  ax[0].set_xlabel("Time series (ms)")
  ax[0].set_xticks(ticks=range(0,999,99))
  ax[0].set_xticklabels(['-500','-400','-300','-200','-100','0','100','200','300','400','500'])
  ax[1].tricontourf(coordinates.x_coor,coordinates.y_coor, df_scores_omi[i], cmap='seismic', levels=125, alpha=0.9, vmin=df_scores_omi_min, vmax=df_scores_omi_max)
  ax[1].plot(coordinates.x_coor,coordinates.y_coor, 'k.', markersize=7)
  ax[1].set_axis_off()
  circle = plt.Circle((0, 0), coordinates.y_coor.max(), color='k', fill=False, linewidth=2)
  ax[1].add_patch(circle)
  ax[1].plot([-0.3,0,0.3,-0.3], [coordinates.y_coor.max(),coordinates.y_coor.max()*1.1,coordinates.y_coor.max(),coordinates.y_coor.max()], color='k')
  plt.tight_layout()
  plt.savefig("figures/FA_typique_omi_comp_lat_topo_" + str(i) + ".png".format("PNG"))


my_components = [19,21,24,23]


################
##PER AGE
################

#get topographies per age
myage = df_ERPdata['age'] #get clean age
df_scores_age['age'] = myage

df_scores_two = df_scores_age[df_scores_age['age']==2]
df_scores_two = df_scores_two.drop(['age'],axis=1)
df_scores_two =df_scores_two.groupby(['electrode'], as_index=False).mean()
df_scores_two = df_scores_two.drop(['electrode'],axis=1)

df_scores_four = df_scores_age[df_scores_age['age']==4]
df_scores_four = df_scores_four.drop(['age'],axis=1)
df_scores_four =df_scores_four.groupby(['electrode'], as_index=False).mean()
df_scores_four = df_scores_four.drop(['electrode'],axis=1)

df_scores_six = df_scores_age[df_scores_age['age']==6]
df_scores_six = df_scores_six.drop(['age'],axis=1)
df_scores_six =df_scores_six.groupby(['electrode'], as_index=False).mean()
df_scores_six = df_scores_six.drop(['electrode'],axis=1)


for i in range(27): #to do in a for loop !!!!
  fig,ax = plt.subplots(figsize=(12, 6))
  grid = plt.GridSpec(3, 3, wspace =0.3, hspace = 0.3)
  ax0 = plt.subplot(grid[0, 0:2])
  ax1 = plt.subplot(grid[1:, 0])
  ax2 = plt.subplot(grid[1:, 1])
  ax3 = plt.subplot(grid[1:, 2])
  fig.suptitle('Component '+str(i+1)+ ', explained variance = '+str(components_variance[i])+'%')
  ax0.plot(pos_loadings[i])
  ax0.set_ylim([pos_loadings.min().min()-0.5, pos_loadings.max().max()+0.5])
  ax0.set_xlabel("Time series (ms)")
  ax0.set_xticks(ticks=range(0,999,99))
  ax0.set_xticklabels(['-100','0','100','200','300','400','500','600','700','800','900'])
  ax1.tricontourf(coordinates.x_coor,coordinates.y_coor, df_scores_two[i], cmap='seismic', levels=125, alpha=0.9, vmin=df_scores_min, vmax=df_scores_max)
  ax2.tricontourf(coordinates.x_coor,coordinates.y_coor, df_scores_four[i], cmap='seismic', levels=125, alpha=0.9, vmin=df_scores_min, vmax=df_scores_max)
  ax3.tricontourf(coordinates.x_coor,coordinates.y_coor, df_scores_six[i], cmap='seismic', levels=125, alpha=0.9, vmin=df_scores_min, vmax=df_scores_max)
  ax1.plot(coordinates.x_coor,coordinates.y_coor, 'k.', markersize=7)
  ax2.plot(coordinates.x_coor,coordinates.y_coor, 'k.', markersize=7)
  ax3.plot(coordinates.x_coor,coordinates.y_coor, 'k.', markersize=7)
  ax1.set_axis_off()
  ax2.set_axis_off()
  ax3.set_axis_off()
  circle1 = plt.Circle((0, 0), coordinates.y_coor.max(), color='k', fill=False, linewidth=2)
  circle2 = plt.Circle((0, 0), coordinates.y_coor.max(), color='k', fill=False, linewidth=2)
  circle3 = plt.Circle((0, 0), coordinates.y_coor.max(), color='k', fill=False, linewidth=2)
  ax1.add_patch(circle1)
  ax2.add_patch(circle2)
  ax3.add_patch(circle3)
  ax1.plot([-0.3,0,0.3,-0.3], [coordinates.y_coor.max(),coordinates.y_coor.max()*1.1,coordinates.y_coor.max(),coordinates.y_coor.max()], color='k')
  ax2.plot([-0.3,0,0.3,-0.3], [coordinates.y_coor.max(),coordinates.y_coor.max()*1.1,coordinates.y_coor.max(),coordinates.y_coor.max()], color='k')
  ax3.plot([-0.3,0,0.3,-0.3], [coordinates.y_coor.max(),coordinates.y_coor.max()*1.1,coordinates.y_coor.max(),coordinates.y_coor.max()], color='k')
  ax1.text(-1,-3.5,'two years old')
  ax2.text(-1,-3.5,'four years old')
  ax3.text(-1,-3.5,'six years old')
  plt.tight_layout()
  plt.savefig("figures/FA_typique_per_age_lat_topo_" + str(i) + ".png".format("PNG"))





























# arrange data per condition and electrodes

control_somato = df_ERPdata[df_ERPdata['condition']==1][df_ERPdata['electrode'].isin([28,29,35,36,41,42,47,52])][df_ERPdata.columns[0:1000]]
deviant_somato = df_ERPdata[df_ERPdata['condition']==2][df_ERPdata['electrode'].isin([28,29,35,36,41,42,47,52])][df_ERPdata.columns[0:1000]]
familiarization_somato = df_ERPdata[df_ERPdata['condition']==3][df_ERPdata['electrode'].isin([28,29,35,36,41,42,47,52])][df_ERPdata.columns[0:1000]]
postom_somato = df_ERPdata[df_ERPdata['condition']==5][df_ERPdata['electrode'].isin([28,29,35,36,41,42,47,52])][df_ERPdata.columns[0:1000]]
standard_somato = df_ERPdata[df_ERPdata['condition']==6][df_ERPdata['electrode'].isin([28,29,35,36,41,42,47,52])][df_ERPdata.columns[0:1000]]
stimmoy_somato = df_ERPdata[df_ERPdata['condition']==7][df_ERPdata['electrode'].isin([28,29,35,36,41,42,47,52])][df_ERPdata.columns[0:1000]]

control_frontal = df_ERPdata[df_ERPdata['condition']==1][df_ERPdata['electrode'].isin([5, 6, 7, 12, 13, 106, 112,129])][df_ERPdata.columns[0:1000]]
deviant_frontal = df_ERPdata[df_ERPdata['condition']==2][df_ERPdata['electrode'].isin([5, 6, 7, 12, 13, 106, 112,129])][df_ERPdata.columns[0:1000]]
familiarization_frontal = df_ERPdata[df_ERPdata['condition']==3][df_ERPdata['electrode'].isin([5, 6, 7, 12, 13, 106, 112,129])][df_ERPdata.columns[0:1000]]
postom_frontal = df_ERPdata[df_ERPdata['condition']==5][df_ERPdata['electrode'].isin([5, 6, 7, 12, 13, 106, 112,129])][df_ERPdata.columns[0:1000]]
standard_frontal = df_ERPdata[df_ERPdata['condition']==6][df_ERPdata['electrode'].isin([5, 6, 7, 12, 13, 106, 112,129])][df_ERPdata.columns[0:1000]]
stimmoy_frontal = df_ERPdata[df_ERPdata['condition']==7][df_ERPdata['electrode'].isin([5, 6, 7, 12, 13, 106, 112,129])][df_ERPdata.columns[0:1000]]

##############
##Twelfth step : Statistics
##############

# perfom t-test

from scipy import stats

#statistics for all components at one

control_somato_stat = factor_scores[control_somato.index]
deviant_somato_stat = factor_scores[deviant_somato.index]
familiarization_somato_stat = factor_scores[familiarization_somato.index]
postom_somato_stat = factor_scores[postom_somato.index]
standard_somato_stat = factor_scores[standard_somato.index]
stimmoy_somato_stat = factor_scores[stimmoy_somato.index]

control_frontal_stat = factor_scores[control_frontal.index]
deviant_frontal_stat = factor_scores[deviant_frontal.index]
familiarization_frontal_stat = factor_scores[familiarization_frontal.index]
postom_frontal_stat = factor_scores[postom_frontal.index]
standard_frontal_stat = factor_scores[standard_frontal.index]
stimmoy_frontal_stat = factor_scores[stimmoy_frontal.index]


for comp in list(range(0,27)):
  var1 = familiarization_frontal_stat[:, comp]
  var2 = control_frontal_stat[:, comp]
  print(stats.ttest_rel(var1, var2))



#perform non parametric hypothesis testing


myfactor_scores = factor_scores[:,my_components]

control_somato_stat = myfactor_scores[control_somato.index]
deviant_somato_stat = myfactor_scores[deviant_somato.index]
familiarization_somato_stat = myfactor_scores[familiarization_somato.index]
postom_somato_stat = myfactor_scores[postom_somato.index]
standard_somato_stat = myfactor_scores[standard_somato.index]
stimmoy_somato_stat = myfactor_scores[stimmoy_somato.index]

control_frontal_stat = myfactor_scores[control_frontal.index]
deviant_frontal_stat = myfactor_scores[deviant_frontal.index]
familiarization_frontal_stat = myfactor_scores[familiarization_frontal.index]
postom_frontal_stat = myfactor_scores[postom_frontal.index]
standard_frontal_stat = myfactor_scores[standard_frontal.index]
stimmoy_frontal_stat = myfactor_scores[stimmoy_frontal.index]



mean_std_somato_comp19 = np.mean(standard_somato_stat[:,0])

from scipy.stats import bootstrap
stats.bootstrap(mean_std_somato_comp19, np.std,n_resamples=10)









datab = (mean_std_somato_comp19)  # samples must be in a sequence
res = bootstrap(datab, np.std, confidence_level=0.9,

                random_state=rng)

fig, ax = plt.subplots()
ax.hist(res.bootstrap_distribution, bins=25)
ax.set_title('Bootstrap Distribution')
ax.set_xlabel('statistic value')
ax.set_ylabel('frequency')
plt.show()