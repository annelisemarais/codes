__author__ = 'Anne-Lise Marais, see annelisemarais.github.io'
__publication__ = 'Marais, AL., Anquetil, A., Dumont, V., Roche-Labarbe, N. (2023). Somatosensory prediction in typical children from 2 to 6 years old'
__corresponding__ = 'nadege.roche@unicaen.fr'


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
from matplotlib.patches import Rectangle

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

#df_rawdata.to_csv("data/typical/df_rawdata.csv")


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

df_data = df_data.append(sixyo_resampled).reset_index(drop=True)

#save
#df_data.to_csv("data/typical/df_data_resampled.csv")

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
#df_badchan.to_csv("data/typical/df_typ_badchan.csv") #for manual checking

#get the last column
df_mybadchan = df_badchan[999]

#find index of non zero (bad channel) rwos
df_mybadchan = df_mybadchan.index[df_mybadchan>0]

#delete bad channels
df_nobadchannel = df_data.drop(index=df_mybadchan)

#save
#df_nobadchannel.to_csv("data/typical/df_typ_nobadchannel.csv")

#delete eye channels
df_clean = df_nobadchannel.drop(df_nobadchannel[df_nobadchannel['electrode'] ==125].index)
df_clean = df_clean.drop(df_clean[df_clean['electrode'] ==126].index)
df_clean = df_clean.drop(df_clean[df_clean['electrode'] ==127].index)
df_clean = df_clean.drop(df_clean[df_clean['electrode'] ==128].index)

#save
#df_clean.to_csv("data/typical/df_typ_clean.csv")

##############
## Third step : separate data for PCA1 (data without omission) and PCA2 (omission only) 
##############

#discard omission data
df_ERPdata = df_clean.drop(df_clean[df_clean["condition"]==4].index)
df_ERPdata = df_ERPdata.reset_index(drop=True)

df_omi = df_clean[df_clean['condition']==4]
df_omi = df_omi.reset_index(drop=True)

#save new dfs
#df_ERPdata.to_csv("data/typical/ERPdata/df_ERP_typ.csv")
#df_omi.to_csv("data/typical/omission/df_omi_typ.csv")

#df to numpy to do the PCA
ERPdata = df_ERPdata.drop(['condition', 'electrode','sub','age'], axis=1)
ERPdata = ERPdata.to_numpy()

##########
##visualization of conditions
##########

def df2data4plot(df,condition):
  data = df[df['condition']==condition]
  data = data.groupby('electrode', as_index=False).mean()
  data = data.drop(['sub','electrode'], axis=1)
  return data

familiarization = df2data4plot(df_ERPdata,3)
control = df2data4plot(df_ERPdata,1)
deviant = df2data4plot(df_ERPdata,2)
standard = df2data4plot(df_ERPdata,7)
postom = df2data4plot(df_ERPdata,5)

#Repetition suppression

plt.plot(np.mean(familiarization.iloc[[35,40,41,45,46,50,51,52]][:]), c=(0.44,0.68,0.28))
plt.plot(np.mean(control.iloc[[35,40,41,45,46,50,51,52]][:]), c=(0.44,0.19,0.63))
plt.xticks(range(0,999,99),('-100','0','100','200','300','400','500','600','700','800','900'), fontsize=12)
plt.xlim(0,999)
plt.ylim(-6,2)
plt.gca().invert_yaxis()
plt.title("Somatosensory repetition suppression")
plt.legend(['Familiarization', 'Control'])
plt.xlabel("Time (ms)")
plt.ylabel("Amplitude (µV)")
plt.savefig("figures/typical/ERPdata/ERP/ERP_RS_somato.png")
plt.show()

plt.plot(np.mean(familiarization.iloc[[4,5,6,11,12,105,111,124]][:]), c=(0.44,0.68,0.28))
plt.plot(np.mean(control.iloc[[4,5,6,11,12,105,111,124]][:]), c=(0.44,0.19,0.63))
plt.xticks(range(0,999,99),('-100','0','100','200','300','400','500','600','700','800','900'), fontsize=12)
plt.xlim(0,999)
plt.ylim(-3,2)
plt.gca().invert_yaxis()
plt.title("Frontal repetition suppression")
plt.legend(['Familiarization', 'Control'])
plt.xlabel("Time (ms)")
plt.ylabel("Amplitude (µV)")
plt.savefig("figures/typical/ERPdata/ERP/ERP_frontal.png")
plt.show()

#Deviance (SP)

plt.plot(np.mean(standard.iloc[[35,40,41,45,46,50,51,52]][:]), c='k')
plt.plot(np.mean(deviant.iloc[[35,40,41,45,46,50,51,52]][:]), c=(0.11,0.1,1))
plt.xticks(range(0,999,99),('-100','0','100','200','300','400','500','600','700','800','900'), fontsize=12)
plt.xlim(0,999)
plt.ylim(-6,2)
plt.gca().invert_yaxis()
plt.title("Somatosensory deviance")
plt.legend(['Standard', 'Deviant'])
plt.xlabel("Time (ms)")
plt.ylabel("Amplitude (µV)")
plt.savefig("figures/typical/ERPdata/ERP/ERP_dev_somato.png")
plt.show()

plt.plot(np.mean(standard.iloc[[4,5,6,11,12,105,111,124]][:]), c='k')
plt.plot(np.mean(deviant.iloc[[4,5,6,11,12,105,111,124]][:]), c=(0.11,0.1,1))
plt.xticks(range(0,999,99),('-100','0','100','200','300','400','500','600','700','800','900'), fontsize=12)
plt.xlim(0,999)
plt.ylim(-3,2)
plt.gca().invert_yaxis()
plt.title("Frontal deviance")
plt.legend(['Standard', 'Deviant'])
plt.xlabel("Time (ms)")
plt.ylabel("Amplitude (µV)")
plt.savefig("figures/typical/ERPdata/ERP/ERP_dev_frontal.png")
plt.show()

#PostOm (SP)

plt.plot(np.mean(standard.iloc[[35,40,41,45,46,50,51,52]][:]), c='k')
plt.plot(np.mean(postom.iloc[[35,40,41,45,46,50,51,52]][:]), c=(1,0.5,0.2))
plt.xticks(range(0,999,99),('-100','0','100','200','300','400','500','600','700','800','900'), fontsize=12)
plt.xlim(0,999)
plt.ylim(-6,2)
plt.gca().invert_yaxis()
plt.title("Somatosensory postomission")
plt.legend(['Standard', 'Postomission'])
plt.xlabel("Time (ms)")
plt.ylabel("Amplitude (µV)")
plt.savefig("figures/typical/ERPdata/ERP/ERP_pom_somato.png")
plt.show()

plt.plot(np.mean(standard.iloc[[4,5,6,11,12,105,111,124]][:]), c='k')
plt.plot(np.mean(postom.iloc[[4,5,6,11,12,105,111,124]][:]), c=(1,0.5,0))
plt.xticks(range(0,999,99),('-100','0','100','200','300','400','500','600','700','800','900'), fontsize=12)
plt.xlim(0,999)
plt.ylim(-3,2)
plt.gca().invert_yaxis()
plt.title("Frontal postomission")
plt.legend(['Standard', 'Postomission'])
plt.xlabel("Time (ms)")
plt.ylabel("Amplitude (µV)")
plt.savefig("figures/typical/ERPdata/ERP/ERP_pom_frontal.png")
plt.show()

#Omission (SP)

omission = df_omi.drop(['condition', 'sub','age'], axis=1)
omission =omission.groupby('electrode', as_index=False).mean()
omission = omission.drop(['electrode'], axis=1)


plt.plot(np.mean(omission.iloc[[35,40,41,45,46,50,51,52]][:]), c=(1,0.11,0.11))
plt.xticks(range(0,999,99),('-500','-400','-300','-200','-100','0','100','200','300','400','500'), fontsize=12)
plt.xlim(0,999)
plt.ylim(-6,2)
plt.gca().invert_yaxis()
plt.title("Somatosensory omission")
plt.xlabel("Time (ms)")
plt.ylabel("Amplitude (µV)")
plt.savefig("figures/typical/omission/ERP_omi_somato.png")
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

plt.close('all')
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
#df_components_variance.to_csv("data/typical/ERPdata/df_ERP_typ_components_variance.csv")


##############
## Fifth step : compute PCA1
##############

model = FactorAnalysis(n_components=n_components, rotation='varimax')
model.fit(ERPdata)
components = model.components_

df_components = pd.DataFrame(components) #nb_components * 1000 time series

#save
#df_components.to_csv("data/typical/ERPdata/df_ERP_components_typical.csv")

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
  plt.ylabel("Component loadings")
  plt.xlim(0,999)
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

factor_scores = model.fit_transform(ERPdata) #time_series * n_comp

df_factor_scores = pd.DataFrame(factor_scores)
#save
#df_factor_scores.to_csv("data/typical/ERPdata/df_ERP_factor_scores_typical.csv")

df_scores = pd.DataFrame(factor_scores)

myage = df_ERPdata['age'] 
mycond = df_ERPdata['condition']
myelec = df_ERPdata['electrode'] 

df_scores['electrode'] = myelec 

df_scores =df_scores.groupby('electrode', as_index=False).mean()
df_scores = df_scores.drop(['electrode'],axis=1)

coordinates = pd.read_excel('EEG_coordinates.xlsx')
coordinates = coordinates.drop(coordinates[coordinates['electrode'] == 'E125'].index)
coordinates = coordinates.drop(coordinates[coordinates['electrode'] == 'E126'].index)
coordinates = coordinates.drop(coordinates[coordinates['electrode'] == 'E127'].index)
coordinates = coordinates.drop(coordinates[coordinates['electrode'] == 'E128'].index)
coordinates = coordinates.drop(['electrode'],axis=1)
coordinates.index = coordinates.index+1

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
mycoor = [5,6,7,12,13,36,41,42,46,47,51,52,53,106,112,129]

def plot_scores(loadings, scores):
  fig,ax = plt.subplots(2,1,figsize=(4, 6),gridspec_kw={'height_ratios': [1, 2]})
  fig.suptitle('Component '+str(i+1)+ ', explained variance = '+str(components_variance[i])+'%')
  ax[0].plot(loadings)
  ax[0].set_ylim([-3, 9])
  ax[0].set_xlabel("Time series (ms)")
  ax[0].set_xlim(0,999)
  ax[0].set_xticks(ticks=range(0,999,99))
  ax[0].set_xticklabels(['-100','0','100','200','300','400','500','600','700','800','900'])
  ax[1].tricontourf(coordinates.x_coor,coordinates.y_coor, scores, cmap='seismic', levels=125, alpha=0.9, vmin=-1, vmax=1)
  ax[1].plot(coordinates.x_coor,coordinates.y_coor, 'k.', markersize=4)
  for coor in mycoor:
    ax[1].plot(coordinates.x_coor.loc[coor],coordinates.y_coor.loc[coor], 'k.', markersize=9)
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

df_scores_cond = pd.DataFrame(factor_scores)

df_scores_cond['electrode'] = myelec #add to df scores
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
df_scores_pom = scores_percond(df_scores_cond,5)
df_scores_MMNpom = df_scores_pom - df_scores_std
df_scores_fam = scores_percond(df_scores_cond,3)
df_scores_con = scores_percond(df_scores_cond,1)
df_scores_RS = df_scores_fam - df_scores_con

plt.close('all')
for i in range(len(components)):
  plot_scores(pos_loadings[i],df_scores_std[i])
  plt.savefig("figures/typical/ERPdata/percond/FA_typ_std_lat_topo_" + str(i+1) + ".png".format("PNG"))

plt.close('all')
for i in range(len(components)):
  plot_scores(pos_loadings[i],df_scores_dev[i])
  plt.savefig("figures/typical/ERPdata/percond/FA_typ_dev_lat_topo_" + str(i+1) + ".png".format("PNG"))

plt.close('all')
for i in range(len(components)):
  plot_scores(pos_loadings[i],df_scores_MMN[i])
  plt.savefig("figures/typical/ERPdata/percond/FA_typ_MMN_lat_topo_" + str(i+1) + ".png".format("PNG"))

plt.close('all')
for i in range(len(components)):
  plot_scores(pos_loadings[i],df_scores_pom[i])
  plt.savefig("figures/typical/ERPdata/percond/FA_typ_pom_lat_topo_" + str(i+1) + ".png".format("PNG"))

plt.close('all')
for i in range(len(components)):
  plot_scores(pos_loadings[i],df_scores_MMNpom[i])
  plt.savefig("figures/typical/ERPdata/percond/FA_typ_MMNpom_lat_topo_" + str(i+1) + ".png".format("PNG"))

plt.close('all')
for i in range(len(components)):
  plot_scores(pos_loadings[i],df_scores_fam[i])
  plt.savefig("figures/typical/ERPdata/percond/FA_typ_fam_lat_topo_" + str(i+1) + ".png".format("PNG"))

plt.close('all')
for i in range(len(components)):
  plot_scores(pos_loadings[i],df_scores_con[i])
  plt.savefig("figures/typical/ERPdata/percond/FA_typ_con_lat_topo_" + str(i+1) + ".png".format("PNG"))


plt.close('all')
for i in range(len(components)):
  plot_scores(pos_loadings[i],df_scores_RS[i])
  plt.savefig("figures/typical/ERPdata/percond/FA_typ_RS_lat_topo_" + str(i+1) + ".png".format("PNG"))

plt.close('all')

##plot components latencies and topographies per age

df_scores_age = pd.DataFrame(factor_scores)

#get topographies per age
df_scores_age['electrode'] = myelec #add to df scores
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

##plot components latencies and topographies per age per condition

df_scores_agecond = pd.DataFrame(factor_scores)

df_scores_agecond['electrode'] = myelec #add to df scores
df_scores_agecond['age'] = myage
df_scores_agecond['condition'] = mycond

def scores_peragecond(age,condition,factor_scores):
  myage = factor_scores[factor_scores['age']==age]
  mycond = myage[myage['condition']==condition]
  myage_cond = mycond.groupby('electrode', as_index=False).mean()
  myage_cond = myage_cond.drop(['electrode'],axis=1)
  return myage_cond

df_scores_two_fam = scores_peragecond(2,3,df_scores_agecond)
df_scores_two_con = scores_peragecond(2,1,df_scores_agecond)
df_scores_two_RS = df_scores_two_fam - df_scores_two_con
df_scores_two_dev = scores_peragecond(2,2,df_scores_agecond)
df_scores_two_std = scores_peragecond(2,7,df_scores_agecond)
df_scores_two_MMN = df_scores_two_dev - df_scores_two_std
df_scores_two_pom = scores_peragecond(2,5,df_scores_agecond)
df_scores_two_MMNpom = df_scores_two_pom - df_scores_two_std

for i in range(len(components)):
  plot_scores(pos_loadings[i],df_scores_two_fam[i])
  plt.savefig("figures/typical/ERPdata/perage/percond/comp_typ_two_fam_lat_topo_" + str(i+1) + ".png".format("PNG"))
  plt.close('all')
  plot_scores(pos_loadings[i],df_scores_two_con[i])
  plt.savefig("figures/typical/ERPdata/perage/percond/comp_typ_two_con_lat_topo_" + str(i+1) + ".png".format("PNG"))
  plt.close('all')
  plot_scores(pos_loadings[i],df_scores_two_RS[i])
  plt.savefig("figures/typical/ERPdata/perage/percond/comp_typ_two_RS_lat_topo_" + str(i+1) + ".png".format("PNG"))
  plt.close('all')
  plot_scores(pos_loadings[i],df_scores_two_dev[i])
  plt.savefig("figures/typical/ERPdata/perage/percond/comp_typ_two_dev_lat_topo_" + str(i+1) + ".png".format("PNG"))
  plt.close('all')
  plot_scores(pos_loadings[i],df_scores_two_std[i])
  plt.savefig("figures/typical/ERPdata/perage/percond/comp_typ_two_std_lat_topo_" + str(i+1) + ".png".format("PNG"))
  plt.close('all')
  plot_scores(pos_loadings[i],df_scores_two_MMN[i])
  plt.savefig("figures/typical/ERPdata/perage/percond/comp_typ_two_MMN_lat_topo_" + str(i+1) + ".png".format("PNG"))
  plt.close('all')
  plot_scores(pos_loadings[i],df_scores_two_pom[i])
  plt.savefig("figures/typical/ERPdata/perage/percond/comp_typ_two_pom_lat_topo_" + str(i+1) + ".png".format("PNG"))
  plt.close('all')
  plot_scores(pos_loadings[i],df_scores_two_MMNpom[i])
  plt.savefig("figures/typical/ERPdata/perage/percond/comp_typ_two_MMNpom_lat_topo_" + str(i+1) + ".png".format("PNG"))
  plt.close('all')

df_scores_four_fam = scores_peragecond(4,3,df_scores_agecond)
df_scores_four_con = scores_peragecond(4,1,df_scores_agecond)
df_scores_four_RS = df_scores_four_fam - df_scores_four_con
df_scores_four_dev = scores_peragecond(4,2,df_scores_agecond)
df_scores_four_std = scores_peragecond(4,7,df_scores_agecond)
df_scores_four_MMN = df_scores_four_dev - df_scores_four_std
df_scores_four_pom = scores_peragecond(4,5,df_scores_agecond)
df_scores_four_MMNpom = df_scores_four_pom - df_scores_four_std

for i in range(len(components)):
  plot_scores(pos_loadings[i],df_scores_four_fam[i])
  plt.savefig("figures/typical/ERPdata/perage/percond/comp_typ_four_fam_lat_topo_" + str(i+1) + ".png".format("PNG"))
  plt.close('all')
  plot_scores(pos_loadings[i],df_scores_four_con[i])
  plt.savefig("figures/typical/ERPdata/perage/percond/comp_typ_four_con_lat_topo_" + str(i+1) + ".png".format("PNG"))
  plt.close('all')
  plot_scores(pos_loadings[i],df_scores_four_RS[i])
  plt.savefig("figures/typical/ERPdata/perage/percond/comp_typ_four_RS_lat_topo_" + str(i+1) + ".png".format("PNG"))
  plt.close('all')
  plot_scores(pos_loadings[i],df_scores_four_dev[i])
  plt.savefig("figures/typical/ERPdata/perage/percond/comp_typ_four_dev_lat_topo_" + str(i+1) + ".png".format("PNG"))
  plt.close('all')
  plot_scores(pos_loadings[i],df_scores_four_std[i])
  plt.savefig("figures/typical/ERPdata/perage/percond/comp_typ_four_std_lat_topo_" + str(i+1) + ".png".format("PNG"))
  plt.close('all')
  plot_scores(pos_loadings[i],df_scores_four_MMN[i])
  plt.savefig("figures/typical/ERPdata/perage/percond/comp_typ_four_MMN_lat_topo_" + str(i+1) + ".png".format("PNG"))
  plt.close('all')
  plot_scores(pos_loadings[i],df_scores_four_pom[i])
  plt.savefig("figures/typical/ERPdata/perage/percond/comp_typ_four_pom_lat_topo_" + str(i+1) + ".png".format("PNG"))
  plt.close('all')
  plot_scores(pos_loadings[i],df_scores_four_MMNpom[i])
  plt.savefig("figures/typical/ERPdata/perage/percond/comp_typ_four_MMNpom_lat_topo_" + str(i+1) + ".png".format("PNG"))
  plt.close('all')


df_scores_six_fam = scores_peragecond(6,3,df_scores_agecond)
df_scores_six_con = scores_peragecond(6,1,df_scores_agecond)
df_scores_six_RS = df_scores_six_fam - df_scores_six_con
df_scores_six_dev = scores_peragecond(6,2,df_scores_agecond)
df_scores_six_std = scores_peragecond(6,7,df_scores_agecond)
df_scores_six_MMN = df_scores_six_dev - df_scores_six_std
df_scores_six_pom = scores_peragecond(6,5,df_scores_agecond)
df_scores_six_MMNpom = df_scores_six_pom - df_scores_six_std

for i in range(len(components)):
  plot_scores(pos_loadings[i],df_scores_six_fam[i])
  plt.savefig("figures/typical/ERPdata/perage/percond/comp_typ_six_fam_lat_topo_" + str(i+1) + ".png".format("PNG"))
  plt.close('all')
  plot_scores(pos_loadings[i],df_scores_six_con[i])
  plt.savefig("figures/typical/ERPdata/perage/percond/comp_typ_six_con_lat_topo_" + str(i+1) + ".png".format("PNG"))
  plt.close('all')
  plot_scores(pos_loadings[i],df_scores_six_RS[i])
  plt.savefig("figures/typical/ERPdata/perage/percond/comp_typ_six_RS_lat_topo_" + str(i+1) + ".png".format("PNG"))
  plt.close('all')
  plot_scores(pos_loadings[i],df_scores_six_dev[i])
  plt.savefig("figures/typical/ERPdata/perage/percond/comp_typ_six_dev_lat_topo_" + str(i+1) + ".png".format("PNG"))
  plt.close('all')
  plot_scores(pos_loadings[i],df_scores_six_std[i])
  plt.savefig("figures/typical/ERPdata/perage/percond/comp_typ_six_std_lat_topo_" + str(i+1) + ".png".format("PNG"))
  plt.close('all')
  plot_scores(pos_loadings[i],df_scores_six_MMN[i])
  plt.savefig("figures/typical/ERPdata/perage/percond/comp_typ_six_MMN_lat_topo_" + str(i+1) + ".png".format("PNG"))
  plt.close('all')
  plot_scores(pos_loadings[i],df_scores_six_pom[i])
  plt.savefig("figures/typical/ERPdata/perage/percond/comp_typ_six_pom_lat_topo_" + str(i+1) + ".png".format("PNG"))
  plt.close('all')
  plot_scores(pos_loadings[i],df_scores_six_MMNpom[i])
  plt.savefig("figures/typical/ERPdata/perage/percond/comp_typ_six_MMNpom_lat_topo_" + str(i+1) + ".png".format("PNG"))
  plt.close('all')

######################
##Tenth step : Extract data for statistics
######################

##All subjects analysis

#define two topographies first
somato = [36,41,42,46,47,51,52,53]
frontal = [5, 6, 7, 12, 13, 106, 112,129]
mycomponents = ([14,25]) #my components are 15 for N140 and 26 for P300

df_scores_cond = pd.DataFrame(factor_scores)

df_scores_cond['electrode'] = myelec #add to df scores
df_scores_cond['condition'] = mycond

def scores2stat(condition,electrode,df_scores,component):
  mydata = df_scores[df_scores['condition']==condition][df_scores['electrode'].isin(electrode)]
  mycomp = mydata[component]
  return mycomp

scores_con_somato = scores2stat(1,somato,df_scores_cond,14)
scores_dev_somato = scores2stat(2,somato,df_scores_cond,14)
scores_fam_somato = scores2stat(3,somato,df_scores_cond,14)
scores_std_somato = scores2stat(7,somato,df_scores_cond,14)
scores_pom_somato = scores2stat(5,somato,df_scores_cond,14)

scores_con_frontal = scores2stat(1,frontal,df_scores_cond,25)
scores_dev_frontal = scores2stat(2,frontal,df_scores_cond,25)
scores_fam_frontal = scores2stat(3,frontal,df_scores_cond,25)
scores_std_frontal = scores2stat(7,frontal,df_scores_cond,25)
scores_pom_frontal = scores2stat(5,frontal,df_scores_cond,25)

##Compare components per age

df_scores_age = pd.DataFrame(factor_scores)

df_scores_age['electrode'] = myelec #add to df scores
df_scores_age['age'] = myage

def comp_perage(df_scores,age,electrode,component):
  mydata = df_scores[df_scores['age']==age][df_scores['electrode'].isin(electrode)]
  mystat = mydata[component]
  return mystat

two_scores_comp14 = comp_perage(df_scores_age,2,somato,14)
four_scores_comp14 = comp_perage(df_scores_age,4,somato,14)
six_scores_comp14 = comp_perage(df_scores_age,6,somato,14)
two_scores_comp25 = comp_perage(df_scores_age,2,somato,25)
four_scores_comp25 = comp_perage(df_scores_age,4,somato,25)
six_scores_comp25 = comp_perage(df_scores_age,6,somato,25)


##Compare condition per age

df_scores_age_cond = pd.DataFrame(factor_scores)

df_scores_age_cond['electrode'] = myelec 
df_scores_age_cond['condition'] = mycond
df_scores_age_cond['age'] = myage

def comp_perage_percond(df_scores,age,electrode,condition,component):
  mydata = df_scores[df_scores['age']==age][df_scores['condition']==condition][df_scores['electrode'].isin(electrode)]
  mystat = mydata[component].reset_index(drop=True)
  return mystat


scores_two_fam_somato = comp_perage_percond(df_scores_age_cond,2,somato,3,14)
scores_two_con_somato = comp_perage_percond(df_scores_age_cond,2,somato,1,14)
scores_two_RS_somato = scores_two_fam_somato - scores_two_con_somato
scores_two_std_somato = comp_perage_percond(df_scores_age_cond,2,somato,7,14)
scores_two_dev_somato = comp_perage_percond(df_scores_age_cond,2,somato,2,14)
scores_two_MMN_somato = scores_two_dev_somato - scores_two_std_somato
scores_two_pom_somato = comp_perage_percond(df_scores_age_cond,2,somato,5,14)
scores_two_MMNpom_somato = scores_two_pom_somato - scores_two_std_somato

scores_two_fam_frtl = comp_perage_percond(df_scores_age_cond,2,frontal,3,25)
scores_two_con_frtl = comp_perage_percond(df_scores_age_cond,2,frontal,1,25)
scores_two_RS_frtl = scores_two_fam_frtl - scores_two_con_frtl
scores_two_std_frtl = comp_perage_percond(df_scores_age_cond,2,frontal,7,25)
scores_two_dev_frtl = comp_perage_percond(df_scores_age_cond,2,frontal,2,25)
scores_two_MMN_frtl = scores_two_dev_frtl - scores_two_std_frtl
scores_two_pom_frtl = comp_perage_percond(df_scores_age_cond,2,frontal,5,25)
scores_two_MMNpom_frtl = scores_two_pom_frtl - scores_two_std_frtl

scores_four_fam_somato = comp_perage_percond(df_scores_age_cond,4,somato,3,14)
scores_four_con_somato = comp_perage_percond(df_scores_age_cond,4,somato,1,14)
scores_four_RS_somato = scores_four_fam_somato - scores_four_con_somato
scores_four_std_somato = comp_perage_percond(df_scores_age_cond,4,somato,7,14)
scores_four_dev_somato = comp_perage_percond(df_scores_age_cond,4,somato,2,14)
scores_four_MMN_somato = scores_four_dev_somato - scores_four_std_somato
scores_four_pom_somato = comp_perage_percond(df_scores_age_cond,4,somato,5,14)
scores_four_MMNpom_somato = scores_four_pom_somato - scores_four_std_somato

scores_four_fam_frtl = comp_perage_percond(df_scores_age_cond,4,frontal,3,25)
scores_four_con_frtl = comp_perage_percond(df_scores_age_cond,4,frontal,1,25)
scores_four_RS_frtl = scores_four_fam_frtl - scores_four_con_frtl
scores_four_std_frtl = comp_perage_percond(df_scores_age_cond,4,frontal,7,25)
scores_four_dev_frtl = comp_perage_percond(df_scores_age_cond,4,frontal,2,25)
scores_four_MMN_frtl = scores_four_dev_frtl - scores_four_std_frtl
scores_four_pom_frtl = comp_perage_percond(df_scores_age_cond,4,frontal,5,25)
scores_four_MMNpom_frtl = scores_four_pom_frtl - scores_four_std_frtl


scores_six_fam_somato = comp_perage_percond(df_scores_age_cond,6,somato,3,14)
scores_six_con_somato = comp_perage_percond(df_scores_age_cond,6,somato,1,14)
scores_six_RS_somato = scores_six_fam_somato - scores_six_con_somato
scores_six_std_somato = comp_perage_percond(df_scores_age_cond,6,somato,7,14)
scores_six_dev_somato = comp_perage_percond(df_scores_age_cond,6,somato,2,14)
scores_six_MMN_somato = scores_six_dev_somato - scores_six_std_somato
scores_six_pom_somato = comp_perage_percond(df_scores_age_cond,6,somato,5,14)
scores_six_MMNpom_somato = scores_six_pom_somato - scores_six_std_somato

scores_six_fam_frtl = comp_perage_percond(df_scores_age_cond,6,frontal,3,25)
scores_six_con_frtl = comp_perage_percond(df_scores_age_cond,6,frontal,1,25)
scores_six_RS_frtl = scores_six_fam_frtl - scores_six_con_frtl
scores_six_std_frtl = comp_perage_percond(df_scores_age_cond,6,frontal,7,25)
scores_six_dev_frtl = comp_perage_percond(df_scores_age_cond,6,frontal,2,25)
scores_six_MMN_frtl = scores_six_dev_frtl - scores_six_std_frtl
scores_six_pom_frtl = comp_perage_percond(df_scores_age_cond,6,frontal,5,25)
scores_six_MMNpom_frtl = scores_six_pom_frtl - scores_six_std_frtl

#####################################################
##OMISSION ANALYSIS
#####################################################

data_omi = df_omi.drop(['condition', 'electrode','sub','age'], axis=1)
data_omi = data_omi.to_numpy()

##############
## Forth step : compute full PCA to get needed nb of components for 99% of variance
##############

kaiser_model_omi = FactorAnalysis()
kaiser_model_omi.fit(data_omi)

# get n components needed to reach 99% of explained variance
perc_var_explained_omi = 0
n_components_omi=0
components_variance_omi = list()
matrix = np.sum(kaiser_model_omi.components_**2, axis=1)

for i in range(len(matrix)):
  variance_comp_omi = (100*matrix[i])/np.sum(matrix)
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
plt.title("Explained variance per component for omission")
plt.savefig("figures/typical/omission/Typ_omi_screeplot.png")
plt.show()


df_components_variance_omi = pd.DataFrame(components_variance_omi)

#save
#df_components_variance_omi.to_csv("data/typical/omission/df_omi_typ_components_variance.csv")


##############
## Fifth step : compute PCA1
##############

model_omi = FactorAnalysis(n_components=n_components_omi, rotation='varimax')
model_omi.fit(data_omi)
components_omi = model_omi.components_

df_components_omi = pd.DataFrame(components_omi) #nb_components * 1000 time series

#save
#df_components_omi.to_csv("data/typical/omission/df_omi_components_typ.csv")

##############
##Sixth step : plot and save components loadings
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
  plt.xlim(0,999)
  plt.ylim([-3,9])
  plt.ylabel("Component loadings")
  plt.xticks(ticks=range(0,999,99), labels =['-500','-400','-300','-200','-100','0','100','200','300','400','500'])
  plt.savefig("figures/typical/omission/FA_loadings_omi_typ.png")

plot_ts_omi(pos_loadings_omi)
plt.show()

##Get max loadings to determine their latencies
loadings_max_omi = {}
for ind, comp in enumerate(pos_loadings_omi):
  load_max_omi = np.argmax(pos_loadings_omi[ind]) - 500
  loadings_max_omi[ind+1] = load_max_omi

sorted_max_omi = dict(sorted(loadings_max_omi.items(), key=lambda item: item[1]))
df_max_omi = pd.DataFrame(data=sorted_max_omi, index=[0]).T
df_max_omi.to_excel('data/typical/omission/df_omi_max_loadings.xlsx')

##############
## Seventh step : build factor scores and get components' scores topographies
##############

factor_scores_omi = model_omi.fit_transform(data_omi) # n_obs * n_comp

#save
#df_factor_scores_omi.to_csv("data/typical/omission/df_omi_factor_scores_typical.csv")

df_scores_omi = pd.DataFrame(factor_scores_omi)

myelec_omi = df_omi['electrode'] #get clean electrodes
df_scores_omi['electrode'] = myelec_omi #add to df scores

df_scores_omi =df_scores_omi.groupby('electrode', as_index=False).mean()
df_scores_omi = df_scores_omi.drop(['electrode'],axis=1)

coordinates = pd.read_excel('EEG_coordinates.xlsx')
coordinates = coordinates.drop(coordinates[coordinates['electrode'] == 'E125'].index)
coordinates = coordinates.drop(coordinates[coordinates['electrode'] == 'E126'].index)
coordinates = coordinates.drop(coordinates[coordinates['electrode'] == 'E127'].index)
coordinates = coordinates.drop(coordinates[coordinates['electrode'] == 'E128'].index)
coordinates = coordinates.drop(['electrode'],axis=1)
coordinates.index = coordinates.index+1

mycoor_omi = [36,41,42,46,47,51,52,53]
##plot and save score topographies and latencies to choose components
def plot_scores_omi(loadings, scores):
  fig,ax = plt.subplots(2,1,figsize=(4, 6),gridspec_kw={'height_ratios': [1, 2]})
  fig.suptitle('Component '+str(i+1)+ ', explained variance = '+str(components_variance_omi[i])+'%')
  ax[0].plot(loadings)
  ax[0].set_ylim([-3, 9])
  ax[0].set_xlabel("Time series (ms)")
  ax[0].set_xlim(0,999)
  ax[0].set_xticks(ticks=range(0,999,99))
  ax[0].set_xticklabels(['-500','-400','-300','-200','-100','0','100','200','300','400','500'])
  ax[1].tricontourf(coordinates.x_coor,coordinates.y_coor, scores, cmap='seismic', levels=125, alpha=0.9, vmin=-1, vmax=1)
  ax[1].plot(coordinates.x_coor,coordinates.y_coor, 'k.', markersize=4)
  for coor in mycoor_omi:
    ax[1].plot(coordinates.x_coor.loc[coor],coordinates.y_coor.loc[coor], 'k.', markersize=9)
  ax[1].set_axis_off()
  circle = plt.Circle((0, 0), coordinates.y_coor.max(), color='k', fill=False, linewidth=2)
  ax[1].add_patch(circle)
  ax[1].plot([-0.3,0,0.3,-0.3], [coordinates.y_coor.max(),coordinates.y_coor.max()*1.1,coordinates.y_coor.max(),coordinates.y_coor.max()], color='k')
  plt.tight_layout()

for i in range(len(components_omi)):
  plot_scores_omi(pos_loadings_omi[i],df_scores_omi[i])
  plt.savefig("figures/typical/FA_omi_typ_comp_lat_topo_" + str(i+1) + ".png".format("PNG"))

#my components are 15 for N140 and 11 for P300

###############
##Plot scores per age
###############

#add data information to get scores for plots
df_scores_omi_age = pd.DataFrame(factor_scores_omi)

myelec_omi = df_omi['electrode'] #get clean electrodes
df_scores_omi_age['electrode'] = myelec_omi #add to df scores
myage_omi = df_omi['age'] #get clean age
df_scores_omi_age['age'] = myage_omi

#get topographies per age

def scores_perage(df, age):
  df_age = df[df['age']==age]
  df_age = df_age.drop(['age'],axis=1)
  df_age = df_age.groupby(['electrode'], as_index=False).mean()
  df_age = df_age.drop(['electrode'],axis=1)
  return df_age

df_scores_omi_two = scores_perage(df_scores_omi_age,2)
df_scores_omi_four = scores_perage(df_scores_omi_age,4)
df_scores_omi_six = scores_perage(df_scores_omi_age,6)


for i in range(len(components_omi)):
  plot_scores_omi(pos_loadings_omi[i],df_scores_omi_two[i])
  plt.savefig("figures/typical/omission/perage/comp_typ_omi_two_lat_topo_" + str(i+1) + ".png".format("PNG"))

for i in range(len(components_omi)):
  plot_scores_omi(pos_loadings_omi[i],df_scores_omi_four[i])
  plt.savefig("figures/typical/omission/perage/comp_typ_omi_four_lat_topo_" + str(i+1) + ".png".format("PNG"))

for i in range(len(components_omi)):
  plot_scores_omi(pos_loadings_omi[i],df_scores_omi_six[i])
  plt.savefig("figures/typical/omission/perage/comp_typ_omi_six_lat_topo_" + str(i+1) + ".png".format("PNG"))


######################
##Tenth step : Extract data for statistics
######################

##All subjects analysis

#define two topographies first
somato = [36,41,42,46,47,51,52,53]
frontal = [5, 6, 7, 12, 13, 106, 112,129]
mycomponents = ([1]) #my component is 2, somatosensory negativity

##Compare components per age

df_scores_omi_age = pd.DataFrame(factor_scores_omi)

df_scores_omi_age['electrode'] = myelec_omi #add to df scores
df_scores_omi_age['age'] = myage_omi

def comp_perage(df_scores,age,electrode,component):
  mydata = df_scores[df_scores['age']==age][df_scores['electrode'].isin(electrode)]
  mystat = mydata[component]
  return mystat

two_scores_omi = comp_perage(df_scores_omi_age,2,somato,1)
four_scores_omi = comp_perage(df_scores_omi_age,4,somato,1)
six_scores_omi = comp_perage(df_scores_omi_age,6,somato,1)


######################
##Tenth step : Statistics
######################

tests = [(scores_fam_somato, scores_con_somato),(scores_fam_frontal, scores_con_frontal), (scores_dev_somato, scores_std_somato),(scores_dev_frontal, scores_std_frontal),(scores_pom_somato, scores_std_somato),(scores_pom_frontal, scores_std_frontal),(two_scores_comp14,four_scores_comp14),(two_scores_comp14,six_scores_comp14),(four_scores_comp14,six_scores_comp14),(two_scores_comp25,four_scores_comp25),(two_scores_comp25,six_scores_comp25),(four_scores_comp25,six_scores_comp25),(scores_two_fam_somato, scores_two_con_somato), (scores_two_fam_frtl, scores_two_con_frtl), (scores_two_dev_somato, scores_two_std_somato), (scores_two_dev_frtl, scores_two_std_frtl),(scores_two_pom_somato, scores_two_std_somato), (scores_two_pom_frtl, scores_two_std_frtl), (scores_four_fam_somato, scores_four_con_somato), (scores_four_fam_frtl, scores_four_con_frtl), (scores_four_dev_somato, scores_four_std_somato), (scores_four_dev_frtl, scores_four_std_frtl),(scores_four_pom_somato, scores_four_std_somato), (scores_four_pom_frtl, scores_four_std_frtl), (scores_six_fam_somato, scores_six_con_somato), (scores_six_fam_frtl, scores_six_con_frtl), (scores_six_dev_somato, scores_six_std_somato), (scores_six_dev_frtl, scores_six_std_frtl),(scores_six_pom_somato, scores_six_std_somato), (scores_six_pom_frtl, scores_six_std_frtl), (scores_two_RS_somato, scores_four_RS_somato), (scores_two_RS_somato, scores_six_RS_somato), (scores_four_RS_somato, scores_six_RS_somato), (scores_two_RS_frtl, scores_four_RS_frtl), (scores_two_RS_frtl, scores_six_RS_frtl), (scores_four_RS_frtl, scores_six_RS_frtl), (scores_two_MMN_somato, scores_four_MMN_somato), (scores_two_MMN_somato, scores_six_MMN_somato), (scores_four_MMN_somato, scores_six_MMN_somato), (scores_two_MMN_frtl, scores_four_MMN_frtl), (scores_two_MMN_frtl, scores_six_MMN_frtl), (scores_four_MMN_frtl, scores_six_MMN_frtl), (scores_two_MMNpom_somato, scores_four_MMNpom_somato), (scores_two_MMNpom_somato, scores_six_MMNpom_somato), (scores_four_MMNpom_somato, scores_six_MMNpom_somato), (scores_two_MMNpom_frtl, scores_four_MMNpom_frtl), (scores_two_MMNpom_frtl, scores_six_MMNpom_frtl), (scores_four_MMNpom_frtl, scores_six_MMNpom_frtl), (two_scores_omi, four_scores_omi),(two_scores_omi,six_scores_omi),(four_scores_omi, six_scores_omi)]
n_tests = len(tests)

df_p_value = pd.DataFrame(columns=['p_values', 'statistics'], index=['RS_somato','RS_frtl','MMN_somato','MMN_frtl','MMNpom_somato','MMNpom_frtl','towfour_N140','twosix_N140','foursix_N140','twofour_P300','twosix_P300', 'foursix_P300', 'two_RS_somato', 'two_RS_frtl', 'two_MMN_somato', 'two_MMN_frtl','two_MMNpom_somato', 'two_MMNpom_frtl', 'four_RS_somato', 'four_RS_frtl', 'four_MMN_somato', 'four_MMN_frtl','four_MMNpom_somato', 'four_MMNpom_frtl', 'six_RS_somato', 'six_RS_frtl', 'six_MMN_somato', 'six_MMN_frtl','six_MMNpom_somato', 'six_MMNpom_frtl', 'twofour_RS_somato', 'twosix_RS_somato', 'foursix_RS_somato', 'twofour_RS_frtl', 'twosix_RS_frtl', 'foursix_RS_frtl', 'twofour_MMN_somato', 'twosix_MMN_somato', 'foursix_MMN_somato', 'twofour_MMN_frtl', 'twosix_MMN_frtl', 'foursix_MMN_frtl','twofour_MMNpom_somato', 'twosix_MMNpom_somato', 'foursix_MMNpom_somato', 'twofour_MMNpom_frtl', 'twosix_MMNpom_frtl', 'foursix_MMNpom_frtl', 'twofour_omi','twosix_omi','foursix_omi'])
p_values = []
for ind, samples in enumerate(tests):
  sample1 = samples[0]
  sample2 = samples[1]
  stat, p = stats.wilcoxon(sample1, sample2)
  df_p_value.iloc[ind] = p, stat
  p_values.append(p)

p_values = np.array(p_values)

significant, corrected_p_values, _, _ = multipletests(p_values, method='fdr_bh')

df_p_value['significant'] = significant
df_p_value['corrected'] = corrected_p_values


#######
##Plot final figures
######

def plot_ts_finale(loadings):
  plt.close('all')
  for ind, comp in enumerate(loadings):
    plt.plot(range(0, 1000), comp, linewidth=1,color='k')
  plt.plot(range(0, 1000), loadings[14], linewidth=3,color='r')
  plt.plot(range(0, 1000), loadings[25], linewidth=3,color='b')
  plt.xlabel("Time series (ms)")
  plt.ylabel("Component loadings")
  plt.xlim(0,999)
  plt.ylim([-3,9])
  plt.xticks(ticks=range(0,999,99), labels =['-100','0','100','200','300','400','500','600','700','800','900'])
  plt.savefig("figures/typical/FA_loadings_ERP_typ_finale.png")

plot_ts_finale(pos_loadings)
plt.show()

def plot_ts_omi_finale(loadings):
  plt.close('all')
  for ind, comp in enumerate(loadings):
     plt.plot(range(0, 1000), comp, linewidth=1,color='k')
  plt.plot(range(0, 1000), loadings[1], linewidth=3,color='g')
  plt.xlabel("Time series (ms)")
  plt.ylabel("Component loadings")
  plt.xlim(0,999)
  plt.ylim([-3,9])
  plt.xticks(ticks=range(0,999,99), labels =['-500','-400','-300','-200','-100','0','100','200','300','400','500'])
  plt.savefig("figures/typical/FA_loadings_omi_typ_finale.png")

plot_ts_omi_finale(pos_loadings_omi)
plt.show()

mycoorsomato = [36,41,42,46,47,51,52,53]

##score topographies for my components
fig,ax = plt.subplots(figsize=(4, 4))
circle = plt.Circle((0, 0.05), coordinates.y_coor.max()-0.05, color='k', fill=False, linewidth=2)
nose = plt.Polygon([(-0.3,coordinates.y_coor.max()), (0,coordinates.y_coor.max()*1.1), (0.3,coordinates.y_coor.max())], color='k', fill=False, linewidth=2)
plt.gca().add_patch(circle)
plt.gca().add_patch(nose)
plt.tricontourf(coordinates.x_coor,coordinates.y_coor, df_scores[14], cmap='seismic',  levels=125, vmin=-1, vmax=1)
plt.plot(coordinates.x_coor,coordinates.y_coor, 'k.', markersize=4)
for coor in mycoorsomato:
  plt.plot(coordinates.x_coor.loc[coor],coordinates.y_coor.loc[coor], 'k.', markersize=9)
plt.gca().set_frame_on(False)
plt.gca().set_xticks([])
plt.gca().set_yticks([])
plt.savefig("figures/typical/ERPdata/FA_ERP_typ_N140_finale.png")
plt.show() 

mycoorfrtl = [5,6,7,12,13,106,112,129]

##score topographies for my components
fig,ax = plt.subplots(figsize=(4, 4))
circle = plt.Circle((0, 0.05), coordinates.y_coor.max()-0.05, color='k', fill=False, linewidth=2)
nose = plt.Polygon([(-0.3,coordinates.y_coor.max()), (0,coordinates.y_coor.max()*1.1), (0.3,coordinates.y_coor.max())], color='k', fill=False, linewidth=2)
plt.gca().add_patch(circle)
plt.gca().add_patch(nose)
plt.tricontourf(coordinates.x_coor,coordinates.y_coor, df_scores[25], cmap='seismic',  levels=125, vmin=-1, vmax=1)
plt.plot(coordinates.x_coor,coordinates.y_coor, 'k.', markersize=4)
for coor in mycoorfrtl:
  plt.plot(coordinates.x_coor.loc[coor],coordinates.y_coor.loc[coor], 'k.', markersize=9)
plt.gca().set_frame_on(False)
plt.gca().set_xticks([])
plt.gca().set_yticks([])
plt.savefig("figures/typical/ERPdata/FA_ERP_typ_P300_finale.png")
plt.show() 

mycoorsomato = [36,41,42,46,47,51,52,53]

##score topographies for my components
fig,ax = plt.subplots(figsize=(4, 4))
circle = plt.Circle((0, 0.05), coordinates.y_coor.max()-0.05, color='k', fill=False, linewidth=2)
nose = plt.Polygon([(-0.3,coordinates.y_coor.max()), (0,coordinates.y_coor.max()*1.1), (0.3,coordinates.y_coor.max())], color='k', fill=False, linewidth=2)
plt.gca().add_patch(circle)
plt.gca().add_patch(nose)
plt.tricontourf(coordinates.x_coor,coordinates.y_coor, df_scores_omi[1], cmap='seismic',  levels=125, vmin=-1, vmax=1)
plt.plot(coordinates.x_coor,coordinates.y_coor, 'k.', markersize=4)
for coor in mycoorsomato:
  plt.plot(coordinates.x_coor.loc[coor],coordinates.y_coor.loc[coor], 'k.', markersize=9)
plt.gca().set_frame_on(False)
plt.gca().set_xticks([])
plt.gca().set_yticks([])
plt.savefig("figures/typical/omission/FA_omi_typ_finale.png")
plt.show() 


def plot_loadings_finale(loadings):
  fig,ax = plt.subplots(figsize=(7, 4))
  plt.plot(loadings)
  plt.ylim([-2, 3.5])
  plt.xlabel("Time series (ms)")
  plt.xlim(0,999)
  plt.xticks(ticks=range(0,999,99))
  plt.xticks(ticks=range(0,999,99), labels =['-100','0','100','200','300','400','500','600','700','800','900'])

plot_loadings_finale(pos_loadings[14])
plt.savefig("figures/typical/ERPdata/ERP_typ_loading_N140_finale.png")
plot_loadings_finale(pos_loadings[25])
plt.savefig("figures/typical/ERPdata/ERP_typ_loading_P300_finale.png")

def plot_loadings_omi_finale(loadings):
  fig,ax = plt.subplots(figsize=(7, 4))
  plt.plot(loadings)
  plt.ylim([-2, 3.5])
  plt.xlabel("Time series (ms)")
  plt.xlim(0,999)
  plt.xticks(ticks=range(0,999,99))
  plt.xticks(ticks=range(0,999,99), labels =['-500','-400','-300','-100','-100','0','100','200','300','400','500'])

plot_loadings_omi_finale(pos_loadings_omi[1])
plt.savefig("figures/typical/omission/omi_typ_loading_N2_finale.png")
plt.show()



mycoorsomato = [36,41,42,46,47,51,52,53]
mycoorfrtl = [5,6,7,12,13,106,112,129]
##plot and save score topographies and latencies to choose components
def plot_scores_finale(scores, coor):
  fig,ax = plt.subplots(figsize=(4, 4))
  plt.tricontourf(coordinates.x_coor,coordinates.y_coor, scores, cmap='seismic', levels=125, alpha=0.9, vmin=-1, vmax=1)
  plt.plot(coordinates.x_coor,coordinates.y_coor, 'k.', markersize=4)
  for coor in coor:
    plt.plot(coordinates.x_coor.loc[coor],coordinates.y_coor.loc[coor], 'k.', markersize=9)
  plt.gca().set_frame_on(False)
  circle = plt.Circle((0, 0), coordinates.y_coor.max(), color='k', fill=False, linewidth=2)
  nose = plt.Polygon([(-0.3,coordinates.y_coor.max()), (0,coordinates.y_coor.max()*1.1), (0.3,coordinates.y_coor.max())], color='k', fill=False, linewidth=2)
  plt.gca().add_patch(circle)
  plt.gca().add_patch(nose)
  plt.plot([-0.3,0,0.3,-0.3], [coordinates.y_coor.max(),coordinates.y_coor.max()*1.1,coordinates.y_coor.max(),coordinates.y_coor.max()], color='k')
  plt.gca().set_xticks([])
  plt.gca().set_yticks([])
  plt.tight_layout()

plot_scores_finale(df_scores_two[14], mycoorsomato)
plt.savefig("figures/typical/ERPdata/ERP_typ_two_N140_finale.png")
plot_scores_finale(df_scores_two[25], mycoorfrtl)
plt.savefig("figures/typical/ERPdata/ERP_typ_two_P300_finale.png")
plot_scores_finale(df_scores_four[14], mycoorsomato)
plt.savefig("figures/typical/ERPdata/ERP_typ_four_N140_finale.png")
plot_scores_finale(df_scores_four[25], mycoorfrtl)
plt.savefig("figures/typical/ERPdata/ERP_typ_four_P300_finale.png")
plot_scores_finale(df_scores_six[14], mycoorsomato)
plt.savefig("figures/typical/ERPdata/ERP_typ_six_N140_finale.png")
plot_scores_finale(df_scores_six[25], mycoorfrtl)
plt.savefig("figures/typical/ERPdata/ERP_typ_six_P300_finale.png")
plot_scores_finale(df_scores_omi_two[1], mycoorsomato)
plt.savefig("figures/typical/omission/omi_typ_two_N200_finale.png")
plot_scores_finale(df_scores_omi_four[1], mycoorsomato)
plt.savefig("figures/typical/omission/omi_typ_four_N200_finale.png")
plot_scores_finale(df_scores_omi_six[1], mycoorsomato)
plt.savefig("figures/typical/omission/omi_typ_six_N200_finale.png")
plot_scores_finale(df_scores_std[14], mycoorsomato)
plt.savefig("figures/typical/ERPdata/ERP_typ_std_N140_finale.png")
plot_scores_finale(df_scores_std[25], mycoorfrtl)
plt.savefig("figures/typical/ERPdata/ERP_typ_std_P300_finale.png")
plot_scores_finale(df_scores_dev[14], mycoorsomato)
plt.savefig("figures/typical/ERPdata/ERP_typ_dev_N140_finale.png")
plot_scores_finale(df_scores_dev[25], mycoorfrtl)
plt.savefig("figures/typical/ERPdata/ERP_typ_dev_P300_finale.png")
plot_scores_finale(df_scores_pom[14], mycoorsomato)
plt.savefig("figures/typical/ERPdata/ERP_typ_pom_N140_finale.png")
plot_scores_finale(df_scores_pom[25], mycoorfrtl)
plt.savefig("figures/typical/ERPdata/ERP_typ_pom_P300_finale.png")
plot_scores_finale(df_scores_fam[14], mycoorsomato)
plt.savefig("figures/typical/ERPdata/ERP_typ_fam_N140_finale.png")
plot_scores_finale(df_scores_fam[25], mycoorfrtl)
plt.savefig("figures/typical/ERPdata/ERP_typ_fam_P300_finale.png")
plot_scores_finale(df_scores_con[14], mycoorsomato)
plt.savefig("figures/typical/ERPdata/ERP_typ_con_N140_finale.png")
plot_scores_finale(df_scores_con[25], mycoorfrtl)
plt.savefig("figures/typical/ERPdata/ERP_typ_con_P300_finale.png")
plot_scores_finale(df_scores_RS[14], mycoorsomato)
plt.savefig("figures/typical/ERPdata/ERP_typ_RS_N140_finale.png")
plot_scores_finale(df_scores_RS[25], mycoorfrtl)
plt.savefig("figures/typical/ERPdata/ERP_typ_RS_P300_finale.png")
plot_scores_finale(df_scores_MMN[14], mycoorsomato)
plt.savefig("figures/typical/ERPdata/ERP_typ_MMN_N140_finale.png")
plot_scores_finale(df_scores_MMN[25], mycoorfrtl)
plt.savefig("figures/typical/ERPdata/ERP_typ_MMN_P300_finale.png")
plot_scores_finale(df_scores_MMNpom[14], mycoorsomato)
plt.savefig("figures/typical/ERPdata/ERP_typ_MMNpom_N140_finale.png")
plot_scores_finale(df_scores_MMNpom[25], mycoorfrtl)
plt.savefig("figures/typical/ERPdata/ERP_typ_MMNpom_P300_finale.png")
plt.show()


plot_scores_finale(df_scores_two_std[14], mycoorsomato)
plt.savefig("figures/typical/ERPdata/ERP_typ_two_std_N140_finale.png")
plot_scores_finale(df_scores_two_std[25], mycoorfrtl)
plt.savefig("figures/typical/ERPdata/ERP_typ_two_std_P300_finale.png")
plot_scores_finale(df_scores_two_dev[14], mycoorsomato)
plt.savefig("figures/typical/ERPdata/ERP_typ_two_dev_N140_finale.png")
plot_scores_finale(df_scores_two_dev[25], mycoorfrtl)
plt.savefig("figures/typical/ERPdata/ERP_typ_two_dev_P300_finale.png")
plot_scores_finale(df_scores_two_pom[14], mycoorsomato)
plt.savefig("figures/typical/ERPdata/ERP_typ_two_pom_N140_finale.png")
plot_scores_finale(df_scores_two_pom[25], mycoorfrtl)
plt.savefig("figures/typical/ERPdata/ERP_typ_two_pom_P300_finale.png")
plot_scores_finale(df_scores_two_fam[14], mycoorsomato)
plt.savefig("figures/typical/ERPdata/ERP_typ_two_fam_N140_finale.png")
plot_scores_finale(df_scores_two_fam[25], mycoorfrtl)
plt.savefig("figures/typical/ERPdata/ERP_typ_two_fam_P300_finale.png")
plot_scores_finale(df_scores_two_con[14], mycoorsomato)
plt.savefig("figures/typical/ERPdata/ERP_typ_two_con_N140_finale.png")
plot_scores_finale(df_scores_two_con[25], mycoorfrtl)
plt.savefig("figures/typical/ERPdata/ERP_typ_two_con_P300_finale.png")
plot_scores_finale(df_scores_two_RS[14], mycoorsomato)
plt.savefig("figures/typical/ERPdata/ERP_typ_two_RS_N140_finale.png")
plot_scores_finale(df_scores_two_RS[25], mycoorfrtl)
plt.savefig("figures/typical/ERPdata/ERP_typ_two_RS_P300_finale.png")
plot_scores_finale(df_scores_two_MMN[14], mycoorsomato)
plt.savefig("figures/typical/ERPdata/ERP_typ_two_MMN_N140_finale.png")
plot_scores_finale(df_scores_two_MMN[25], mycoorfrtl)
plt.savefig("figures/typical/ERPdata/ERP_typ_two_MMN_P300_finale.png")
plot_scores_finale(df_scores_two_MMNpom[14], mycoorsomato)
plt.savefig("figures/typical/ERPdata/ERP_typ_two_MMNpom_N140_finale.png")
plot_scores_finale(df_scores_two_MMNpom[25], mycoorfrtl)
plt.savefig("figures/typical/ERPdata/ERP_typ_two_MMNpom_P300_finale.png")
plt.show()

plot_scores_finale(df_scores_four_std[14], mycoorsomato)
plt.savefig("figures/typical/ERPdata/ERP_typ_four_std_N140_finale.png")
plot_scores_finale(df_scores_four_std[25], mycoorfrtl)
plt.savefig("figures/typical/ERPdata/ERP_typ_four_std_P300_finale.png")
plot_scores_finale(df_scores_four_dev[14], mycoorsomato)
plt.savefig("figures/typical/ERPdata/ERP_typ_four_dev_N140_finale.png")
plot_scores_finale(df_scores_four_dev[25], mycoorfrtl)
plt.savefig("figures/typical/ERPdata/ERP_typ_four_dev_P300_finale.png")
plot_scores_finale(df_scores_four_pom[14], mycoorsomato)
plt.savefig("figures/typical/ERPdata/ERP_typ_four_pom_N140_finale.png")
plot_scores_finale(df_scores_four_pom[25], mycoorfrtl)
plt.savefig("figures/typical/ERPdata/ERP_typ_four_pom_P300_finale.png")
plot_scores_finale(df_scores_four_fam[14], mycoorsomato)
plt.savefig("figures/typical/ERPdata/ERP_typ_four_fam_N140_finale.png")
plot_scores_finale(df_scores_four_fam[25], mycoorfrtl)
plt.savefig("figures/typical/ERPdata/ERP_typ_four_fam_P300_finale.png")
plot_scores_finale(df_scores_four_con[14], mycoorsomato)
plt.savefig("figures/typical/ERPdata/ERP_typ_four_con_N140_finale.png")
plot_scores_finale(df_scores_four_con[25], mycoorfrtl)
plt.savefig("figures/typical/ERPdata/ERP_typ_four_con_P300_finale.png")
plot_scores_finale(df_scores_four_RS[14], mycoorsomato)
plt.savefig("figures/typical/ERPdata/ERP_typ_four_RS_N140_finale.png")
plot_scores_finale(df_scores_four_RS[25], mycoorfrtl)
plt.savefig("figures/typical/ERPdata/ERP_typ_four_RS_P300_finale.png")
plot_scores_finale(df_scores_four_MMN[14], mycoorsomato)
plt.savefig("figures/typical/ERPdata/ERP_typ_four_MMN_N140_finale.png")
plot_scores_finale(df_scores_four_MMN[25], mycoorfrtl)
plt.savefig("figures/typical/ERPdata/ERP_typ_four_MMN_P300_finale.png")
plot_scores_finale(df_scores_four_MMNpom[14], mycoorsomato)
plt.savefig("figures/typical/ERPdata/ERP_typ_four_MMNpom_N140_finale.png")
plot_scores_finale(df_scores_four_MMNpom[25], mycoorfrtl)
plt.savefig("figures/typical/ERPdata/ERP_typ_four_MMNpom_P300_finale.png")
plt.show()

plot_scores_finale(df_scores_six_std[14], mycoorsomato)
plt.savefig("figures/typical/ERPdata/ERP_typ_six_std_N140_finale.png")
plot_scores_finale(df_scores_six_std[25], mycoorfrtl)
plt.savefig("figures/typical/ERPdata/ERP_typ_six_std_P300_finale.png")
plot_scores_finale(df_scores_six_dev[14], mycoorsomato)
plt.savefig("figures/typical/ERPdata/ERP_typ_six_dev_N140_finale.png")
plot_scores_finale(df_scores_six_dev[25], mycoorfrtl)
plt.savefig("figures/typical/ERPdata/ERP_typ_six_dev_P300_finale.png")
plot_scores_finale(df_scores_six_pom[14], mycoorsomato)
plt.savefig("figures/typical/ERPdata/ERP_typ_six_pom_N140_finale.png")
plot_scores_finale(df_scores_six_pom[25], mycoorfrtl)
plt.savefig("figures/typical/ERPdata/ERP_typ_six_pom_P300_finale.png")
plot_scores_finale(df_scores_six_fam[14], mycoorsomato)
plt.savefig("figures/typical/ERPdata/ERP_typ_six_fam_N140_finale.png")
plot_scores_finale(df_scores_six_fam[25], mycoorfrtl)
plt.savefig("figures/typical/ERPdata/ERP_typ_six_fam_P300_finale.png")
plot_scores_finale(df_scores_six_con[14], mycoorsomato)
plt.savefig("figures/typical/ERPdata/ERP_typ_six_con_N140_finale.png")
plot_scores_finale(df_scores_six_con[25], mycoorfrtl)
plt.savefig("figures/typical/ERPdata/ERP_typ_six_con_P300_finale.png")
plot_scores_finale(df_scores_six_RS[14], mycoorsomato)
plt.savefig("figures/typical/ERPdata/ERP_typ_six_RS_N140_finale.png")
plot_scores_finale(df_scores_six_RS[25], mycoorfrtl)
plt.savefig("figures/typical/ERPdata/ERP_typ_six_RS_P300_finale.png")
plot_scores_finale(df_scores_six_MMN[14], mycoorsomato)
plt.savefig("figures/typical/ERPdata/ERP_typ_six_MMN_N140_finale.png")
plot_scores_finale(df_scores_six_MMN[25], mycoorfrtl)
plt.savefig("figures/typical/ERPdata/ERP_typ_six_MMN_P300_finale.png")
plot_scores_finale(df_scores_six_MMNpom[14], mycoorsomato)
plt.savefig("figures/typical/ERPdata/ERP_typ_six_MMNpom_N140_finale.png")
plot_scores_finale(df_scores_six_MMNpom[25], mycoorfrtl)
plt.savefig("figures/typical/ERPdata/ERP_typ_six_MMNpom_P300_finale.png")
plt.show()
