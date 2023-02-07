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
data = loadmat('data/Typical_PCA2.mat')['Typical_PCA'] # 26187*1000

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
#save#df_badchan.to_excel("df_badchan_typical.xlsx") #for manual checking
#get the last column
df_mybadchan = df_badchan[999]
#find index of non zero (bad channel) rwos
df_mybadchan = df_mybadchan.index[df_mybadchan>0]

##############
## Second step : construct pandas dataframe with label of each row 
##############
myages = list()
two_yo = [2]*12
myages.extend(two_yo)
four_yo = [4]*13
myages.extend(four_yo)
six_yo = [6]*3
myages.extend(six_yo)



age = []
sub = []
for i in range(28): #28 subjects
    new_sub = [i+1] * 129* 7 #129 channels and 7 conditions
    new_age = [myages[i]] * 129 * 7
    sub += new_sub
    age += new_age

condition = []
for j in range(1, 29): #28 subjects
  for i in range(1,8): #7conditions
    new_condition = [i] * 129 #channels
    condition += new_condition

electrode = list(range(1, 130)) * 28 *7


df_data['condition'] = condition
df_data['electrode'] = electrode
df_data['sub'] = sub
df_data['age'] = age

##############
##Third step Discard eye channels (125:128) and bad channels  
##############
#check difference between length must be equal to number of dropped index

#delete bad channels
df_nobadchannel = df_data.drop(index=df_mybadchan)
#save#df_nobadchannel.to_excel("df_nobadchannel_typical.xlsx")
#delete eye channels
df_clean = df_nobadchannel.drop(df_nobadchannel[df_nobadchannel['electrode'] ==125].index)
df_clean = df_clean.drop(df_clean[df_clean['electrode'] ==126].index)
df_clean = df_clean.drop(df_clean[df_clean['electrode'] ==127].index)
df_clean = df_clean.drop(df_clean[df_clean['electrode'] ==128].index)

#save#df_clean.to_excel("df_clean_typical.xlsx")

##############
## Fourth step : separate data for PCA1 (data without omission) and PCA2 (omission only) 
##############

df_omission = df_clean[df_clean["condition"]==4]
df_omission = df_omission.reset_index(drop=True)
data_omission = df_omission.drop(['condition', 'electrode','sub','age'], axis=1)
data_omission = data_omission.to_numpy()

df_ERPdata = df_clean.drop(df_clean[df_clean["condition"]==4].index)
df_ERPdata = df_ERPdata.reset_index(drop=True)

#save#df_ERPdata.to_excel("df_ERPdata_typical.xlsx")
#to replace moy by std data['condition'] = data['condition'].replace(7, 6)
data_ERP = df_ERPdata.drop(['condition', 'electrode','sub','age'], axis=1)
data_ERP = data_ERP.to_numpy()


####
##visualization of conditions
###

familiarization = df_ERPdata[df_ERPdata['condition']==3]
familiarization =familiarization.groupby('electrode', as_index=False).mean()
familiarization = familiarization.drop(['condition', 'sub','electrode','age'], axis=1)

deviant = df_ERPdata[df_ERPdata['condition']==2]
deviant =deviant.groupby('electrode', as_index=False).mean()
deviant = deviant.drop(['condition', 'sub','electrode','age'], axis=1)

postom = df_ERPdata[df_ERPdata['condition']==5]
postom =postom.groupby('electrode', as_index=False).mean()
postom = postom.drop(['condition', 'sub','electrode','age'], axis=1)

standard = df_ERPdata[df_ERPdata['condition']==6]
standard =standard.groupby('electrode', as_index=False).mean()
standard = standard.drop(['condition', 'sub','electrode','age'], axis=1)

plt.plot(np.mean(familiarization.iloc[[51]][:]))
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
plt.savefig("figures/scree_plot_typiques.png")
plt.show()


df_components_variance = pd.DataFrame(components_variance)
#save#df_components_variance.to_excel("components_variance_typical.xlsx")
##############
## Sixth step : compute PCA with number of components
##############

model = FactorAnalysis(n_components=n_components, rotation='varimax')
model.fit(data_ERP)
components = model.components_

df_components = pd.DataFrame(components)
#save#df_components.to_excel("components_typical.xlsx")
##############
##Seventh step : plot components
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
  plt.ylim([round(components.min().min())-0.5,round(components.max().max())+0.5])
  plt.ylabel("arbitrary unit")
  plt.xlabel("Time series (ms)")
  plt.title("comp_" + str(ind) + " explained variance = " + str(round(components_variance[ind],3)))
  plt.savefig("figures/FA_typique_pos_comp_" + str(ind) + ".png".format("PNG"))





##############
## Eleventh step : build factor scores and get components' scores topographies
##############

factor_scores = model.fit_transform(data_ERP)

df_scores = pd.DataFrame(factor_scores)
#save#df_scores.to_excel("scores_typical.xlsx")

myelectrode = df_ERPdata['electrode'] #get clean electrodes
df_scores['electrode'] = myelectrode #add to df scores

df_scores_age = df_scores #keep for later

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

#save score topographies
for ind, comp in enumerate(df_scores):
  circle = plt.Circle((0, 0.05), coordinates.y_coor.max()-0.05, color='k', fill=False, linewidth=2)
  nose = plt.Polygon([(-0.3,coordinates.y_coor.max()), (0,coordinates.y_coor.max()*1.1), (0.3,coordinates.y_coor.max())], color='k', fill=False, linewidth=2)
  plt.gca().add_patch(circle)
  plt.gca().add_patch(nose)
  plt.tricontourf(coordinates.x_coor,coordinates.y_coor, df_scores[comp], cmap='seismic',  levels=125, vmin=df_scores_min, vmax=df_scores_max)
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

for i in range(27):
  fig,ax = plt.subplots(2,1,figsize=(4, 6),gridspec_kw={'height_ratios': [1, 2]})
  fig.suptitle('Component '+str(i+1)+ ', explained variance = '+str(components_variance[i])+'%')
  ax[0].plot(pos_loadings[i])
  ax[0].set_ylim([pos_loadings.min().min()-0.5, pos_loadings.max().max()+0.5])
  ax[0].set_xlabel("Time series (ms)")
  ax[0].set_xticks(ticks=range(0,999,99))
  ax[0].set_xticklabels(['-100','0','100','200','300','400','500','600','700','800','900'])
  ax[1].tricontourf(coordinates.x_coor,coordinates.y_coor, df_scores[i], cmap='seismic', levels=125, alpha=0.9, vmin=df_scores_min, vmax=df_scores_max)
  ax[1].plot(coordinates.x_coor,coordinates.y_coor, 'k.', markersize=7)
  ax[1].set_axis_off()
  circle = plt.Circle((0, 0), coordinates.y_coor.max(), color='k', fill=False, linewidth=2)
  ax[1].add_patch(circle)
  ax[1].plot([-0.3,0,0.3,-0.3], [coordinates.y_coor.max(),coordinates.y_coor.max()*1.1,coordinates.y_coor.max(),coordinates.y_coor.max()], color='k')
  plt.tight_layout()
  plt.savefig("figures/FA_typique_comp_lat_topo_" + str(i) + ".png".format("PNG"))


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









control_somato_two = df_ERPdata[df_ERPdata['condition']==1][df_ERPdata['electrode'].isin([28,29,35,36,41,42,47,52])][df_ERPdata['age']==2][df_ERPdata.columns[0:1000]]




















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