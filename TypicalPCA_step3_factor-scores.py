__author__ = 'Anne-Lise Marais, see annelisemarais.github.io'
__publication__ = 'Marais, AL., Anquetil, A., Dumont, V., Roche-Labarbe, N. (2023). Somatosensory prediction in typical children from 2 to 6 years old'
__corresponding__ = 'nadege.roche@unicaen.fr'

##############Compute factor scores and plots##############

############
###Load data
############ 

##Load explained variance by component

components_variance = pd.read_csv("data/typical/ERPdata/df_ERP_typ_components_variance.csv",index_col=0)

components_variance_omi = pd.read_csv("data/typical/omission/df_omi_typ_components_variance.csv",index_col=0)

##Load components

components = pd.read_csv("data/typical/ERPdata/df_ERP_components_typical.csv",index_col=0)

components_omi = pd.read_csv("data/typical/omission/df_omi_components_typ.csv",index_col=0)

##Load ERP all condition except omission

#load csv
df_ERPdata = pd.read_csv('data/typical/ERPdata/df_ERP_typ.csv',index_col=0)

#df to numpy to do the PCA
ERPdata = df_ERPdata.drop(['condition', 'electrode','sub','age'], axis=1)
ERPdata = ERPdata.to_numpy()

##Load ERP omission only

#load csv
df_omi = pd.read_csv('data/typical/omission/df_omi_typ.csv',index_col=0)

data_omi = df_omi.drop(['condition', 'electrode','sub','age'], axis=1)
data_omi = data_omi.to_numpy()

############
###Compute factor scores
############

##Factor scores for all conditions except omission

factor_scores = model.fit_transform(ERPdata) #time_series * n_comp

df_scores = pd.DataFrame(factor_scores)

#save
df_scores.to_csv("data/typical/ERPdata/df_ERP_factor_scores_typical.csv")

##Factor scores for omission only

factor_scores_omi = model_omi.fit_transform(data_omi) # n_obs * n_comp

df_scores_omi = pd.DataFrame(factor_scores_omi)

#save
df_factor_scores_omi.to_csv("data/typical/omission/df_omi_factor_scores_typical.csv")


############
###Plot factor scores
############

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
