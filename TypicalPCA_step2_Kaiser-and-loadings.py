__author__ = 'Anne-Lise Marais, see annelisemarais.github.io'
__publication__ = 'Marais, AL., Anquetil, A., Dumont, V., Roche-Labarbe, N. (2023). Somatosensory prediction in typical children from 2 to 6 years old'
__corresponding__ = 'nadege.roche@unicaen.fr'


##############Kaiser model and loadings##############

############
###Load data
############

##ERP all condition except omission

#load csv
df_ERPdata = pd.read_csv('data/typical/ERPdata/df_ERP_typ.csv',index_col=0)

#df to numpy to do the PCA
ERPdata = df_ERPdata.drop(['condition', 'electrode','sub','age'], axis=1)
ERPdata = ERPdata.to_numpy()

##ERP omission only

#load csv
df_omi = pd.read_csv('data/typical/omission/df_omi_typ.csv',index_col=0)

#mean data by electrode for this single condition
omission = df_omi.drop(['condition', 'sub','age'], axis=1)
omission =omission.groupby('electrode', as_index=False).mean()
omission = omission.drop(['electrode'], axis=1)

############
###Kaiser estimations
############

##Kaiser for ERPs except omission

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

df_components_variance = pd.DataFrame(components_variance)

#save
df_components_variance.to_csv("data/typical/ERPdata/df_ERP_typ_components_variance.csv")

##Kaiser for omission only 

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

############
###Plot scree plots
############

##scree plot for all conditions except omission
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
df_components_variance_omi.to_csv("data/typical/omission/df_omi_typ_components_variance.csv")

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

############
###Compute component loadings
############

##Loadings for all conditions except omission

model = FactorAnalysis(n_components=n_components, rotation='varimax')
model.fit(ERPdata)
components = model.components_

df_components = pd.DataFrame(components) #nb_components * 1000 time series

#save
df_components.to_csv("data/typical/ERPdata/df_ERP_components_typical.csv")

##Loadings for omission only

model_omi = FactorAnalysis(n_components=n_components_omi, rotation='varimax')
model_omi.fit(data_omi)
components_omi = model_omi.components_

df_components_omi = pd.DataFrame(components_omi) #nb_components * 1000 time series

#save
df_components_omi.to_csv("data/typical/omission/df_omi_components_typ.csv")

############
###Plot loadings
############

#if components[n] peak is negative, invert
pos_loadings = model.components_
for ind, comp in enumerate(pos_loadings):
  peakmax = max(comp)
  peakmin = abs(min(comp))
  if peakmax < peakmin:
    pos_loadings[ind] = np.negative(comp)


# plot the loadings:
def plot_ERP_loadings(loadings):
  plt.close('all')
  for ind, comp in enumerate(loadings):
    plt.plot(range(0, 1000), comp, linewidth=3)
  plt.xlabel("Time series (ms)")
  plt.ylabel("Component loadings")
  plt.xlim(0,999)
  plt.xticks(ticks=range(0,999,99), labels =['-100','0','100','200','300','400','500','600','700','800','900'])
  plt.savefig("figures/typical/FA_loadings_ERP_typ.png")

plot_ERP_loadings(pos_loadings)
plt.show()


pos_loadings_omi = model_omi.components_
for ind, comp in enumerate(pos_loadings_omi):
  peakmax = max(comp)
  peakmin = abs(min(comp))
  if peakmax < peakmin:
    pos_loadings_omi[ind] = np.negative(comp)


# plot the loadings:
def plot_omi_loadings(loadings):
  plt.close('all')
  for ind, comp in enumerate(loadings):
    plt.plot(range(0, 1000), comp, linewidth=3)
  plt.xlabel("Time series (ms)")
  plt.xlim(0,999)
  plt.ylim([-3,9])
  plt.ylabel("Component loadings")
  plt.xticks(ticks=range(0,999,99), labels =['-500','-400','-300','-200','-100','0','100','200','300','400','500'])
  plt.savefig("figures/typical/omission/FA_loadings_omi_typ.png")

plot_omi_loadings(pos_loadings_omi)
plt.show()

############
###Get loadings max to determine their accending latencies
############

##loading max for all condition except omission

loadings_max = {}
for ind, comp in enumerate(pos_loadings):
  load_max = np.argmax(pos_loadings[ind]) - 100
  loadings_max[ind+1] = load_max

sorted_max = dict(sorted(loadings_max.items(), key=lambda item: item[1]))
df_max = pd.DataFrame(data=sorted_max, index=[0]).T
df_max.to_excel('data/typical/ERPdata/df_ERP_max_loadings.xlsx')

##loading max omission only

loadings_max_omi = {}
for ind, comp in enumerate(pos_loadings_omi):
  load_max_omi = np.argmax(pos_loadings_omi[ind]) - 500
  loadings_max_omi[ind+1] = load_max_omi

sorted_max_omi = dict(sorted(loadings_max_omi.items(), key=lambda item: item[1]))
df_max_omi = pd.DataFrame(data=sorted_max_omi, index=[0]).T
df_max_omi.to_excel('data/typical/omission/df_omi_max_loadings.xlsx')