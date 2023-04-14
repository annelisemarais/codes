__author__ = 'Anne-Lise Marais, see annelisemarais.github.io'
__publication__ = 'Marais, AL., Anquetil, A., Dumont, V., Roche-Labarbe, N. (2023). Somatosensory prediction in typical children from 2 to 6 years old'
__corresponding__ = 'nadege.roche@unicaen.fr'


##############ERP Visualization##############

###Load data and process to plot ERPs

##ERP all condition except omission

#load csv
df_ERPdata = pd.read_csv('data/typical/ERPdata/df_ERP_typ.csv',index_col=0)

#df to numpy to do the PCA
ERPdata = df_ERPdata.drop(['condition', 'electrode','sub','age'], axis=1)
ERPdata = ERPdata.to_numpy()

#function to tranform df to plottable data
def df2data4plot(df,condition):
  data = df[df['condition']==condition]
  data = data.groupby('electrode', as_index=False).mean()
  data = data.drop(['sub','electrode'], axis=1)
  return data

#apply func to all conditions
familiarization = df2data4plot(df_ERPdata,3)
control = df2data4plot(df_ERPdata,1)
deviant = df2data4plot(df_ERPdata,2)
standard = df2data4plot(df_ERPdata,7)
postom = df2data4plot(df_ERPdata,5)

##ERP omission only

#load csv
df_omi = pd.read_csv('data/typical/omission/df_omi_typ.csv',index_col=0)

#mean data by electrode for this single condition
omission = df_omi.drop(['condition', 'sub','age'], axis=1)
omission =omission.groupby('electrode', as_index=False).mean()
omission = omission.drop(['electrode'], axis=1)

###Plot ERP

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