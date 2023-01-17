########################
# LOADING DATA
########################

from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Rectangle

# load atypical and typical matlab file (as dictionnary)
aty_ML = loadmat('data/Atypical.mat') #output is dict
ty_ML = loadmat('data/Typical.mat') #output is dict
#dict to np.array
aty_arrays = aty_ML['Atypical'] #output is np of arrays (3,8)
ty_arrays = ty_ML['Typical'] #output is np of arrays (3,8)

# function to melt data (needed formating for seaborn)
def rawmatrix2melted(ROI_Matrix,Neurodev_status):
    #Example fam_somato_aty = rawmatrix2melted(aty_ML[0,2],'atypical')
    Cond_ROI_NS = pd.DataFrame(ROI_Matrix)
    Cond_ROI_NS["sub"] = Cond_ROI_NS.index +1
    Cond_ROI_NS['status'] = Neurodev_status
    Cond_ROI_NS = Cond_ROI_NS.melt(id_vars=['sub', 'status'], ignore_index=False)
    return Cond_ROI_NS

# launch the data melting for each arrays for typical AND atypical
aty_status = []
ty_status = []

for ind_row in range(0, 3):
    for ind_col in range(0, 8):
        result_atyp = rawmatrix2melted(aty_arrays[ind_row][ind_col],'atypical')
        result_typ = rawmatrix2melted(ty_arrays[ind_row][ind_col],'typical')
        ty_status.append(result_typ)
        aty_status.append(result_atyp)
aty_status = np.array(aty_status).reshape(3,8)
ty_status = np.array(ty_status).reshape(3,8)

# cheking
assert (rawmatrix2melted(aty_arrays[0,1],'atypical') != aty_status[0,1]).sum().sum() ==0
assert (rawmatrix2melted(ty_arrays[0,1],'typical') != ty_status[0,1]).sum().sum() ==0



########################
# PLOTTING FIG RS -> 4 figures
########################

sns.set_style("ticks")
plt.rcParams['xtick.major.pad']='10'
plt.rcParams['ytick.major.pad']='10'
###############################
# small fig 1
###############################
f, ax = plt.subplots(figsize=(10, 10))
plt.subplots_adjust(top=0.959,
bottom=0.111,
left=0.092,
right=0.971,
hspace=0.2,
wspace=0.2)

sns.lineplot(ax=ax, x='variable', y="value",
             data=ty_status[0][2], color = 'green')
sns.lineplot(ax=ax,x='variable', y="value", 
            data=ty_status[0][0], color = 'purple')
ax.set_title('Somatosensory repetition suppression of typical children',fontsize=24)
ax.grid(False)

ax.set_xlabel("ms", fontsize = 38)
ax.set_ylabel("µV", fontsize = 38)

ax.set_xlim(0, 999)
ax.set_ylim(-7.5, 6)

ax.set_xticks(range(0,999,99))
ax.set_xticklabels(['-100','0','100','200','300','400','500','600','700','800','900'], fontsize=32)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.tick_params(axis='y',labelsize=32)
ax.invert_yaxis()

ax.add_patch(Rectangle((99, -7.5), 200, 13.5,
             facecolor = ('grey'),
             edgecolor = None,
             alpha = 0.3,
             fill=True,
             lw=0))

#ax.text(x=160,y=-6.5,s='Stimulation', fontsize = 20)

plt.tight_layout()
plt.show()

