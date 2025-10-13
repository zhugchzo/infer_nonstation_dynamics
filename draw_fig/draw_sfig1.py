import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib import font_manager

font_x = {'family':'Arial','weight':'medium','size': 30}
font_y = {'family':'Arial','weight':'medium','size': 30}

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = 25
plt.rcParams['ytick.labelsize'] = 25

df_smape = pd.read_csv('../results/sMAPE/Koscillators_size_smape.csv')

fig, axs = plt.subplots(1, 1, figsize=(12,10))

network_size = df_smape['network_size'].values

odv = df_smape[['mean_smape', 'lower_smape', 'upper_smape']].values
pars = df_smape[['mean_smape_p', 'lower_smape_p', 'upper_smape_p']].values
timv = df_smape[['mean_smape_t', 'lower_smape_t', 'upper_smape_t']].values

axs.plot(network_size,odv[:,0],c='cornflowerblue',linewidth=5,alpha=0.9,zorder=6)
axs.fill_between(network_size,odv[:,1],odv[:,2],color='cornflowerblue',alpha=0.15,linewidth=0,zorder=5)

axs.plot(network_size,pars[:,0],c='blueviolet',linewidth=5,alpha=0.9,zorder=4)
axs.fill_between(network_size,pars[:,1],pars[:,2],color='blueviolet',alpha=0.15,linewidth=0,zorder=3)

axs.plot(network_size,timv[:,0],c='violet',linewidth=5,alpha=0.9,zorder=2)
axs.fill_between(network_size,timv[:,1],timv[:,2],color='violet',alpha=0.15,linewidth=0,zorder=1)

axs.set_xticks([4,10,20,30,40,50])
axs.set_xticklabels(['4','10','20','30','40','50'])

axs.set_ylim(-0.08,0.88)
axs.set_yticks([0,0.8])
axs.set_yticklabels(['0','0.8'])
axs.tick_params(direction='in')

axs.set_xlabel('Network size',font_x)
axs.set_ylabel('Inference inaccuracy (sMAPE)',font_y)
axs.yaxis.set_label_coords(-0.04, 0.5)

# legend
legend_v = mlines.Line2D([], [], color='cornflowerblue',linestyle='-',linewidth=5,markerfacecolor='cornflowerblue')
legend_p = mlines.Line2D([], [], color='blueviolet', linestyle='-',linewidth=5,markerfacecolor='blueviolet')
legend_t = mlines.Line2D([], [], color='violet', linestyle='-',linewidth=5,markerfacecolor='violet')

fig.legend(
    handles=[legend_v, legend_p, legend_t],
    labels=['Optimal driving variable', 'Forcing parameter', 'Time variable'],
    loc='upper center',
    bbox_to_anchor=(0.5, 1.01),
    ncol=1,
    frameon=False,
    # markerscale=1.5,
    prop=font_manager.FontProperties(family='Arial Unicode MS', size=24)
)

plt.subplots_adjust(top=0.84, bottom=0.09, left=0.1, right=0.9)
plt.savefig('../figures/SFIG1.pdf',format='pdf')
plt.savefig('/Users/zhugchzo/Desktop/3paper_fig/SFIG1.png',format='png',dpi=600)




