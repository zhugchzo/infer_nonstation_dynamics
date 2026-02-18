import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib import font_manager

font_x = {'family':'Arial','weight':'medium','size': 25}
font_y = {'family':'Arial','weight':'medium','size': 25}
font_title = {'family':'DejaVu Sans','weight':'normal','size': 25, 'style': 'italic'}

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20

# df for ax1
df_cusp_smape = pd.read_csv('../results/sMAPE/cusp_smape_interval.csv')
# df for ax2
df_Koscillators_smape = pd.read_csv('../results/sMAPE/Koscillators_smape_interval.csv')

fig, axs = plt.subplots(1, 2, figsize=(12,7))

ax1, ax2 = axs[0], axs[1]

sampling_interval = [0.01, 0.02, 0.03, 0.04, 0.05]

# ax1
cusp_smape = df_cusp_smape[['mean_smape','mean_smape_ab','mean_smape_t']].values
cusp_lower = df_cusp_smape[['lower_smape','lower_smape_ab','lower_smape_t']].values
cusp_upper = df_cusp_smape[['upper_smape','upper_smape_ab','upper_smape_t']].values

smape_ax1 = cusp_smape[:,0]
smape_ab_ax1 = cusp_smape[:,1]
smape_t_ax1 = cusp_smape[:,2]

lower_ax1 = cusp_lower[:,0]
lower_ab_ax1 = cusp_lower[:,1]
lower_t_ax1 = cusp_lower[:,2]

upper_ax1 = cusp_upper[:,0]
upper_ab_ax1 = cusp_upper[:,1]
upper_t_ax1 = cusp_upper[:,2]

ax1.plot(sampling_interval,smape_ax1,c='cornflowerblue',linewidth=5,alpha=0.9,zorder=2)
ax1.scatter(sampling_interval,smape_ax1,c='cornflowerblue',marker='s',s=80,zorder=2)
ax1.fill_between(sampling_interval,lower_ax1,upper_ax1,color='cornflowerblue',alpha=0.15,linewidth=0,zorder=2)

ax1.plot(sampling_interval,smape_ab_ax1,c='blueviolet',linewidth=5,alpha=0.9,zorder=1)
ax1.scatter(sampling_interval,smape_ab_ax1,c='blueviolet',marker='s',s=80,zorder=1)
ax1.fill_between(sampling_interval,lower_ab_ax1,upper_ab_ax1,color='blueviolet',alpha=0.15,linewidth=0,zorder=1)

ax1.plot(sampling_interval,smape_t_ax1,c='violet',linewidth=5,alpha=0.9,zorder=1)
ax1.scatter(sampling_interval,smape_t_ax1,c='violet',marker='s',s=80,zorder=1)
ax1.fill_between(sampling_interval,lower_t_ax1,upper_t_ax1,color='violet',alpha=0.15,linewidth=0,zorder=1)

ax1.set_xlim(0.007,0.053)
ax1.set_xticks([0.01, 0.02, 0.03, 0.04, 0.05])
ax1.set_ylim(-0.1,1.1)
ax1.set_yticks([0,1])
ax1.tick_params(direction='in')

ax1.set_xlabel('Sampling interval',font_x)
ax1.set_ylabel('Inference inaccuracy (sMAPE)',font_y)
ax1.yaxis.set_label_coords(-0.1, 0.45)

ax1.set_title('Cusp bifurcation',fontdict=font_title,y=1.01)
ax1.text(-0.18, 1,'a',ha='left', transform=ax1.transAxes,fontdict={'family':'Arial','size':35,'weight':'bold'})

# ax2
Koscillators_smape = df_Koscillators_smape[['mean_smape','mean_smape_p','mean_smape_t']].values
Koscillators_lower = df_Koscillators_smape[['lower_smape','lower_smape_p','lower_smape_t']].values
Koscillators_upper = df_Koscillators_smape[['upper_smape','upper_smape_p','upper_smape_t']].values

len_ax2 = len(Koscillators_smape)
t_ax2 = np.arange(0,len_ax2)

smape_ax2 = Koscillators_smape[:,0]
smape_ab_ax2 = Koscillators_smape[:,1]
smape_t_ax2 = Koscillators_smape[:,2]

lower_ax2 = Koscillators_lower[:,0]
lower_ab_ax2 = Koscillators_lower[:,1]
lower_t_ax2 = Koscillators_lower[:,2]

upper_ax2 = Koscillators_upper[:,0]
upper_ab_ax2 = Koscillators_upper[:,1]
upper_t_ax2 = Koscillators_upper[:,2]

ax2.plot(sampling_interval,smape_ax2,c='cornflowerblue',linewidth=5,alpha=0.9,zorder=2)
ax2.scatter(sampling_interval,smape_ax2,c='cornflowerblue',marker='s',s=80,zorder=2)
ax2.fill_between(sampling_interval,lower_ax2,upper_ax2,color='cornflowerblue',alpha=0.15,linewidth=0,zorder=2)

ax2.plot(sampling_interval,smape_ab_ax2,c='blueviolet',linewidth=5,alpha=0.9,zorder=1)
ax2.scatter(sampling_interval,smape_ab_ax2,c='blueviolet',marker='s',s=80,zorder=1)
ax2.fill_between(sampling_interval,lower_ab_ax2,upper_ab_ax2,color='blueviolet',alpha=0.15,linewidth=0,zorder=1)

ax2.plot(sampling_interval,smape_t_ax2,c='violet',linewidth=5,alpha=0.9,zorder=1)
ax2.scatter(sampling_interval,smape_t_ax2,c='violet',marker='s',s=80,zorder=1)
ax2.fill_between(sampling_interval,lower_t_ax2,upper_t_ax2,color='violet',alpha=0.15,linewidth=0,zorder=1)

ax2.set_xlim(0.007,0.053)
ax2.set_xticks([0.01, 0.02, 0.03, 0.04, 0.05])
ax2.set_ylim(-0.1,1.1)
ax2.set_yticks([0,1])
ax2.tick_params(direction='in')

ax2.set_xlabel('Sampling interval',font_x)
ax2.set_ylabel('Inference inaccuracy (sMAPE)',font_y)
ax2.yaxis.set_label_coords(-0.1, 0.45)

ax2.set_title('Kuramoto oscillators',fontdict=font_title,y=1.01)
ax2.text(-0.18, 1,'b',ha='left', transform=ax2.transAxes,fontdict={'family':'Arial','size':35,'weight':'bold'})

# legend
legend_v = mlines.Line2D([], [], color='cornflowerblue',linestyle='-',linewidth=5,marker='s',markersize=7,markerfacecolor='cornflowerblue')
legend_p = mlines.Line2D([], [], color='blueviolet', linestyle='-',linewidth=5,marker='s',markersize=7,markerfacecolor='blueviolet')
legend_t = mlines.Line2D([], [], color='violet', linestyle='-',linewidth=5,marker='s',markersize=7,markerfacecolor='violet')

fig.legend(
    handles=[legend_v, legend_p, legend_t],
    labels=['Optimal driving variable','Forcing parameters','Time variable'],
    loc='upper center',
    bbox_to_anchor=(0.5, 1.025),
    ncol=3,
    frameon=False,
    markerscale=1.5,
    prop=font_manager.FontProperties(family='Arial', size=22)
)

plt.subplots_adjust(top=0.8, bottom=0.1, left=0.1, right=0.98, wspace=0.5)
plt.savefig('../figures/SFIG1.pdf',format='pdf')