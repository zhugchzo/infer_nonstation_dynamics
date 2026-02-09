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
df_cusp_ned = pd.read_csv('../results/NED/cusp_NED.csv')
# df for ax2
df_Koscillators_ned = pd.read_csv('../results/NED/Koscillators_NED.csv')

fig, axs = plt.subplots(1, 2, figsize=(12,7))

ax1, ax2 = axs[0], axs[1]

# ax1
cusp_ned = df_cusp_ned[['mean_NED','mean_NED_ab','mean_NED_t']].values
cusp_lower = df_cusp_ned[['lower_NED','lower_NED_ab','lower_NED_t']].values
cusp_upper = df_cusp_ned[['upper_NED','upper_NED_ab','upper_NED_t']].values

len_ax1 = len(cusp_ned)
t_ax1 = np.arange(0,len_ax1)

ned_ax1 = cusp_ned[:,0]
ned_ab_ax1 = cusp_ned[:,1]
ned_t_ax1 = cusp_ned[:,2]

lower_ax1 = cusp_lower[:,0]
lower_ab_ax1 = cusp_lower[:,1]
lower_t_ax1 = cusp_lower[:,2]

upper_ax1 = cusp_upper[:,0]
upper_ab_ax1 = cusp_upper[:,1]
upper_t_ax1 = cusp_upper[:,2]

ax1.plot(t_ax1,ned_ax1,c='cornflowerblue',linewidth=5,alpha=0.9,zorder=2)
ax1.fill_between(t_ax1,lower_ax1,upper_ax1,color='cornflowerblue',alpha=0.15,linewidth=0,zorder=2)

ax1.plot(t_ax1,ned_ab_ax1,c='blueviolet',linewidth=5,alpha=0.9,zorder=1)
ax1.fill_between(t_ax1,lower_ab_ax1,upper_ab_ax1,color='blueviolet',alpha=0.15,linewidth=0,zorder=1)

ax1.plot(t_ax1,ned_t_ax1,c='violet',linewidth=5,alpha=0.9,zorder=1)
ax1.fill_between(t_ax1,lower_t_ax1,upper_t_ax1,color='violet',alpha=0.15,linewidth=0,zorder=1)

ax1.set_xticks([0,250,500])
ax1.set_ylim(-0.04,0.44)
ax1.set_yticks([0,0.4])
ax1.set_yticklabels(['0','0.4'])
ax1.tick_params(direction='in')

ax1.set_xlabel('Timepoints',font_x)
ax1.set_ylabel('Prediction inaccuracy (NED)',font_y)
ax1.yaxis.set_label_coords(-0.15, 0.5)

ax1.set_title('Cusp bifurcation',fontdict=font_title,y=1.01)
ax1.text(-0.18, 1,'a',ha='left', transform=ax1.transAxes,fontdict={'family':'Arial','size':35,'weight':'bold'})

# ax2
Koscillators_ned = df_Koscillators_ned[['mean_NED','mean_NED_p','mean_NED_t']].values
Koscillators_lower = df_Koscillators_ned[['lower_NED','lower_NED_p','lower_NED_t']].values
Koscillators_upper = df_Koscillators_ned[['upper_NED','upper_NED_p','upper_NED_t']].values

len_ax2 = len(Koscillators_ned)
t_ax2 = np.arange(0,len_ax2)

ned_ax2 = Koscillators_ned[:,0]
ned_ab_ax2 = Koscillators_ned[:,1]
ned_t_ax2 = Koscillators_ned[:,2]

lower_ax2 = Koscillators_lower[:,0]
lower_ab_ax2 = Koscillators_lower[:,1]
lower_t_ax2 = Koscillators_lower[:,2]

upper_ax2 = Koscillators_upper[:,0]
upper_ab_ax2 = Koscillators_upper[:,1]
upper_t_ax2 = Koscillators_upper[:,2]

ax2.plot(t_ax2,ned_ax2,c='cornflowerblue',linewidth=5,alpha=0.9,zorder=2)
ax2.fill_between(t_ax2,lower_ax2,upper_ax2,color='cornflowerblue',alpha=0.15,linewidth=0,zorder=2)

ax2.plot(t_ax2,ned_ab_ax2,c='blueviolet',linewidth=5,alpha=0.9,zorder=1)
ax2.fill_between(t_ax2,lower_ab_ax2,upper_ab_ax2,color='blueviolet',alpha=0.15,linewidth=0,zorder=1)

ax2.plot(t_ax2,ned_t_ax2,c='violet',linewidth=5,alpha=0.9,zorder=1)
ax2.fill_between(t_ax2,lower_t_ax2,upper_t_ax2,color='violet',alpha=0.15,linewidth=0,zorder=1)

ax2.set_xticks([0,350,700])
ax2.set_ylim(-0.0025,0.0275)
ax2.set_yticks([0,0.025])
ax2.set_yticklabels(['0','0.025'])
ax2.tick_params(direction='in')

ax2.set_xlabel('Timepoints',font_x)
ax2.set_ylabel('Prediction inaccuracy (NED)',font_y)
ax2.yaxis.set_label_coords(-0.2, 0.5)

ax2.set_title('Kuramoto oscillators',fontdict=font_title,y=1.01)
ax2.text(-0.18, 1,'b',ha='left', transform=ax2.transAxes,fontdict={'family':'Arial','size':35,'weight':'bold'})

# legend
legend_v = mlines.Line2D([], [], color='cornflowerblue', marker='none', linestyle='-', linewidth=5,alpha=0.9)
legend_p = mlines.Line2D([], [], color='blueviolet', marker='none', linestyle='-', linewidth=5,alpha=0.9)
legend_t = mlines.Line2D([], [], color='violet', marker='none', linestyle='-', linewidth=5,alpha=0.9)

fig.legend(
    handles=[legend_v, legend_p, legend_t],
    labels=['Our approach', 'RC with forcing parameters', 'RC with time variable'],
    loc='upper center',
    bbox_to_anchor=(0.5, 1.025),
    ncol=3,
    frameon=False,
    markerscale=1.5,
    prop=font_manager.FontProperties(family='Arial Unicode MS', size=20)
)

plt.subplots_adjust(top=0.8, bottom=0.1, left=0.1, right=0.98, wspace=0.5)
plt.savefig('../figures/SFIG10.pdf',format='pdf')