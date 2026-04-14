import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib import font_manager

font_x = {'family':'Arial','weight':'medium','size': 25}
font_y = {'family':'Arial','weight':'medium','size': 22}

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20

df_cusp_smape = pd.read_csv('../results/sMAPE/cusp_curve_smape.csv')

fig, axs = plt.subplots(1, 1, figsize=(5,7))

ax1 = axs

# ax1
cusp_smape = df_cusp_smape[['mean_smape','mean_smape_ab','mean_smape_t','mean_smape_AIC']].values
cusp_lower = df_cusp_smape[['lower_smape','lower_smape_ab','lower_smape_t','lower_smape_AIC']].values
cusp_upper = df_cusp_smape[['upper_smape','upper_smape_ab','upper_smape_t','upper_smape_AIC']].values

len_ax1 = len(cusp_smape)
t_ax1 = np.arange(0,len_ax1)

smape_ax1 = cusp_smape[:,0]
smape_ab_ax1 = cusp_smape[:,1]
smape_t_ax1 = cusp_smape[:,2]
smape_AIC_ax1 = cusp_smape[:,3]

lower_ax1 = cusp_lower[:,0]
lower_ab_ax1 = cusp_lower[:,1]
lower_t_ax1 = cusp_lower[:,2]
lower_AIC_ax1 = cusp_lower[:,3]

upper_ax1 = cusp_upper[:,0]
upper_ab_ax1 = cusp_upper[:,1]
upper_t_ax1 = cusp_upper[:,2]
upper_AIC_ax1 = cusp_upper[:,3]

ax1.plot(t_ax1,smape_ax1,c='cornflowerblue',linewidth=5,alpha=0.9,zorder=2)
ax1.fill_between(t_ax1,lower_ax1,upper_ax1,color='cornflowerblue',alpha=0.15,linewidth=0,zorder=2)

ax1.plot(t_ax1,smape_ab_ax1,c='blueviolet',linewidth=5,alpha=0.9,zorder=1)
ax1.fill_between(t_ax1,lower_ab_ax1,upper_ab_ax1,color='blueviolet',alpha=0.15,linewidth=0,zorder=1)

ax1.plot(t_ax1,smape_t_ax1,c='violet',linewidth=5,alpha=0.9,zorder=1)
ax1.fill_between(t_ax1,lower_t_ax1,upper_t_ax1,color='violet',alpha=0.15,linewidth=0,zorder=1)

# ax1.plot(t_ax1,smape_AIC_ax1,c='turquoise',linewidth=5,alpha=0.9,zorder=0)
# ax1.fill_between(t_ax1,lower_AIC_ax1,upper_AIC_ax1,color='turquoise',alpha=0.15,linewidth=0,zorder=0)

ax1.set_xticks([0,500,1000])
ax1.set_ylim(-0.07,0.77)
ax1.set_yticks([0,0.7])
ax1.set_yticklabels(['0','0.7'])
ax1.tick_params(direction='in')

ax1.set_xlabel('Timepoint',font_x)
ax1.set_ylabel('Inference inaccuracy (sMAPE)',font_y)
ax1.yaxis.set_label_coords(-0.1, 0.45)

# legend
legend_v = mlines.Line2D([], [], color='cornflowerblue', marker='none', linestyle='-', linewidth=5,alpha=0.9)
legend_p = mlines.Line2D([], [], color='blueviolet', marker='none', linestyle='-', linewidth=5,alpha=0.9)
legend_t = mlines.Line2D([], [], color='violet', marker='none', linestyle='-', linewidth=5,alpha=0.9)
legend_AIC = mlines.Line2D([], [], color='turquoise', marker='none', linestyle='-', linewidth=5,alpha=0.9)

fig.legend(
    handles=[legend_v, legend_p, legend_t],
    labels=['Optimal driving variable', 'Forcing parameters', 'Time variable'],
    loc='upper center',
    bbox_to_anchor=(0.5, 1.025),
    ncol=1,
    frameon=False,
    markerscale=1.5,
    prop=font_manager.FontProperties(family='Arial', size=18)
)

plt.subplots_adjust(top=0.8, bottom=0.1, left=0.17, right=0.96)
plt.savefig('../figures/SFIG13.pdf',format='pdf')
plt.savefig('/Users/zhugchzo/Desktop/3paper_fig/SFIG13.png',format='png',dpi=600)
