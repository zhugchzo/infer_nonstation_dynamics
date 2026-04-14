import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.lines as mlines
from matplotlib import font_manager
from sklearn.metrics import r2_score

font = font_manager.FontProperties(family='Arial',weight='medium')
font_AIC = font_manager.FontProperties(family='Arial',weight='bold')

font_x = {'family':'Arial','weight':'bold','size': 20}
font_y = {'family':'Arial','weight':'bold','size': 24}
font_title = {'family':'Arial','weight':'normal','color':'navy','size': 25}

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18

df_cusp_pred_sum = pd.read_csv('../results/cusp/cusp_sin_traj_pred.csv')

trajx_cusp_sum = df_cusp_pred_sum['traj'].to_numpy()
predx_cusp_sum = df_cusp_pred_sum['pred'].to_numpy()

df_cusp_k_pred_sum = pd.read_csv('../results/cusp/cusp_sin_known_traj_pred.csv')

trajx_cusp_k_sum = df_cusp_k_pred_sum['traj'].to_numpy()
predx_cusp_k_sum = df_cusp_k_pred_sum['pred'].to_numpy()

# draw
fig, axs = plt.subplots(1, 2, figsize=(10,5))

ax1, ax2 = axs[0], axs[1]

# ax1
ax1.set_aspect('equal')

cusp_r2 = r2_score(trajx_cusp_sum, predx_cusp_sum)

sns.scatterplot(
    x=trajx_cusp_sum,
    y=predx_cusp_sum,
    s=15,
    color='forestgreen',
    edgecolor=None,
    alpha=0.05,
    zorder=1,
    ax=ax1
)
ax1.plot([min(trajx_cusp_sum), max(trajx_cusp_sum)], [min(trajx_cusp_sum), max(trajx_cusp_sum)],color='royalblue',linestyle='--',lw=2,alpha=0.5,zorder=2)
ax1.set_xlabel('True',font_x)
ax1.set_ylabel('Prediction',font_x,labelpad=10)
ax1.tick_params(direction='in')
ax1.set_xticks([-2,0,2])
ax1.set_yticks([-2,0,2])
ax1.set_title('Unknown trend',fontdict=font_title, pad=10)
ax1.text(0.1, 0.8, f'$R^2 = {cusp_r2:.2f}$', transform=ax1.transAxes, fontsize=24, color='black')
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)

ax1.text(-0.12, 1.03,'a',ha='left', transform=ax1.transAxes,fontdict={'family':'Arial','size':35,'weight':'bold'})

# ax2
ax2.set_aspect('equal')

cusp_k_r2 = r2_score(trajx_cusp_k_sum, predx_cusp_k_sum)

sns.scatterplot(
    x=trajx_cusp_k_sum,
    y=predx_cusp_k_sum,
    s=15,
    color='forestgreen',
    edgecolor=None,
    alpha=0.05,
    zorder=1,
    ax=ax2
)
ax2.plot([min(trajx_cusp_k_sum), max(trajx_cusp_k_sum)], [min(trajx_cusp_k_sum), max(trajx_cusp_k_sum)],color='royalblue',linestyle='--',lw=2,alpha=0.5,zorder=2)
ax2.set_xlabel('True',font_x)
ax2.set_ylabel('Prediction',font_x,labelpad=10)
ax2.tick_params(direction='in')
ax2.set_xticks([-2,0,2])
ax2.set_yticks([-2,0,2])
ax2.set_title('Known trend',fontdict=font_title, pad=10)
ax2.text(0.1, 0.8, f'$R^2 = {cusp_k_r2:.2f}$', transform=ax2.transAxes, fontsize=24, color='black')
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)

ax2.text(-0.12, 1.03,'b',ha='left', transform=ax2.transAxes,fontdict={'family':'Arial','size':35,'weight':'bold'})

# legend_state = mlines.Line2D([], [], color='black', marker=None, linewidth=5, linestyle='-')
# legend_train = mlines.Line2D([], [], color='silver', marker=None, linewidth=5, linestyle='-')
# legend_pstate = mlines.Line2D([], [], markerfacecolor='none',color='crimson', marker='o', markersize=8, linestyle='None', markeredgewidth=1.5)
# legend_gstate = mlines.Line2D([], [], markerfacecolor='none',color='darkorange', marker='o', markersize=8, linestyle='None', markeredgewidth=1.5)

# fig.legend(
#     handles=[legend_state,legend_train,legend_pstate,legend_gstate],
#     labels=['System state','Training data','Prediction','Generation'],
#     loc='upper center',
#     bbox_to_anchor=(0.5, 1),
#     ncol=4,
#     frameon=False,
#     markerscale=2.5,
#     prop=font_manager.FontProperties(family='Arial Unicode MS', size=18)
# )

plt.subplots_adjust(top=0.9, bottom=0.13, left=0.06, right=0.98, wspace=0.2)
plt.savefig('../figures/SFIG14.pdf',format='pdf')