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
font_title = {'family':'Arial','weight':'bold','size': 20}
font_title_x = {'family':'Arial','weight':'bold','size': 30}

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18

# df_cusp
df_cusp_data = pd.read_csv('../cusp/cusp_sin_data_0.2_5.csv')
df_cusp_pred = pd.read_csv('../results/cusp/cusp_sin_pred_0.2_5.csv')
df_cusp_gen = pd.read_csv('../results/cusp/cusp_sin_gen_0.2_5.csv')

train_length_cusp = len(df_cusp_data) - len(df_cusp_pred) - 1

t_cusp = np.arange(len(df_cusp_data))#[::5]
t_cusp_train = np.arange(len(df_cusp_data))[:train_length_cusp]#[::5]
t_cusp_pred = np.arange(len(df_cusp_data))[train_length_cusp:]

x_cusp = df_cusp_data['x']#[::5]
trainx_cusp = df_cusp_data['x'][:train_length_cusp]#[::5]

predx_cusp = df_cusp_pred['pred']
genx_cusp = df_cusp_gen['gen']

#############################################################################

df_cusp_pred_sum = pd.read_csv('../results/cusp/cusp_sin_traj_pred.csv')

trajx_cusp_sum = df_cusp_pred_sum['traj'].to_numpy()
predx_cusp_sum = df_cusp_pred_sum['pred'].to_numpy()

# draw
fig, axs = plt.subplots(1, 3, figsize=(15,5))

ax1, ax2, ax3 = axs[0], axs[1], axs[2]

# ax1
ax1.plot(t_cusp,x_cusp,c='black',linewidth=5,zorder=2)
ax1.plot(t_cusp_train,trainx_cusp,c='silver',linewidth=5,zorder=2)
ax1.scatter(t_cusp_pred[::5],predx_cusp[::5],s=50,marker='o',facecolors='none',edgecolors='crimson',zorder=3)

ax1.set_xlabel('Timepoints',font_x)
ax1.set_ylabel(r'$\mathbf{x}$',font_y,labelpad=-15)
ax1.set_xlim(-50,1050)
ax1.set_xticks([0,700,1000])
ax1.set_xticklabels(['0','700','1000'])
ax1.set_ylim(-2.6,-1.5)
ax1.set_yticks([-2.5,-1.8])
ax1.set_yticklabels(['-2.4','-1.8'])

ax1.tick_params(direction='in')

legend_state = mlines.Line2D([], [], color='black', marker=None, linewidth=5, linestyle='-')
legend_train = mlines.Line2D([], [], color='silver', marker=None, linewidth=5, linestyle='-')
legend_pstate = mlines.Line2D([], [], markerfacecolor='none',color='crimson', marker='o', markersize=5, linestyle='None', markeredgewidth=1.5)

ax1.legend(handles=[legend_state,legend_train,legend_pstate ],labels=['State','Training data','Prediction'],loc='center', frameon=False, bbox_to_anchor=(0.5, 0.85), markerscale=2.5,prop={'size':16})

ax1.text(-0.12, 1.03,'a',ha='left', transform=ax1.transAxes,fontdict={'family':'Arial','size':35,'weight':'bold'})

# ax2
ax2.plot(t_cusp,x_cusp,c='black',linewidth=5,zorder=2)
ax2.scatter(t_cusp[1:][::5],genx_cusp[::5],s=50,marker='o',facecolors='none',edgecolors='darkorange',zorder=3)

ax2.set_xlabel('Timepoints',font_x)
ax2.set_ylabel(r'$\mathbf{x}$',font_y,labelpad=-15)
ax2.set_xlim(-50,1050)
ax2.set_xticks([0,1000])
ax2.set_xticklabels(['0','1000'])
ax2.set_ylim(-2.6,-1.5)
ax2.set_yticks([-2.5,-1.8])
ax2.set_yticklabels(['-2.4','-1.8'])

ax2.tick_params(direction='in')

legend_state = mlines.Line2D([], [], color='black', marker=None, linewidth=5, linestyle='-')
legend_gstate = mlines.Line2D([], [], markerfacecolor='none',color='darkorange', marker='o', markersize=5, linestyle='None', markeredgewidth=1.5)

ax2.legend(handles=[legend_state,legend_gstate],labels=['State','Generation'],loc='center', frameon=False, bbox_to_anchor=(0.5, 0.85), markerscale=2.5,prop={'size':16})

ax2.text(-0.12, 1.03,'b',ha='left', transform=ax2.transAxes,fontdict={'family':'Arial','size':35,'weight':'bold'})

# ax3
ax3.set_aspect('equal')

cusp_r2 = r2_score(trajx_cusp_sum, predx_cusp_sum)

sns.scatterplot(
    x=trajx_cusp_sum,
    y=predx_cusp_sum,
    s=15,
    color='forestgreen',
    edgecolor=None,
    alpha=0.05,
    zorder=1,
    ax=ax3
)
ax3.plot([min(trajx_cusp_sum), max(trajx_cusp_sum)], [min(trajx_cusp_sum), max(trajx_cusp_sum)],color='royalblue',linestyle='--',lw=2,alpha=0.5,zorder=2)
ax3.set_xlabel('True',font_x)
ax3.set_ylabel('Prediction',font_x,labelpad=10)
ax3.tick_params(direction='in')
ax3.set_xticks([-2,0,2])
ax3.set_yticks([-2,0,2])
# ax3.set_title(r'$\mathbf{x}$',fontdict=font_title_x, pad=10)
ax3.text(0.1, 0.8, f'$R^2 = {cusp_r2:.2f}$', transform=ax3.transAxes, fontsize=24, color='black')
ax3.spines['right'].set_visible(False)
ax3.spines['top'].set_visible(False)

ax3.text(-0.12, 1.03,'c',ha='left', transform=ax3.transAxes,fontdict={'family':'Arial','size':35,'weight':'bold'})

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

plt.subplots_adjust(top=0.9, bottom=0.13, left=0.04, right=0.98, wspace=0.25)
plt.savefig('../figures/SFIG11.pdf',format='pdf')