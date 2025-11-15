import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.lines as mlines
from matplotlib import font_manager
from matplotlib.patches import FancyArrow
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
df_cusp_AIC = pd.read_csv('../results/cusp/cusp_AIC.csv', header=None)
df_cusp_pred = pd.read_csv('../results/cusp/cusp_pred.csv')
df_cusp_pred_sum = pd.read_csv('../results/cusp/cusp_traj_pred.csv')
df_cusp_data = pd.read_csv('../cusp/cusp_data.csv')

cusp_AIC = df_cusp_AIC.to_numpy()[1:,1:] # shape = (11,12)

train_length_cusp = len(df_cusp_data) - len(df_cusp_pred) - 1

t_cusp = np.arange(len(df_cusp_data))#[::5]
t_cusp_train = np.arange(len(df_cusp_data))[:train_length_cusp]#[::5]

x_cusp = df_cusp_data['x']#[::5]
trainx_cusp = df_cusp_data['x'][:train_length_cusp]#[::5]

trajx_cusp = df_cusp_pred_sum['traj'].to_numpy()
predx_cusp = df_cusp_pred_sum['pred'].to_numpy()

cusp_AIC_filtered = cusp_AIC[cusp_AIC != 1e5]
max_value_cusp = np.max(cusp_AIC_filtered)
cusp_AIC[cusp_AIC == 1e5] = max_value_cusp

# df_Koscillators
df_Koscillators_AIC = pd.read_csv('../results/Koscillators/Koscillators_AIC.csv', header=None)
df_Koscillators_pred = pd.read_csv('../results/Koscillators/Koscillators_pred.csv')
df_Koscillators_pred_sum = pd.read_csv('../results/Koscillators/Koscillators_traj_pred.csv')
df_Koscillators_data = pd.read_csv('../Koscillators/Koscillators_data.csv')

col_traj = ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10']
# col_pred = ['predx_0', 'predx_1', 'predx_2', 'predx_3', 'predx_4', 'predx_5', 'predx_6', 'predx_7', 'predx_8', 'predx_9', 'predx_10']

Koscillators_AIC = df_Koscillators_AIC.to_numpy()[1:,1:]

train_length_Koscillators = len(df_Koscillators_data) - len(df_Koscillators_pred) - 1

t_Koscillators = np.arange(len(df_Koscillators_data))#[::5]
t_Koscillators_train = np.arange(len(df_Koscillators_data))[:train_length_Koscillators]#[::5]

x_Koscillators = df_Koscillators_data[col_traj]
trainx_Koscillators = df_Koscillators_data[col_traj][:train_length_Koscillators]

trajx_Koscillators = df_Koscillators_pred_sum['traj'].to_numpy()#[::5]
predx_Koscillators = df_Koscillators_pred_sum['pred'].to_numpy()#[::5]

Koscillators_AIC_filtered = Koscillators_AIC[Koscillators_AIC != 1e5]
max_value_Koscillators = np.max(Koscillators_AIC_filtered)
Koscillators_AIC[Koscillators_AIC == 1e5] = max_value_Koscillators

# draw
fig, axs = plt.subplots(2, 4, figsize=(22,11))

ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8 = axs[0,0], axs[0,1], axs[0,2], axs[0,3], axs[1,0], axs[1,1], axs[1,2], axs[1,3]

x_labels = [r'$1\times10^{-5}$', r'$5\times10^{-5}$', r'$1\times10^{-4}$', r'$5\times10^{-4}$', r'$1\times10^{-3}$',
            r'$5\times10^{-3}$', r'$1\times10^{-2}$', r'$5\times10^{-2}$', r'$1\times10^{-1}$', r'$5\times10^{-1}$', '1', '5']
y_labels = ['-20', '-15', '-10', '-5', '-1', '0', '1', '5', '10', '15', '20']

# ax1
ax1.plot(t_cusp,x_cusp,c='black',linewidth=5,zorder=2)
ax1.plot(t_cusp_train,trainx_cusp,c='silver',linewidth=5,zorder=2)

legend_state = mlines.Line2D([], [], color='black', marker=None, linewidth=5, linestyle='-')
legend_train = mlines.Line2D([], [], color='silver', marker=None, linewidth=5, linestyle='-')

ax1.legend(handles=[legend_state,legend_train],labels=['State','Training data'],loc='center', frameon=False, bbox_to_anchor=(0.35, 0.8), markerscale=2.5,prop={'size':16})

ax1.set_xlabel('Timepoints',font_x)
ax1.set_ylabel(r'$\mathbf{x}$',font_y,labelpad=10)
ax1.set_xlim(-50,1050)
ax1.set_xticks([0,500,1000])
ax1.set_xticklabels(['0','500','1000'])
ax1.set_ylim(-2.3,2.4)
ax1.set_yticks([-2,2])
ax1.set_yticklabels(['-2','2'])

ax1.tick_params(direction='in')
ax1.set_title('Cusp bifurcation',fontdict=font_title, pad=10)

ax1.text(-0.12, 1.1,'a',ha='left', transform=ax1.transAxes,fontdict={'family':'Arial','size':35,'weight':'bold'})

# ax2
ax2.axis('off')

ax2.text(0.1,1,'True cusp bifurcation dynamics :',fontdict={'family':'Arial','size':20,'weight':'medium'},color='royalblue')
ax2.text(0.3,0.8,r'$\dot{x}=\phi_1+\phi_2x-x^3$',fontdict={'family':'Arial','size':20,'weight':'medium'})
ax2.text(0.2,0.65,r'$(\phi_1,\phi_2):(1,4)\rightarrow(3,3)$',fontdict={'family':'Arial','size':20,'weight':'medium'})

ax2.text(0.3,0.3,'Inferred equation :',fontdict={'family':'Arial','size':20,'weight':'medium'},color='royalblue')
ax2.text(-0.05,0.1,r'$\dot{x}=(1.4+0.4\nu)+(3.8-0.2\nu)x-x^3$',fontdict={'family':'Arial','size':20,'weight':'medium'})
ax2.text(0.35,-0.05,r'$\nu:-1\rightarrow4$',fontdict={'family':'Arial','size':20,'weight':'medium'})

ax2.text(-0.12, 1.1,'b',ha='left', transform=ax2.transAxes,fontdict={'family':'Arial','size':35,'weight':'bold'})

# ax3
heatmap = sns.heatmap(cusp_AIC, 
            cmap='PuBu',
            linewidths=0.005, 
            linecolor='silver',
            annot=False,
            square=True,
            ax=ax3,
            cbar_kws={'shrink': 0.75
            })

heatmap.set_xticks(np.arange(len(x_labels)) + 0.7)
heatmap.set_yticks(np.arange(len(y_labels)) + 0.5)
heatmap.set_xticklabels(x_labels, rotation=90, ha='right', fontproperties=font)
heatmap.set_yticklabels(y_labels, rotation=0, fontproperties=font)

heatmap.tick_params(axis='both', length=0) 

cbar = heatmap.collections[0].colorbar
cbar.set_ticks([-2e6, -1e6])
cbar.set_ticklabels(['-2', '-1'])
cbar.ax.text(0.8, -2.5e6, r'$10^{6}$', ha='center', va='top', fontsize=15, fontproperties=font)
cbar.ax.text(3.5, -0.25e6, r'$\epsilon$AIC', ha='center', va='top', fontsize=18, fontproperties=font_AIC)
cbar.ax.tick_params(length=0)

for label in cbar.ax.get_yticklabels():
    label.set_fontproperties(font)
    label.set_fontsize(18)

for label in ax3.get_xticklabels():
    label.set_fontproperties(font)
    label.set_fontsize(12)

for label in ax3.get_yticklabels():
    label.set_fontproperties(font)
    label.set_fontsize(15)

ax3.set_title('Grid search result :'+'\n'+r'$\nu_1=-1,\,\Delta \nu=5\times10^{-3}$',fontdict=font_title, pad=10)

# ax4
ax4.set_aspect('equal')

cusp_r2 = r2_score(trajx_cusp, predx_cusp)

sns.scatterplot(
    x=trajx_cusp,
    y=predx_cusp,
    s=15,
    color='forestgreen',
    edgecolor=None,
    alpha=0.05,
    zorder=1,
    ax=ax4
)
ax4.plot([min(trajx_cusp), max(trajx_cusp)], [min(trajx_cusp), max(trajx_cusp)],color='royalblue',linestyle='--',lw=2,alpha=0.5,zorder=2)
ax4.set_xlabel('True',font_x)
ax4.set_ylabel('Prediction',font_x,labelpad=10)
ax4.tick_params(direction='in')
ax4.set_title(r'$\mathbf{x}$',fontdict=font_title_x, pad=10)
ax4.text(0.1, 0.8, f'$R^2 = {cusp_r2:.2f}$', transform=ax4.transAxes, fontsize=24, color='black')
ax4.spines['right'].set_visible(False)
ax4.spines['top'].set_visible(False)

ax4.text(-0.12, 1.1,'c',ha='left', transform=ax4.transAxes,fontdict={'family':'Arial','size':35,'weight':'bold'})

# ax5

ax5.plot(t_Koscillators,x_Koscillators,c='black',linewidth=5,zorder=2)
ax5.plot(t_Koscillators_train,trainx_Koscillators,c='silver',linewidth=5,zorder=2)

legend_state = mlines.Line2D([], [], color='black', marker=None, linewidth=5, linestyle='-')
legend_train = mlines.Line2D([], [], color='silver', marker=None, linewidth=5, linestyle='-')

ax5.legend(handles=[legend_state,legend_train],labels=['Phase','Training data'],loc='center', frameon=False, bbox_to_anchor=(0.3, 0.4), markerscale=2.5,prop={'size':16})

ax5.set_xlabel('Timepoints',font_x)
ax5.set_ylabel(r'$\theta_i$',font_y,labelpad=-5)
ax5.set_xlim(-49.5,1049.5)
ax5.set_xticks([0,300,1000])
ax5.set_xticklabels(['0','300','1000'])
ax5.set_ylim(-45,35)
ax5.set_yticks([-40,0,30])
ax5.set_yticklabels(['-40','0','30'])

ax5.tick_params(direction='in')
ax5.set_title('Kuramoto oscillators',fontdict=font_title, pad=10)

ax5.text(-0.12, 1.1,'d',ha='left', transform=ax5.transAxes,fontdict={'family':'Arial','size':35,'weight':'bold'})

# ax6
ax6.axis('off')

ax6.text(0.05,1,'True Kuramoto oscillators dynamics :',fontdict={'family':'Arial','size':20,'weight':'medium'},color='royalblue')
ax6.text(0.2,0.8,r'$\dot{\theta_i}=\omega_i+\sigma\sum_{j=1}^{N}A_{ij}\sin(\theta_j-\theta_i)$',fontdict={'family':'Arial','size':20,'weight':'medium'})
ax6.text(0.25,0.6,r'$\sum_{i=1}^{N}\omega_i=0,\,\sigma:0\rightarrow1$',fontdict={'family':'Arial','size':20,'weight':'medium'})

ax6.text(0.3,0.3,'Inferred equation :',fontdict={'family':'Arial','size':20,'weight':'medium'},color='royalblue')
ax6.text(-0.1,0.1,r'$\dot{\theta_i}=(0.0127+0.0202\nu)\sum_{j=1}^{N}A_{ij}\sin(\theta_j-\theta_i)$',fontdict={'family':'Arial','size':20,'weight':'medium'})
ax6.text(0.35,-0.1,r'$\nu:-1\rightarrow49$',fontdict={'family':'Arial','size':20,'weight':'medium'})

ax6.text(-0.12, 1.1,'e',ha='left', transform=ax6.transAxes,fontdict={'family':'Arial','size':35,'weight':'bold'})

# ax7
heatmap = sns.heatmap(Koscillators_AIC, 
            cmap='PuBu',
            linewidths=0.005, 
            linecolor='silver',
            annot=False,
            square=True,
            ax=ax7,
            cbar_kws={'shrink': 0.75
            })

heatmap.set_xticks(np.arange(len(x_labels)) + 0.7)
heatmap.set_yticks(np.arange(len(y_labels)) + 0.5)
heatmap.set_xticklabels(x_labels, rotation=90, ha='right', fontproperties=font)
heatmap.set_yticklabels(y_labels, rotation=0)

heatmap.tick_params(axis='both', length=0) 

cbar = heatmap.collections[0].colorbar
cbar.set_ticks([-6e23, -3e23])
cbar.set_ticklabels(['-6', '-3'])
cbar.ax.text(1, -8.5e23, r'$10^{23}$', ha='center', va='top', fontsize=15, fontproperties=font)
cbar.ax.text(3.5, -0.3e23, r'$\epsilon$AIC', ha='center', va='top', fontsize=18, fontproperties=font_AIC)
cbar.ax.tick_params(length=0)

for label in cbar.ax.get_yticklabels():
    label.set_fontproperties(font)
    label.set_fontsize(18)

for label in ax7.get_xticklabels():
    label.set_fontproperties(font)
    label.set_fontsize(12)

for label in ax7.get_yticklabels():
    label.set_fontproperties(font)
    label.set_fontsize(15)

ax7.set_title('Grid search result :'+'\n'+r'$\nu_1=-1,\,\Delta \nu=5\times10^{-2}$',fontdict=font_title, pad=10)

# ax8
ax8.set_aspect('equal')

Koscillators_r2 = r2_score(trajx_Koscillators, predx_Koscillators)

sns.scatterplot(
    x=trajx_Koscillators,
    y=predx_Koscillators,
    s=15,
    color='forestgreen',
    edgecolor=None,
    alpha=0.05,
    zorder=1,
    ax=ax8
)
ax8.plot([min(trajx_Koscillators), max(trajx_Koscillators)], [min(trajx_Koscillators), max(trajx_Koscillators)],color='royalblue',linestyle='--',lw=2,alpha=0.5,zorder=2)
ax8.set_xlabel('True',font_x)
ax8.set_ylabel('Prediction',font_x,labelpad=10)
ax8.tick_params(direction='in')
ax8.set_title(r'$\theta_i$',fontdict=font_title_x, pad=10)
ax8.text(0.1, 0.8, f'$R^2 = {Koscillators_r2:.2f}$', transform=ax8.transAxes, fontsize=24, color='black')
ax8.spines['right'].set_visible(False)
ax8.spines['top'].set_visible(False)

ax8.text(-0.12, 1.1,'f',ha='left', transform=ax8.transAxes,fontdict={'family':'Arial','size':35,'weight':'bold'})

arrow1 = FancyArrow(
    0.395, 0.22,       # 起点 (x, y) in figure coords
    0, 0.015,        # dx, dy
    width=0.007,
    color='silver',
    head_width=0.015,
    head_length=0.015,
    transform=fig.transFigure
)
fig.patches.append(arrow1)

arrow2 = FancyArrow(
    0.395, 0.22,    
    0, -0.015,       
    width=0.007,
    color='silver',
    head_width=0.015,
    head_length=0.015,
    transform=fig.transFigure
)
fig.patches.append(arrow2)

arrow3 = FancyArrow(
    0.395, 0.75,       # 起点 (x, y) in figure coords
    0, 0.015,        # dx, dy
    width=0.007,
    color='silver',
    head_width=0.015,
    head_length=0.015,
    transform=fig.transFigure
)
fig.patches.append(arrow3)

arrow4 = FancyArrow(
    0.395, 0.75,    
    0, -0.015,       
    width=0.007,
    color='silver',
    head_width=0.015,
    head_length=0.015,
    transform=fig.transFigure
)
fig.patches.append(arrow4)

plt.subplots_adjust(top=0.92, bottom=0.06, left=0.04, right=0.98, hspace=0.5, wspace=0.3)
plt.savefig('../figures/FIG2.pdf',format='pdf')
plt.savefig('/Users/zhugchzo/Desktop/3paper_fig/FIG2.png',format='png',dpi=600)



