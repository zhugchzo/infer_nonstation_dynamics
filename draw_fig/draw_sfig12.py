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
df_cusp_curve_AIC = pd.read_csv('../results/cusp/cusp_curve_AIC.csv', header=None)
df_cusp_curve_pred = pd.read_csv('../results/cusp/cusp_curve_pred.csv')
df_cusp_curve_pred_sum = pd.read_csv('../results/cusp/cusp_curve_traj_pred.csv')
df_cusp_curve_data = pd.read_csv('../cusp/cusp_curve_data.csv')

cusp_curve_AIC = df_cusp_curve_AIC.to_numpy()[1:,1:] # shape = (11,12)

train_length_cusp = len(df_cusp_curve_data) - len(df_cusp_curve_pred) - 1

t_cusp = np.arange(len(df_cusp_curve_data))#[::5]
t_cusp_curve_train = np.arange(len(df_cusp_curve_data))[:train_length_cusp]#[::5]

x_cusp = df_cusp_curve_data['x']#[::5]
trainx_cusp = df_cusp_curve_data['x'][:train_length_cusp]#[::5]

trajx_cusp = df_cusp_curve_pred_sum['traj'].to_numpy()
predx_cusp = df_cusp_curve_pred_sum['pred'].to_numpy()

cusp_curve_AIC_filtered = cusp_curve_AIC[cusp_curve_AIC != 1e5]
max_value_cusp = np.max(cusp_curve_AIC_filtered)
cusp_curve_AIC[cusp_curve_AIC == 1e5] = max_value_cusp

# draw
fig, axs = plt.subplots(1, 4, figsize=(22,5.5))

ax1, ax2, ax3, ax4 = axs[0], axs[1], axs[2], axs[3]

x_labels = [r'$1\times10^{-5}$', r'$5\times10^{-5}$', r'$1\times10^{-4}$', r'$5\times10^{-4}$', r'$1\times10^{-3}$',
            r'$5\times10^{-3}$', r'$1\times10^{-2}$', r'$5\times10^{-2}$', r'$1\times10^{-1}$', r'$5\times10^{-1}$', '1', '5']
y_labels = ['-20', '-15', '-10', '-5', '-1', '0', '1', '5', '10', '15', '20']

# ax1
ax1.plot(t_cusp,x_cusp,c='black',linewidth=5,zorder=2)
ax1.plot(t_cusp_curve_train,trainx_cusp,c='silver',linewidth=5,zorder=2)

legend_state = mlines.Line2D([], [], color='black', marker=None, linewidth=5, linestyle='-')
legend_train = mlines.Line2D([], [], color='silver', marker=None, linewidth=5, linestyle='-')

ax1.legend(handles=[legend_state,legend_train],labels=['State','Training data'],loc='center', frameon=False, bbox_to_anchor=(0.35, 0.8), markerscale=2.5,prop={'size':16})

ax1.set_xlabel('Timepoints',font_x)
ax1.set_ylabel(r'$\mathbf{x}$',font_y,labelpad=10)
ax1.set_xlim(-50,1050)
ax1.set_xticks([0,500,1000])
ax1.set_xticklabels(['0','500','1000'])
ax1.set_ylim(-2.2,-0.8)
ax1.set_yticks([-2,-1])
ax1.set_yticklabels(['-2','-1'])

ax1.tick_params(direction='in')
ax1.set_title('Cusp bifurcation',fontdict=font_title, pad=10)

ax1.text(-0.12, 1.1,'a',ha='left', transform=ax1.transAxes,fontdict={'family':'Arial','size':35,'weight':'bold'})

# ax2
ax2.axis('off')

ax2.text(0,1.05,'True cusp bifurcation dynamics :',fontdict={'family':'Arial','size':20,'weight':'medium'},color='royalblue')
ax2.text(0.2,0.9,r'$\dot{x}=\phi_1+\phi_2x-x^3$',fontdict={'family':'Arial','size':18,'weight':'medium'})
ax2.text(-0.1,0.75,r'$\{(\phi_1,\phi_2):4\phi_1^2-15\phi_1+2\phi_2+3=0\}$',fontdict={'family':'Arial','size':18,'weight':'medium'})
ax2.text(0.15,0.6,r'$(\phi_1,\phi_2):(1,4)\rightarrow(3,3)$',fontdict={'family':'Arial','size':18,'weight':'medium'})

ax2.text(0.2,0.25,'Inferred equation :',fontdict={'family':'Arial','size':20,'weight':'medium'},color='royalblue')
ax2.text(-0.2,0.1,r'$\dot{x}=(1.4+0.4\nu)+(5.08+0.76\nu-0.32\nu^2)x-x^3$',fontdict={'family':'Arial','size':18,'weight':'medium'})
ax2.text(0.3,-0.05,r'$\nu:-1\rightarrow4$',fontdict={'family':'Arial','size':18,'weight':'medium'})

ax2.text(-0.2, 1.1,'b',ha='left', transform=ax2.transAxes,fontdict={'family':'Arial','size':35,'weight':'bold'})

# ax3
heatmap = sns.heatmap(cusp_curve_AIC, 
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
cbar.set_ticks([-6e10, -3e10])
cbar.set_ticklabels(['-6', '-3'])
cbar.ax.text(0.8, -9.5e10, r'$10^{10}$', ha='center', va='top', fontsize=15, fontproperties=font)
cbar.ax.text(3.5, -0.5e10, r'$\epsilon$AIC', ha='center', va='top', fontsize=18, fontproperties=font_AIC)
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

cusp_curve_r2 = r2_score(trajx_cusp, predx_cusp)

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
ax4.set_xticks([-2,-1.5,-1,-0.5])
ax4.set_xticklabels(['-2','-1.5','-1','-0.5'])
ax4.set_yticks([-2,-1.5,-1,-0.5])
ax4.set_yticklabels(['-2','-1.5','-1','-0.5'])
ax4.set_xlabel('True',font_x)
ax4.set_ylabel('Prediction',font_x,labelpad=10)
ax4.tick_params(direction='in')
ax4.set_title(r'$\mathbf{x}$',fontdict=font_title_x, pad=10)
ax4.text(0.1, 0.8, f'$R^2 = {cusp_curve_r2:.2f}$', transform=ax4.transAxes, fontsize=24, color='black')
ax4.spines['right'].set_visible(False)
ax4.spines['top'].set_visible(False)

ax4.text(-0.12, 1.1,'c',ha='left', transform=ax4.transAxes,fontdict={'family':'Arial','size':35,'weight':'bold'})

arrow1 = FancyArrow(
    0.38, 0.45,       # 起点 (x, y) in figure coords
    0, 0.03,        # dx, dy
    width=0.007,
    color='silver',
    head_width=0.015,
    head_length=0.015,
    transform=fig.transFigure
)
fig.patches.append(arrow1)

arrow2 = FancyArrow(
    0.38, 0.45,    
    0, -0.03,       
    width=0.007,
    color='silver',
    head_width=0.015,
    head_length=0.015,
    transform=fig.transFigure
)
fig.patches.append(arrow2)

plt.subplots_adjust(top=0.84, bottom=0.12, left=0.04, right=0.99, wspace=0.3)
plt.savefig('../figures/SFIG12.pdf',format='pdf')
plt.savefig('/Users/zhugchzo/Desktop/3paper_fig/SFIG12.png',format='png',dpi=600)