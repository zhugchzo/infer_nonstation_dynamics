import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib import font_manager
from matplotlib.patches import FancyArrow

font = font_manager.FontProperties(family='Arial',weight='medium')
font_AIC = font_manager.FontProperties(family='Arial',weight='bold')

font_x = {'family':'Arial','weight':'bold','size': 18}
font_y = {'family':'Arial','weight':'bold','size': 24}
font_title = {'family':'Arial','weight':'bold','size': 20}

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18

# df_cusp
df_cusp_AIC = pd.read_csv('../results/cusp/cusp_AIC.csv', header=None)
df_cusp_pred = pd.read_csv('../results/cusp/cusp_pred.csv')
df_cusp_data = pd.read_csv('../cusp/cusp_data.csv')

cusp_AIC = df_cusp_AIC.to_numpy()[1:,1:] # shape = (11,12)

train_length_cusp = len(df_cusp_data) - len(df_cusp_pred) - 1

t_cusp = np.arange(len(df_cusp_data))[::5]
t_cusp_pred = np.arange(train_length_cusp,train_length_cusp+len(df_cusp_pred))[::5]
t_cusp_train = np.arange(len(df_cusp_data))[:train_length_cusp][::5]

x_cusp = df_cusp_data['x'][::5]
trainx_cusp = df_cusp_data['x'][:train_length_cusp][::5]
predx_cusp = df_cusp_pred['pred'][::5]

cusp_AIC_filtered = cusp_AIC[cusp_AIC != 1e5]
max_value_cusp = np.max(cusp_AIC_filtered)
cusp_AIC[cusp_AIC == 1e5] = max_value_cusp

# draw
fig, axs = plt.subplots(1, 3, figsize=(16,5.5))

ax1, ax2, ax3 = axs[0], axs[1], axs[2]

x_labels = [r'$1\times10^{-5}$', r'$5\times10^{-5}$', r'$1\times10^{-4}$', r'$5\times10^{-4}$', r'$1\times10^{-3}$',
            r'$5\times10^{-3}$', r'$1\times10^{-2}$', r'$5\times10^{-2}$', r'$1\times10^{-1}$', r'$5\times10^{-1}$', '1', '5']
y_labels = ['-20', '-15', '-10', '-5', '-1', '0', '1', '5', '10', '15', '20']

# ax1
ax1.axis('off')

ax1.text(0.1,1,'True cusp bifurcation dynamics :',fontdict={'family':'Arial','size':20,'weight':'medium'},color='royalblue')
ax1.text(0.3,0.8,r'$\dot{x}=\phi_1+\phi_2x-x^3$',fontdict={'family':'Arial','size':20,'weight':'medium'})
ax1.text(0.2,0.65,r'$(\phi_1,\phi_2):(1,4)\rightarrow(3,3)$',fontdict={'family':'Arial','size':20,'weight':'medium'})

ax1.text(0.3,0.3,'Surrogate model :',fontdict={'family':'Arial','size':20,'weight':'medium'},color='royalblue')
ax1.text(-0.05,0.1,r'$\dot{x}=(1.4+0.4\nu)+(3.8-0.2\nu)x-x^3$',fontdict={'family':'Arial','size':20,'weight':'medium'})
ax1.text(0.35,-0.05,r'$\nu:-1\rightarrow4$',fontdict={'family':'Arial','size':20,'weight':'medium'})

ax1.text(-0.12, 1.02,'b',ha='left', transform=ax1.transAxes,fontdict={'family':'Arial','size':40,'weight':'bold'})

# ax2
heatmap = sns.heatmap(cusp_AIC, 
            cmap='PuBu',
            linewidths=0.005, 
            linecolor='silver',
            annot=False,
            square=True,
            ax=ax2,
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

for label in ax2.get_xticklabels():
    label.set_fontproperties(font)
    label.set_fontsize(12)

for label in ax2.get_yticklabels():
    label.set_fontproperties(font)
    label.set_fontsize(15)

ax2.set_title('Grid search result :'+'\n'+r'$\nu_1=-1,\,\Delta \nu=5\times10^{-3}$',fontdict=font_title, pad=10)

# ax3
ax3.plot(t_cusp,x_cusp,c='black',zorder=2)
ax3.scatter(t_cusp,x_cusp,s=10,c='black',marker='o',zorder=2)
ax3.scatter(t_cusp_pred,predx_cusp,s=50,marker='o',facecolors='none',edgecolors='crimson',zorder=3)

ax3.fill_between(t_cusp_train,trainx_cusp-0.1,trainx_cusp+0.1,color='silver',alpha=0.9,linewidth=0,zorder=1)

legend_state = mlines.Line2D([], [], color='black', marker='o', markersize=3, linestyle='-', markeredgewidth=1.5)
legend_pstate = mlines.Line2D([], [], markerfacecolor='none',color='crimson', marker='o', markersize=5, linestyle='None', markeredgewidth=1.5)
legend_fill = mpatches.Patch(color='silver', alpha=0.9, linewidth=0)

ax3.legend(handles=[legend_state,legend_pstate,legend_fill],labels=['True','Prediction','Training data'],loc='center', frameon=False, bbox_to_anchor=(0.35, 0.8), markerscale=2.5,prop={'size':16})

ax3.set_xlabel('Timepoints',font_x)
ax3.set_ylabel(r'$\mathbf{x}$',font_y,labelpad=10)
ax3.set_xlim(-50,1050)
ax3.set_xticks([0,500,1000])
ax3.set_xticklabels(['0','500','1000'])
ax3.set_ylim(-2.3,2.4)
ax3.set_yticks([-2,2])
ax3.set_yticklabels(['-2','2'])

ax3.tick_params(direction='in')
ax3.set_title('Dynamics prediction',fontdict=font_title, pad=10)

arrow1 = FancyArrow(
    0.175, 0.5,       # 起点 (x, y) in figure coords
    0, 0.03,        # dx, dy
    width=0.007,
    color='silver',
    head_width=0.015,
    head_length=0.025,
    transform=fig.transFigure
)
fig.patches.append(arrow1)

arrow2 = FancyArrow(
    0.175, 0.5,    
    0, -0.03,       
    width=0.007,
    color='silver',
    head_width=0.015,
    head_length=0.025,
    transform=fig.transFigure
)
fig.patches.append(arrow2)

plt.subplots_adjust(top=0.88, bottom=0.15, left=0.04, right=0.98, wspace=0.3)
plt.savefig('../figures/FIG1.2.pdf',format='pdf')
plt.savefig('/Users/zhugchzo/Desktop/3paper_fig/FIG1.2.png',format='png',dpi=600)