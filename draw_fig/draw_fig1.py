import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib import font_manager
from matplotlib.patches import FancyArrow

font = font_manager.FontProperties(family='Arial',weight='medium')

font_x = {'family':'Arial','weight':'medium','size': 20}
font_y = {'family':'Arial','weight':'medium','size': 20}
font_title = {'family':'DejaVu Sans','weight':'normal','size': 16, 'style': 'italic'}

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15

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

# df_Koscillators
df_Koscillators_AIC = pd.read_csv('../results/Koscillators/Koscillators_AIC.csv', header=None)
df_Koscillators_pred = pd.read_csv('../results/Koscillators/Koscillators_pred.csv')
df_Koscillators_data = pd.read_csv('../Koscillators/Koscillators_data.csv')

col_traj = ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10']
col_pred = ['predx_0', 'predx_1', 'predx_2', 'predx_3', 'predx_4', 'predx_5', 'predx_6', 'predx_7', 'predx_8', 'predx_9', 'predx_10']

Koscillators_AIC = df_Koscillators_AIC.to_numpy()[1:,1:]

Koscillators_AIC_filtered = Koscillators_AIC[Koscillators_AIC != 1e5]
max_value_Koscillators = np.max(Koscillators_AIC_filtered)
Koscillators_AIC[Koscillators_AIC == 1e5] = max_value_Koscillators

train_length_Koscillators = len(df_Koscillators_data) - len(df_Koscillators_pred) - 1

span = 10

t_Koscillators = np.arange(span,len(df_Koscillators_data))[::5]
t_Koscillators_pred = np.arange(span+train_length_Koscillators,train_length_Koscillators+len(df_Koscillators_pred))[::5]
t_Koscillators_train = np.arange(span,len(df_Koscillators_data))[:train_length_Koscillators-span][::5]

Koscillators_pred = df_Koscillators_pred[col_pred].to_numpy()
Koscillators_data = df_Koscillators_data[col_traj].to_numpy()

row1 = Koscillators_pred.shape[0] - span
col1 = Koscillators_pred.shape[1]

delta_predx = np.zeros((row1,col1))

for j in range(col1):
    for i in range(row1):
        delta_predx[i, j] = abs(Koscillators_pred[i + span, j] - Koscillators_pred[i, j])

sum_delta_predx = np.sum(delta_predx,axis=1)[::5]

row2 = Koscillators_data.shape[0] - span
col2 = Koscillators_data.shape[1]

delta_datax = np.zeros((row2,col2))

for j in range(col2):
    for i in range(row2):
        delta_datax[i, j] = abs(Koscillators_data[i + span, j] - Koscillators_data[i, j])

sum_delta_datax = np.sum(delta_datax,axis=1)
sum_delta_datax_train = sum_delta_datax[:train_length_Koscillators-span][::5]
sum_delta_datax = sum_delta_datax[::5]

# draw
fig, axs = plt.subplots(2, 3, figsize=(15,9))

ax1, ax2, ax3 = axs[0, 0], axs[0, 1], axs[0, 2]
ax4, ax5, ax6 = axs[1, 0], axs[1, 1], axs[1, 2]

x_labels = [r'$1\times10^{-5}$', r'$5\times10^{-5}$', r'$1\times10^{-4}$', r'$5\times10^{-4}$', r'$1\times10^{-3}$',
            r'$5\times10^{-3}$', r'$1\times10^{-2}$', r'$5\times10^{-2}$', r'$1\times10^{-1}$', r'$5\times10^{-1}$', '1', '5']
y_labels = [-20, -15, -10, -5, -1, 0, 1, 5, 10, 15, 20]

# ax1
heatmap = sns.heatmap(cusp_AIC, 
            cmap='PuBu',
            linewidths=0.005, 
            linecolor='silver',
            annot=False,
            square=True,
            ax=ax1,
            cbar_kws={'shrink': 0.75
            })

# ax1.scatter(5+0.5, 4+0.5, color='blueviolet', s=150, marker=(5, 1))

heatmap.set_xticks(np.arange(len(x_labels)) + 0.7)
heatmap.set_yticks(np.arange(len(y_labels)) + 0.5)
heatmap.set_xticklabels(x_labels, rotation=90, ha='right', fontsize=12, fontproperties=font)
heatmap.set_yticklabels(y_labels, rotation=0, fontsize=12)

heatmap.tick_params(axis='both', length=0) 

cbar = heatmap.collections[0].colorbar
cbar.set_ticks([-2e6, -1e6])
cbar.set_ticklabels(['-2', '-1'])
cbar.ax.text(0.8, -2.5e6, r'$10^{6}$', ha='center', va='top', fontsize=10, fontproperties=font)
cbar.ax.text(3.2, -0.25e6, r'$\epsilon$AIC', ha='center', va='top', fontsize=14, fontproperties=font)
cbar.ax.tick_params(length=0)
for label in cbar.ax.get_yticklabels():
    label.set_fontproperties(font)
    label.set_fontsize(12)

ax1.set_title(r'Grid search: $\nu_1=-1,\,\Delta \nu=5\times10^{-3}$',fontdict=font_title, pad=5)
ax1.text(-0.12, 1.125,'a',ha='left', transform=ax1.transAxes,fontdict={'family':'Arial','size':24,'weight':'bold'})
ax1.text(0.22, 1.125,'Cusp bifurcation',ha='left', transform=ax1.transAxes,color='royalblue',fontdict={'family':'DejaVu Sans','weight':'normal','size': 16, 'style': 'italic'})

# ax2
ax2.axis('off')

ax2.text(0.35,0.8,'True:',fontdict={'family':'Arial','size':18,'weight':'medium'},color='royalblue')
ax2.text(0.2,0.7,r'$\dot{x}=\phi_1+\phi_2x-x^3$',fontdict={'family':'Arial','size':18,'weight':'medium'})
ax2.text(0.2,0.6,r'$(\phi_1,\phi_2):(1,4)\rightarrow(3,3)$',fontdict={'family':'Arial','size':15,'weight':'medium'})

ax2.text(0.35,0.35,'Infer:',fontdict={'family':'Arial','size':18,'weight':'medium'},color='royalblue')
ax2.text(-0.1,0.25,r'$\dot{x}=(1.4+0.4\nu)+(3.8-0.2\nu)x-x^3$',fontdict={'family':'Arial','size':18,'weight':'medium'})
ax2.text(0.3,0.15,r'$\nu:-1\rightarrow4$',fontdict={'family':'Arial','size':15,'weight':'medium'})

ax2.text(-0.12, 0.99,'b',ha='left', transform=ax2.transAxes,fontdict={'family':'Arial','size':24,'weight':'bold'})

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
ax3.set_ylabel(r'$x$',font_y,labelpad=-10)
ax3.set_xlim(-50,1050)
ax3.set_xticks([0,500,1000])
ax3.set_xticklabels(['0','500','1000'])
ax3.set_ylim(-2.3,2.4)
ax3.set_yticks([-2,2])
ax3.set_yticklabels(['-2','2'])

ax3.tick_params(direction='in')

ax3.text(-0.12, 0.99,'c',ha='left', transform=ax3.transAxes,fontdict={'family':'Arial','size':24,'weight':'bold'})

# ax4
heatmap = sns.heatmap(Koscillators_AIC, 
            cmap='PuBu',
            linewidths=0.005, 
            linecolor='silver',
            annot=False,
            square=True,
            ax=ax4,
            cbar_kws={'shrink': 0.75
            })

heatmap.set_xticks(np.arange(len(x_labels)) + 0.7)
heatmap.set_yticks(np.arange(len(y_labels)) + 0.5)
heatmap.set_xticklabels(x_labels, rotation=90, ha='right', fontsize=12, fontproperties=font)
heatmap.set_yticklabels(y_labels, rotation=0, fontsize=12)

heatmap.tick_params(axis='both', length=0) 

cbar = heatmap.collections[0].colorbar
cbar.set_ticks([-6e23, -3e23])
cbar.set_ticklabels(['-6', '-3'])
cbar.ax.text(1, -8.5e23, r'$10^{23}$', ha='center', va='top', fontsize=10, fontproperties=font)
cbar.ax.text(3.2, -0.3e23, r'$\epsilon$AIC', ha='center', va='top', fontsize=14, fontproperties=font)
cbar.ax.tick_params(length=0)
for label in cbar.ax.get_yticklabels():
    label.set_fontproperties(font)
    label.set_fontsize(12)

ax4.set_title(r'Grid search: $\nu_1=-1,\,\Delta \nu=5\times10^{-2}$',fontdict=font_title, pad=5)
ax4.text(-0.12, 1.125,'d',ha='left', transform=ax4.transAxes,fontdict={'family':'Arial','size':24,'weight':'bold'})
ax4.text(0.12, 1.125,'Kuramoto oscillators',ha='left', transform=ax4.transAxes,color='royalblue',fontdict={'family':'DejaVu Sans','weight':'normal','size': 16, 'style': 'italic'})

# ax5
ax5.axis('off')

ax5.text(0.35,0.85,'True:',fontdict={'family':'Arial','size':18,'weight':'medium'},color='royalblue')
ax5.text(0.1,0.7,r'$\dot{\theta_i}=\omega_i+\sigma\sum_{j=1}^{N}A_{ij}\sin(\theta_j-\theta_i)$',fontdict={'family':'Arial','size':18,'weight':'medium'})
ax5.text(0.2,0.52,r'$\sum_{i=1}^{N}\omega_i=0,\,\sigma:0\rightarrow1$',fontdict={'family':'Arial','size':15,'weight':'medium'})

ax5.text(0.35,0.3,'Infer:',fontdict={'family':'Arial','size':18,'weight':'medium'},color='royalblue')
ax5.text(-0.1,0.2,r'$\dot{\theta_i}=(0.0127+0.0202\nu)\sum_{j=1}^{N}A_{ij}\sin(\theta_j-\theta_i)$',fontdict={'family':'Arial','size':18,'weight':'medium'})
ax5.text(0.3,0.05,r'$\nu:-1\rightarrow49$',fontdict={'family':'Arial','size':15,'weight':'medium'})

ax5.text(-0.12, 0.99,'e',ha='left', transform=ax5.transAxes,fontdict={'family':'Arial','size':24,'weight':'bold'})

# ax6
ax6.plot(t_Koscillators,sum_delta_datax,c='black',zorder=2)
ax6.scatter(t_Koscillators,sum_delta_datax,s=10,c='black',marker='o',zorder=2)
ax6.scatter(t_Koscillators_pred,sum_delta_predx,s=50,marker='o',facecolors='none',edgecolors='crimson',zorder=3)

ax6.fill_between(t_Koscillators_train,sum_delta_datax_train-0.04,sum_delta_datax_train+0.04,color='silver',alpha=0.9,linewidth=0,zorder=1)

legend_state = mlines.Line2D([], [], color='black', marker='o', markersize=3, linestyle='-', markeredgewidth=1.5)
legend_pstate = mlines.Line2D([], [], markerfacecolor='none',color='crimson', marker='o', markersize=5, linestyle='None', markeredgewidth=1.5)
legend_fill = mpatches.Patch(color='silver', alpha=0.9, linewidth=0)

ax6.legend(handles=[legend_state,legend_pstate,legend_fill],labels=['True','Prediction','Training data'],loc='center', frameon=False, bbox_to_anchor=(0.3, 0.8), markerscale=2.5,prop={'size':16})

ax6.set_xlabel('Timepoints',font_x)
ax6.set_ylabel(r'$\sum\left|\Delta \theta_i\right|$',font_y,labelpad=-20)
ax6.set_xlim(-49.5,1049.5)
ax6.set_xticks([10,500,1000])
ax6.set_xticklabels(['10','500','1000'])
ax6.set_ylim(-0.1,1.3)
ax6.set_yticks([0,1.2])
ax6.set_yticklabels(['0','1.2'])

ax6.tick_params(direction='in')

ax6.text(-0.12, 0.99,'f',ha='left', transform=ax6.transAxes,fontdict={'family':'Arial','size':24,'weight':'bold'})


arrow1 = FancyArrow(0.3, 0.75, 0.015, 0, width=0.01, color='silver', head_width=0.02, head_length=0.01,transform=fig.transFigure)
fig.patches.append(arrow1)

arrow2 = FancyArrow(0.64, 0.75, 0.015, 0, width=0.01, color='silver', head_width=0.02, head_length=0.01,transform=fig.transFigure)
fig.patches.append(arrow2)

arrow3 = FancyArrow(0.3, 0.25, 0.015, 0, width=0.01, color='silver', head_width=0.02, head_length=0.01,transform=fig.transFigure)
fig.patches.append(arrow3)

arrow4 = FancyArrow(0.64, 0.25, 0.015, 0, width=0.01, color='silver', head_width=0.02, head_length=0.01,transform=fig.transFigure)
fig.patches.append(arrow4)

plt.subplots_adjust(top=0.965, bottom=0.065, left=0.04, right=0.98, hspace=0.25, wspace=0.3)
plt.savefig('../figures/FIG1.pdf',format='pdf')
plt.savefig('/Users/zhugchzo/Desktop/3paper_fig/FIG1.png',format='png',dpi=600)