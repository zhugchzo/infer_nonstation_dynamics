import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

font = font_manager.FontProperties(family='Arial',weight='medium')

font_x = {'family':'Arial','weight':'medium','size': 20}
font_y = {'family':'Arial','weight':'medium','size': 20}
font_title = {'family':'DejaVu Sans','weight':'normal','size': 16, 'style': 'italic'}

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15

# df_mitochondria
df_mitochondria_AIC = pd.read_csv('../results/mitochondria/mitochondria_AIC.csv', header=None)

mitochondria_AIC = df_mitochondria_AIC.to_numpy()[1:,1:] # shape = (11,12)

mitochondria_AIC_filtered = mitochondria_AIC[mitochondria_AIC != 1e5]
max_value_mitochondria = np.max(mitochondria_AIC_filtered)
mitochondria_AIC[mitochondria_AIC == 1e5] = max_value_mitochondria

# df_UAV
df_UAV_AIC = pd.read_csv('../results/UAV/UAV_AIC.csv', header=None)

UAV_AIC = df_UAV_AIC.to_numpy()[1:,1:] # shape = (11,12)

UAV_AIC_filtered = UAV_AIC[UAV_AIC != 1e5]
max_value_UAV = np.max(UAV_AIC_filtered)
UAV_AIC[UAV_AIC == 1e5] = max_value_UAV

# df_fish
df_fish_AIC = pd.read_csv('../results/fish/fish_AIC.csv', header=None)

fish_AIC = df_fish_AIC.to_numpy()[1:,1:] # shape = (11,12)

fish_AIC_filtered = fish_AIC[fish_AIC != 1e5]
max_value_fish = np.max(fish_AIC_filtered)
fish_AIC[fish_AIC == 1e5] = max_value_fish

# draw
fig, axs = plt.subplots(1, 3, figsize=(15,4.5))

ax1, ax2, ax3 = axs[0], axs[1], axs[2]

x_labels = [r'$1\times10^{-5}$', r'$5\times10^{-5}$', r'$1\times10^{-4}$', r'$5\times10^{-4}$', r'$1\times10^{-3}$',
            r'$5\times10^{-3}$', r'$1\times10^{-2}$', r'$5\times10^{-2}$', r'$1\times10^{-1}$', r'$5\times10^{-1}$', '1', '5']
y_labels = [-20, -15, -10, -5, -1, 0, 1, 5, 10, 15, 20]

# ax1
heatmap = sns.heatmap(mitochondria_AIC, 
            cmap='PuBu',
            linewidths=0.005, 
            linecolor='silver',
            annot=False,
            square=True,
            ax=ax1,
            cbar_kws={'shrink': 0.65
            })

ax1.scatter(4+0.5, 6+0.5, color='blueviolet', s=150, marker=(5, 1))

heatmap.set_xticks(np.arange(len(x_labels)) + 0.7)
heatmap.set_yticks(np.arange(len(y_labels)) + 0.5)
heatmap.set_xticklabels(x_labels, rotation=90, ha='right', fontsize=12, fontproperties=font)
heatmap.set_yticklabels(y_labels, rotation=0, fontsize=12)

heatmap.tick_params(axis='both', length=0) 

cbar = heatmap.collections[0].colorbar
cbar.set_ticks([-1.25e5, -0.75e5])
cbar.set_ticklabels(['-1.25', '-0.75'])
cbar.ax.text(0.8, -1.55e5, r'$10^{5}$', ha='center', va='top', fontsize=10, fontproperties=font)
cbar.ax.text(3.2, -0.3e5, r'$\epsilon$AIC', ha='center', va='top', fontsize=14, fontproperties=font)
cbar.ax.tick_params(length=0)
for label in cbar.ax.get_yticklabels():
    label.set_fontproperties(font)
    label.set_fontsize(12)

ax1.set_title(r'Grid search: $\nu_1=1,\,\Delta \nu=1\times10^{-3}$',fontdict=font_title, pad=5)
ax1.text(-0.12, 1.16,'a',ha='left', transform=ax1.transAxes,fontdict={'family':'Arial','size':24,'weight':'bold'})
ax1.text(0.05, 1.14,'Cellular energy depletion',ha='left', transform=ax1.transAxes,color='royalblue',fontdict={'family':'DejaVu Sans','weight':'normal','size': 16, 'style': 'italic'})

# ax2
heatmap = sns.heatmap(UAV_AIC,
            cmap='PuBu',
            linewidths=0.005, 
            linecolor='silver',
            annot=False,
            square=True,
            ax=ax2,
            cbar_kws={'shrink': 0.65
            })

ax2.scatter(5+0.5, 6+0.5, color='blueviolet', s=150, marker=(5, 1))

heatmap.set_xticks(np.arange(len(x_labels)) + 0.7)
heatmap.set_yticks(np.arange(len(y_labels)) + 0.5)
heatmap.set_xticklabels(x_labels, rotation=90, ha='right', fontsize=12, fontproperties=font)
heatmap.set_yticklabels(y_labels, rotation=0, fontsize=12)

heatmap.tick_params(axis='both', length=0) 

cbar = heatmap.collections[0].colorbar
cbar.set_ticks([-1.5e5, -0.75e5])
cbar.set_ticklabels(['-1.5', '-0.75'])
cbar.ax.text(0.8, -1.82e5, r'$10^{5}$', ha='center', va='top', fontsize=10, fontproperties=font)
cbar.ax.text(3.2, -0.07e5, r'$\epsilon$AIC', ha='center', va='top', fontsize=14, fontproperties=font)
cbar.ax.tick_params(length=0)
for label in cbar.ax.get_yticklabels():
    label.set_fontproperties(font)
    label.set_fontsize(12)

ax2.set_title(r'Grid search: $\nu_1=1,\,\Delta \nu=5\times10^{-3}$',fontdict=font_title, pad=5)
ax2.text(-0.12, 1.16,'b',ha='left', transform=ax2.transAxes,fontdict={'family':'Arial','size':24,'weight':'bold'})
ax2.text(0.1, 1.14,'UAV obstacle avoidance',ha='left', transform=ax2.transAxes,color='royalblue',fontdict={'family':'DejaVu Sans','weight':'normal','size': 16, 'style': 'italic'})

# ax3
heatmap = sns.heatmap(fish_AIC,
            cmap='PuBu',
            linewidths=0.005, 
            linecolor='silver',
            annot=False,
            square=True,
            ax=ax3,
            cbar_kws={'shrink': 0.65
            })

ax3.scatter(10+0.5, 5+0.5, color='blueviolet', s=150, marker=(5, 1))

heatmap.set_xticks(np.arange(len(x_labels)) + 0.7)
heatmap.set_yticks(np.arange(len(y_labels)) + 0.5)
heatmap.set_xticklabels(x_labels, rotation=90, ha='right', fontsize=12, fontproperties=font)
heatmap.set_yticklabels(y_labels, rotation=0, fontsize=12)

heatmap.tick_params(axis='both', length=0) 

cbar = heatmap.collections[0].colorbar
cbar.set_ticks([-500, 0, 500])
cbar.set_ticklabels(['-5', '0', '5'])
cbar.ax.text(0.8, -750, r'$10^{2}$', ha='center', va='top', fontsize=10, fontproperties=font)
cbar.ax.text(3.2, 820, r'$\epsilon$AIC', ha='center', va='top', fontsize=14, fontproperties=font)
cbar.ax.tick_params(length=0)
for label in cbar.ax.get_yticklabels():
    label.set_fontproperties(font)
    label.set_fontsize(12)

ax3.set_title(r'Grid search: $\nu_1=0,\,\Delta \nu=1$',fontdict=font_title, pad=5)
ax3.text(-0.12, 1.16,'c',ha='left', transform=ax3.transAxes,fontdict={'family':'Arial','size':24,'weight':'bold'})
ax3.text(0.2, 1.14,'Fish community',ha='left', transform=ax3.transAxes,color='royalblue',fontdict={'family':'DejaVu Sans','weight':'normal','size': 16, 'style': 'italic'})

plt.subplots_adjust(top=0.99, bottom=0.01, left=0.04, right=0.98, hspace=0.25, wspace=0.3)
plt.savefig('../figures/SFIG1.pdf',format='pdf')
plt.savefig('/Users/zhugchzo/Desktop/3paper_fig/SFIG1.png',format='png',dpi=600)