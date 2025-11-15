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

# df_chick_220
df_chick_220_AIC = pd.read_csv('../results/chick/chick_220_AIC.csv', header=None)

chick_220_AIC = df_chick_220_AIC.to_numpy()[1:,1:] # shape = (11,12)

chick_220_AIC_filtered = chick_220_AIC[chick_220_AIC != 1e5]
max_value_chick_220 = np.max(chick_220_AIC_filtered)
chick_220_AIC[chick_220_AIC == 1e5] = max_value_chick_220

# df_chick_230
df_chick_230_AIC = pd.read_csv('../results/chick/chick_230_AIC.csv', header=None)

chick_230_AIC = df_chick_230_AIC.to_numpy()[1:,1:] # shape = (11,12)

chick_230_AIC_filtered = chick_230_AIC[chick_230_AIC != 1e5]
max_value_chick_230 = np.max(chick_230_AIC_filtered)
chick_230_AIC[chick_230_AIC == 1e5] = max_value_chick_230

# df_chick_335
df_chick_335_AIC = pd.read_csv('../results/chick/chick_335_AIC.csv', header=None)

chick_335_AIC = df_chick_335_AIC.to_numpy()[1:,1:] # shape = (11,12)

chick_335_AIC_filtered = chick_335_AIC[chick_335_AIC != 1e5]
max_value_chick_335 = np.max(chick_335_AIC_filtered)
chick_335_AIC[chick_335_AIC == 1e5] = max_value_chick_335

# df_fish
df_fish_AIC = pd.read_csv('../results/fish/fish_AIC.csv', header=None)

fish_AIC = df_fish_AIC.to_numpy()[1:,1:] # shape = (11,12)

fish_AIC_filtered = fish_AIC[fish_AIC != 1e5]
max_value_fish = np.max(fish_AIC_filtered)
fish_AIC[fish_AIC == 1e5] = max_value_fish

# draw
fig, axs = plt.subplots(2, 3, figsize=(15,9))

ax1, ax2, ax3 = axs[0,0], axs[0,1], axs[0,2]
ax4, ax5, ax6 = axs[1,0], axs[1,1], axs[1,2]

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
ax1.text(-0.12, 1.16,'a',ha='left', transform=ax1.transAxes,fontdict={'family':'Arial','size':30,'weight':'bold'})
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
ax2.text(-0.12, 1.16,'b',ha='left', transform=ax2.transAxes,fontdict={'family':'Arial','size':30,'weight':'bold'})
ax2.text(0.1, 1.14,'UAV autonomous flight',ha='left', transform=ax2.transAxes,color='royalblue',fontdict={'family':'DejaVu Sans','weight':'normal','size': 16, 'style': 'italic'})

# ax3
heatmap = sns.heatmap(chick_220_AIC,
            cmap='PuBu',
            linewidths=0.005, 
            linecolor='silver',
            annot=False,
            square=True,
            ax=ax3,
            cbar_kws={'shrink': 0.65
            })

ax3.scatter(5+0.5, 5+0.5, color='blueviolet', s=150, marker=(5, 1))

heatmap.set_xticks(np.arange(len(x_labels)) + 0.7)
heatmap.set_yticks(np.arange(len(y_labels)) + 0.5)
heatmap.set_xticklabels(x_labels, rotation=90, ha='right', fontsize=12, fontproperties=font)
heatmap.set_yticklabels(y_labels, rotation=0, fontsize=12)

heatmap.tick_params(axis='both', length=0) 

cbar = heatmap.collections[0].colorbar
cbar.set_ticks([-3e14, -1.5e14])
cbar.set_ticklabels(['-3', '-1.5'])
cbar.ax.text(0.8, -4.5e14, r'$10^{14}$', ha='center', va='top', fontsize=10, fontproperties=font)
cbar.ax.text(3.2, -0.2e14, r'$\epsilon$AIC', ha='center', va='top', fontsize=14, fontproperties=font)
cbar.ax.tick_params(length=0)
for label in cbar.ax.get_yticklabels():
    label.set_fontproperties(font)
    label.set_fontsize(12)

ax3.set_title(r'Grid search: $\nu_1=0,\,\Delta \nu=5\times10^{-3}$',fontdict=font_title, pad=5)
ax3.text(-0.12, 1.16,'c',ha='left', transform=ax3.transAxes,fontdict={'family':'Arial','size':30,'weight':'bold'})
ax3.text(0.1, 1.14,'Beating chick-heart (I)',ha='left', transform=ax3.transAxes,color='royalblue',fontdict={'family':'DejaVu Sans','weight':'normal','size': 16, 'style': 'italic'})

# ax4
heatmap = sns.heatmap(chick_230_AIC,
            cmap='PuBu',
            linewidths=0.005, 
            linecolor='silver',
            annot=False,
            square=True,
            ax=ax4,
            cbar_kws={'shrink': 0.65
            })

ax4.scatter(5+0.5, 4+0.5, color='blueviolet', s=150, marker=(5, 1))

heatmap.set_xticks(np.arange(len(x_labels)) + 0.7)
heatmap.set_yticks(np.arange(len(y_labels)) + 0.5)
heatmap.set_xticklabels(x_labels, rotation=90, ha='right', fontsize=12, fontproperties=font)
heatmap.set_yticklabels(y_labels, rotation=0, fontsize=12)

heatmap.tick_params(axis='both', length=0) 

cbar = heatmap.collections[0].colorbar
cbar.set_ticks([-2e16, -1e16])
cbar.set_ticklabels(['-2', '-1'])
cbar.ax.text(0.8, -3e16, r'$10^{16}$', ha='center', va='top', fontsize=10, fontproperties=font)
cbar.ax.text(3.2, -0.1e16, r'$\epsilon$AIC', ha='center', va='top', fontsize=14, fontproperties=font)
cbar.ax.tick_params(length=0)
for label in cbar.ax.get_yticklabels():
    label.set_fontproperties(font)
    label.set_fontsize(12)

ax4.set_title(r'Grid search: $\nu_1=-1,\,\Delta \nu=5\times10^{-3}$',fontdict=font_title, pad=5)
ax4.text(-0.12, 1.16,'d',ha='left', transform=ax4.transAxes,fontdict={'family':'Arial','size':30,'weight':'bold'})
ax4.text(0.1, 1.14,'Beating chick-heart (II)',ha='left', transform=ax4.transAxes,color='royalblue',fontdict={'family':'DejaVu Sans','weight':'normal','size': 16, 'style': 'italic'})

# ax5
heatmap = sns.heatmap(chick_335_AIC,
            cmap='PuBu',
            linewidths=0.005, 
            linecolor='silver',
            annot=False,
            square=True,
            ax=ax5,
            cbar_kws={'shrink': 0.65
            })

ax5.scatter(4+0.5, 4+0.5, color='blueviolet', s=150, marker=(5, 1))

heatmap.set_xticks(np.arange(len(x_labels)) + 0.7)
heatmap.set_yticks(np.arange(len(y_labels)) + 0.5)
heatmap.set_xticklabels(x_labels, rotation=90, ha='right', fontsize=12, fontproperties=font)
heatmap.set_yticklabels(y_labels, rotation=0, fontsize=12)

heatmap.tick_params(axis='both', length=0) 

cbar = heatmap.collections[0].colorbar
cbar.set_ticks([-4e15, -2e15])
cbar.set_ticklabels(['-4', '-2'])
cbar.ax.text(0.8, -5.8e15, r'$10^{15}$', ha='center', va='top', fontsize=10, fontproperties=font)
cbar.ax.text(3.2, -0.2e15, r'$\epsilon$AIC', ha='center', va='top', fontsize=14, fontproperties=font)
cbar.ax.tick_params(length=0)
for label in cbar.ax.get_yticklabels():
    label.set_fontproperties(font)
    label.set_fontsize(12)

ax5.set_title(r'Grid search: $\nu_1=-1,\,\Delta \nu=1\times10^{-3}$',fontdict=font_title, pad=5)
ax5.text(-0.12, 1.16,'e',ha='left', transform=ax5.transAxes,fontdict={'family':'Arial','size':30,'weight':'bold'})
ax5.text(0.1, 1.14,'Beating chick-heart (III)',ha='left', transform=ax5.transAxes,color='royalblue',fontdict={'family':'DejaVu Sans','weight':'normal','size': 16, 'style': 'italic'})

# ax6
heatmap = sns.heatmap(fish_AIC,
            cmap='PuBu',
            linewidths=0.005, 
            linecolor='silver',
            annot=False,
            square=True,
            ax=ax6,
            cbar_kws={'shrink': 0.65
            })

ax6.scatter(10+0.5, 5+0.5, color='blueviolet', s=150, marker=(5, 1))

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

ax6.set_title(r'Grid search: $\nu_1=0,\,\Delta \nu=1$',fontdict=font_title, pad=5)
ax6.text(-0.12, 1.16,'f',ha='left', transform=ax6.transAxes,fontdict={'family':'Arial','size':30,'weight':'bold'})
ax6.text(0.2, 1.14,'Fish community',ha='left', transform=ax6.transAxes,color='royalblue',fontdict={'family':'DejaVu Sans','weight':'normal','size': 16, 'style': 'italic'})

plt.subplots_adjust(top=0.96, bottom=0.03, left=0.04, right=0.98, hspace=0.2, wspace=0.3)
plt.savefig('../figures/SFIG2.pdf',format='pdf')
plt.savefig('/Users/zhugchzo/Desktop/3paper_fig/SFIG2.png',format='png',dpi=600)