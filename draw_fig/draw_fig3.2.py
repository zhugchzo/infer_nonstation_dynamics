import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib import font_manager
from matplotlib.colors import to_rgba

font_x = {'family':'Arial','weight':'medium','size': 24}
font_x1 = {'family':'Arial Unicode MS','weight':'medium','size': 20}
font_y = {'family':'Arial','weight':'medium','size': 16}
font_title = {'family':'DejaVu Sans','weight':'normal','size': 15, 'style': 'italic'}

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15

fig, axs = plt.subplots(1, 5, figsize=(15,5))

ax1, ax2, ax3, ax4, ax5 = axs[0], axs[1], axs[2], axs[3], axs[4]

labels = ['\u2460', '\u2461', '\u2462']
base_colors = ['cornflowerblue', 'blueviolet', 'violet']

colors = [to_rgba(c, alpha=0.9) for c in base_colors]

# ax1

values = [1, 0.19, 0.7]
percent_text = ['100%', '19%', '70%']

bars = ax1.bar(labels, values, color=colors, width=0.6)

for bar, pct in zip(bars, percent_text):
    height = bar.get_height()
    ax1.text(bar.get_x() + 1.25*bar.get_width()/2, height + 0.05, pct,
             ha='center', va='bottom', fontsize=15, fontname='Arial', weight='bold')

ax1.set_xticks([0, 1, 2])
ax1.set_xticklabels(labels, ha='center', fontdict=font_x1)

for spine in ['top', 'right']:
    ax1.spines[spine].set_visible(False)
for spine in ['left', 'bottom']:
    ax1.spines[spine].set_linewidth(1)

ax1.set_ylim(-0.025,1.1)
ax1.set_yticks([0,1])

ax1.tick_params(axis='x', direction='out')
ax1.tick_params(axis='y', direction='in')

ax1.set_ylabel('Proportion of bifurcation',font_y)

ax1.text(-0.15, 1.15,'c',ha='left', transform=ax1.transAxes,fontdict={'family':'Arial','size':30,'weight':'bold'})

# ax2 cusp bif errror

box_data_ax2 = [
    {'med': 2.5, 'q1': 1.0, 'q3': 4.0, 'whislo': 0, 'whishi': 8, 'fliers': [9]},
    {'med': 40.0, 'q1': 23.5, 'q3': 70.0, 'whislo': 1, 'whishi': 129, 'fliers': [154]},
    {'med': 38.5, 'q1': 15.5, 'q3': 55.75, 'whislo': 1, 'whishi': 84, 'fliers': []}
]

boxplot = ax2.bxp(box_data_ax2, positions=[1, 2, 3], widths=0.6, showfliers=True,
                  patch_artist=True,
                  boxprops=dict(edgecolor='k', linewidth=1, alpha=0.8),
                  medianprops=dict(color='k', linewidth=1.5, alpha=0.8),
                  whiskerprops=dict(color='k', linewidth=1, alpha=0.8),
                  capprops=dict(color='k', linewidth=1, alpha=0.8),
                  flierprops=dict(marker='o', markersize=4, linestyle='none',
                                  markerfacecolor='dimgray', markeredgecolor='k'))

for patch, color in zip(boxplot['boxes'], colors):
    patch.set_facecolor(color)

ax2.set_xticks([1, 2, 3])
ax2.set_xticklabels(labels, ha='center', fontdict=font_x1)

ax2.set_ylim(-(160/44),160)
ax2.set_yticks([0,150])

ax2.tick_params(axis='x', direction='out')
ax2.tick_params(axis='y', direction='in')

for spine in ['top', 'right']:
    ax2.spines[spine].set_visible(False)
for spine in ['left', 'bottom']:
    ax2.spines[spine].set_linewidth(1)

ax2.grid(False)

ax2.set_ylabel('Bifurcation inaccuracy (timepoint)',font_y)
ax2.yaxis.set_label_coords(-0.2, 0.5)

ax2.text(-0.15, 1.15,'d',ha='left', transform=ax2.transAxes,fontdict={'family':'Arial','size':30,'weight':'bold'})

# ax3 cusp NED

box_data_ax3 = [
    {'med': 0.0016802107925532588, 'q1': 0.0014221597789689005, 'q3': 0.001831719250326466, 'whislo': 0.0008443017457906452, 'whishi': 0.0020409223629032753, 'fliers': [0.00078741, 0.00075343, 0.00055752, 0.00042978, 0.00028405, 0.0007167]},
    {'med': 0.197261023029447, 'q1': 0.170148055389852, 'q3': 0.23601663067418574, 'whislo': 0.13852815704586935, 'whishi': 0.27493014649418274, 'fliers': [0.40842582]},
    {'med': 0.011514888000427697, 'q1': 0.0034661445201276016, 'q3': 0.026021935672772792, 'whislo': 0.00018700873385007628, 'whishi': 0.05688103846390456, 'fliers': []}
]

boxplot = ax3.bxp(box_data_ax3, positions=[1, 2, 3], widths=0.6, showfliers=True,
                  patch_artist=True,
                  boxprops=dict(edgecolor='k', linewidth=1, alpha=0.8),
                  medianprops=dict(color='k', linewidth=1.5, alpha=0.8),
                  whiskerprops=dict(color='k', linewidth=1, alpha=0.8),
                  capprops=dict(color='k', linewidth=1, alpha=0.8),
                  flierprops=dict(marker='o', markersize=4, linestyle='none',
                                  markerfacecolor='dimgray', markeredgecolor='k'))

for patch, color in zip(boxplot['boxes'], colors):
    patch.set_facecolor(color)

ax3.set_xticks([1, 2, 3])
ax3.set_xticklabels(labels, ha='center', fontdict=font_x1)

ax3.set_ylim(-(0.44/44),0.44)
ax3.set_yticks([0,0.4])
ax3.set_yticklabels(['0', '0.4'])

ax3.tick_params(axis='x', direction='out')
ax3.tick_params(axis='y', direction='in')

for spine in ['top', 'right']:
    ax3.spines[spine].set_visible(False)
for spine in ['left', 'bottom']:
    ax3.spines[spine].set_linewidth(1)

ax3.grid(False)

ax3.set_ylabel('Prediction inaccuracy (NED)',font_y)

ax3.text(-0.15, 1.15,'e',ha='left', transform=ax3.transAxes,fontdict={'family':'Arial','size':30,'weight':'bold'})

# ax4 Koscillators sin+ NED

box_data_ax4 = [
    {'med': 0.058481485581073786, 'q1': 0.045587354181636425, 'q3': 0.0845272922374028, 'whislo': 0.01161080117180554, 'whishi': 0.14061468678858047,
      'fliers': [0.15168087, 0.17183394, 0.18513509, 0.16195769, 0.26443577, 0.20078666, 0.53163078, 0.18928288, 0.1446918, 0.30165725, 0.45750837, 0.40360858, 0.1431024, 0.15310802]},
    {'med': 0.09078157349288518, 'q1': 0.054745330506964444, 'q3': 0.17334841532458703, 'whislo': 0.011480777172070224, 'whishi': 0.3300182058900741,
      'fliers': [0.64668358, 0.39989923, 0.58481462, 0.40258555, 0.45619668, 0.53087003, 0.50069802, 0.69466915, 0.65408875, 0.39691462, 0.45750837, 0.40174435]},
    {'med': 0.0807471940051033, 'q1': 0.04966424867513705, 'q3': 0.18780399355377037, 'whislo': 0.01148077717207015, 'whishi': 0.39021255484173484,
      'fliers': [0.66921586, 0.52549625, 0.40213643, 0.45619668, 0.5189422, 0.53758802, 0.69466915, 0.42479783, 0.65408875, 0.51828722, 0.40782487]}
]

boxplot = ax4.bxp(box_data_ax4, positions=[1, 2, 3], widths=0.6, showfliers=True,
                  patch_artist=True,
                  boxprops=dict(edgecolor='k', linewidth=1, alpha=0.8),
                  medianprops=dict(color='k', linewidth=1.5, alpha=0.8),
                  whiskerprops=dict(color='k', linewidth=1, alpha=0.8),
                  capprops=dict(color='k', linewidth=1, alpha=0.8),
                  flierprops=dict(marker='o', markersize=4, linestyle='none',
                                  markerfacecolor='dimgray', markeredgecolor='k'))

for patch, color in zip(boxplot['boxes'], colors):
    patch.set_facecolor(color)

ax4.set_xticks([1, 2, 3])
ax4.set_xticklabels(labels, ha='center', fontdict=font_x1)

ax4.set_ylim(-(0.77/44),0.77)
ax4.set_yticks([0,0.7])
ax4.set_yticklabels(['0', '0.7'])

ax4.tick_params(axis='x', direction='out')
ax4.tick_params(axis='y', direction='in')

for spine in ['top', 'right']:
    ax4.spines[spine].set_visible(False)
for spine in ['left', 'bottom']:
    ax4.spines[spine].set_linewidth(1)

ax4.grid(False)

ax4.set_ylabel('Prediction inaccuracy (NED)',font_y)

ax4.set_title(r'Interaction: $\sin(\theta_j+\theta_i)$',fontdict=font_title,y=1.1)

ax4.text(-0.25, 1.15,'f',ha='left', transform=ax4.transAxes,fontdict={'family':'Arial','size':30,'weight':'bold'})


# ax5 Koscillators cos- NED

box_data_ax5 = [
    {'med': 0.07551759239978195, 'q1': 0.05890319543582125, 'q3': 0.11488246629154836, 'whislo': 0.04160420793767043, 'whishi': 0.19366278263482456,
      'fliers': [0.22915556, 0.48886963, 0.2200613, 0.29888296, 0.32988459, 0.23714655, 0.26664822]},
    {'med': 0.12383854619281821, 'q1': 0.07912675347377963, 'q3': 0.24443759649872357, 'whislo': 0.04396895861848049, 'whishi': 0.491853680524469,
      'fliers': [0.50910902, 0.60039138, 0.59947012, 0.67461606, 0.63567653, 0.66471854]},
    {'med': 0.11828701413277307, 'q1': 0.05646147466207225, 'q3': 0.2584688356251418, 'whislo': 0.03504564563799664, 'whishi': 0.4794308881231995,
      'fliers': [0.58184795, 0.63467994, 0.59947012, 0.60915031, 0.57695468, 0.63567653, 0.58991725]}
]

boxplot = ax5.bxp(box_data_ax5, positions=[1, 2, 3], widths=0.6, showfliers=True,
                  patch_artist=True,
                  boxprops=dict(edgecolor='k', linewidth=1, alpha=0.8),
                  medianprops=dict(color='k', linewidth=1.5, alpha=0.8),
                  whiskerprops=dict(color='k', linewidth=1, alpha=0.8),
                  capprops=dict(color='k', linewidth=1, alpha=0.8),
                  flierprops=dict(marker='o', markersize=4, linestyle='none',
                                  markerfacecolor='dimgray', markeredgecolor='k'))

for patch, color in zip(boxplot['boxes'], colors):
    patch.set_facecolor(color)

ax5.set_xticks([1, 2, 3])
ax5.set_xticklabels(labels, ha='center', fontdict=font_x1)

ax5.set_ylim(-(0.77/44),0.77)
ax5.set_yticks([0,0.7])
ax5.set_yticklabels(['0', '0.7'])

ax5.tick_params(axis='x', direction='out')
ax5.tick_params(axis='y', direction='in')

for spine in ['top', 'right']:
    ax5.spines[spine].set_visible(False)
for spine in ['left', 'bottom']:
    ax5.spines[spine].set_linewidth(1)

ax5.grid(False)

ax5.set_ylabel('Prediction inaccuracy (NED)',font_y)

ax5.set_title(r'Interaction: $\cos(\theta_j-\theta_i)$',fontdict=font_title,y=1.1)

ax5.text(-0.25, 1.15,'g',ha='left', transform=ax5.transAxes,fontdict={'family':'Arial','size':30,'weight':'bold'})

plt.subplots_adjust(top=0.6, bottom=0.1, left=0.05, right=0.98, wspace=0.35)

fig.text(0.3, 0.85, 'Cusp bifurcation', ha='center', va='center',fontsize=25,fontname='Arial',color='navy')
fig.text(0.8, 0.85, 'Kuramoto oscillators', ha='center', va='center',fontsize=25,fontname='Arial',color='navy')


# legend_v = mlines.Line2D([], [], color='cornflowerblue', linestyle='-',linewidth=3)
# legend_p = mlines.Line2D([], [], color='blueviolet', linestyle='-',linewidth=3)
# legend_t = mlines.Line2D([], [], color='violet', linestyle='-',linewidth=3)

# fig.legend(handles = [legend_v, legend_p, legend_t],
#             labels = ['\u2460  Optimal driving signal','\u2461  Forcing parameters','\u2462  Time variable'],
#               loc='upper center', bbox_to_anchor=(0.5,1), ncol=3, frameon=False, 
#               prop=font_manager.FontProperties(family='Arial Unicode MS', size=20))

plt.savefig('../figures/FIG3.2.pdf',format='pdf')
plt.savefig('/Users/zhugchzo/Desktop/3paper_fig/FIG3.2.png',format='png',dpi=600)