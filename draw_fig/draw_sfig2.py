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

fig, axs = plt.subplots(3, 3, figsize=(12,15))

ax1, ax2, ax3 = axs[0,0], axs[0,1], axs[0,2]
ax4, ax5, ax6 = axs[1,0], axs[1,1], axs[1,2]
ax7, ax8, ax9 = axs[2,0], axs[2,1], axs[2,2]

labels = ['\u2460', '\u2461', '\u2462']
base_colors = ['cornflowerblue', 'blueviolet', 'violet']

colors = [to_rgba(c, alpha=0.9) for c in base_colors]

#library4######################################################################################

# ax1

values = [1, 0.04, 0.43]
percent_text = ['100%', '4%', '43%']

bars = ax1.bar(labels, values, color=colors, width=0.5)

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

ax1.set_ylabel('Bifurcation detection rate',font_y)

ax1.text(-0.15, 1.1,'a',ha='left', transform=ax1.transAxes,fontdict={'family':'Arial','size':30,'weight':'bold'})

# ax2 cusp bif errror

box_data_ax2 = [
    {'med': 7.0, 'q1': 3.25, 'q3': 13.0, 'whislo': 1, 'whishi': 24, 'fliers': [29,28,29,108,31,28]},
    {'med': 301.0, 'q1': 231.0, 'q3': 323.25, 'whislo': 92.625, 'whishi': 461.625, 'fliers': [51]},
    {'med': 253.0, 'q1': 240.0, 'q3': 266.0, 'whislo': 230, 'whishi': 300, 'fliers': []}
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

ax2.set_ylim(-(500/44),500)
ax2.set_yticks([0,450])

ax2.tick_params(axis='x', direction='out')
ax2.tick_params(axis='y', direction='in')

for spine in ['top', 'right']:
    ax2.spines[spine].set_visible(False)
for spine in ['left', 'bottom']:
    ax2.spines[spine].set_linewidth(1)

ax2.grid(False)

ax2.set_ylabel('Bifurcation inaccuracy (timepoint)',font_y)
ax2.yaxis.set_label_coords(-0.2, 0.5)

ax2.text(-0.15, 1.1,'b',ha='left', transform=ax2.transAxes,fontdict={'family':'Arial','size':30,'weight':'bold'})

# ax3 cusp NED

box_data_ax3 = [
    {'med': 0.0035316635604938527, 'q1': 0.002701812779790569, 'q3': 0.0048636513932439115, 'whislo': 5.670081156007021e-05, 'whishi': 0.006412955220832086, 'fliers': []},
    {'med': 1.2438233388067883, 'q1': 1.1822537315561088, 'q3': 1.2537318635180288, 'whislo': 1.0750365336132288, 'whishi': 1.3609490614609088, 'fliers': [1.00871654]},
    {'med': 1.242851725919981, 'q1': 1.2262518773653017, 'q3': 1.2525760827930033, 'whislo': 1.2036949610789742, 'whishi': 1.2672164677139666, 'fliers': []}
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

ax3.set_ylim(-(1.5/44),1.5)
ax3.set_yticks([0,1.2])
ax3.set_yticklabels(['0', '1.2'])

ax3.tick_params(axis='x', direction='out')
ax3.tick_params(axis='y', direction='in')

for spine in ['top', 'right']:
    ax3.spines[spine].set_visible(False)
for spine in ['left', 'bottom']:
    ax3.spines[spine].set_linewidth(1)

ax3.grid(False)

ax3.set_ylabel('Prediction inaccuracy (NED)',font_y)

ax3.text(-0.15, 1.1,'c',ha='left', transform=ax3.transAxes,fontdict={'family':'Arial','size':30,'weight':'bold'})

#library5######################################################################################

# ax4

values = [0.85, 0, 0.11]
percent_text = ['85%', '0%', '11%']

bars = ax4.bar(labels, values, color=colors, width=0.5)

for bar, pct in zip(bars, percent_text):
    height = bar.get_height()
    ax4.text(bar.get_x() + 1.25*bar.get_width()/2, height + 0.05, pct,
             ha='center', va='bottom', fontsize=15, fontname='Arial', weight='bold')

ax4.set_xticks([0, 1, 2])
ax4.set_xticklabels(labels, ha='center', fontdict=font_x1)

for spine in ['top', 'right']:
    ax4.spines[spine].set_visible(False)
for spine in ['left', 'bottom']:
    ax4.spines[spine].set_linewidth(1)

ax4.set_ylim(-0.025,1.1)
ax4.set_yticks([0,1])

ax4.tick_params(axis='x', direction='out')
ax4.tick_params(axis='y', direction='in')

ax4.set_ylabel('Bifurcation detection rate',font_y)

ax4.text(-0.15, 1.1,'d',ha='left', transform=ax4.transAxes,fontdict={'family':'Arial','size':30,'weight':'bold'})

# ax5 cusp bif errror

box_data_ax5 = [
    {'med': 10.0, 'q1': 3.0, 'q3': 14.0, 'whislo': 0, 'whishi': 30, 'fliers': [37,38,48,41,73,118,35,108,144,50,49]},
    {'med': 275.0, 'q1': 254.5, 'q3': 302.0, 'whislo': 221, 'whishi': 314, 'fliers': []}
]

boxplot = ax5.bxp(box_data_ax5, positions=[1, 3], widths=0.6, showfliers=True,
                  patch_artist=True,
                  boxprops=dict(edgecolor='k', linewidth=1, alpha=0.8),
                  medianprops=dict(color='k', linewidth=1.5, alpha=0.8),
                  whiskerprops=dict(color='k', linewidth=1, alpha=0.8),
                  capprops=dict(color='k', linewidth=1, alpha=0.8),
                  flierprops=dict(marker='o', markersize=4, linestyle='none',
                                  markerfacecolor='dimgray', markeredgecolor='k'))

for patch, color in zip(boxplot['boxes'], [colors[0],colors[2]]):
    patch.set_facecolor(color)

ax5.text(2, 350/6, 'no bifurcation', ha='center', va='center', fontsize=15, fontweight='bold', rotation=45)

ax5.set_xticks([1, 2, 3])
ax5.set_xticklabels(labels, ha='center', fontdict=font_x1)

ax5.set_ylim(-(350/44),350)
ax5.set_yticks([0,300])

ax5.tick_params(axis='x', direction='out')
ax5.tick_params(axis='y', direction='in')

for spine in ['top', 'right']:
    ax5.spines[spine].set_visible(False)
for spine in ['left', 'bottom']:
    ax5.spines[spine].set_linewidth(1)

ax5.grid(False)

ax5.set_ylabel('Bifurcation inaccuracy (timepoint)',font_y)
ax5.yaxis.set_label_coords(-0.2, 0.5)

ax5.text(-0.15, 1.1,'e',ha='left', transform=ax5.transAxes,fontdict={'family':'Arial','size':30,'weight':'bold'})

# ax6 cusp NED

box_data_ax6 = [
    {'med': 0.0034118375933913153, 'q1': 0.001024615220432068, 'q3': 0.007266950135640166, 'whislo': 2.691014205933858e-05, 'whishi': 0.013831109471891272, 'fliers': [0.01892834]},
    {'med': 1.136297411210664, 'q1': 1.049507844799646, 'q3': 1.1471614073946257, 'whislo': 1.0452954398020995, 'whishi': 1.1522917565665796, 'fliers': []}
]

boxplot = ax6.bxp(box_data_ax6, positions=[1, 3], widths=0.6, showfliers=True,
                  patch_artist=True,
                  boxprops=dict(edgecolor='k', linewidth=1, alpha=0.8),
                  medianprops=dict(color='k', linewidth=1.5, alpha=0.8),
                  whiskerprops=dict(color='k', linewidth=1, alpha=0.8),
                  capprops=dict(color='k', linewidth=1, alpha=0.8),
                  flierprops=dict(marker='o', markersize=4, linestyle='none',
                                  markerfacecolor='dimgray', markeredgecolor='k'))

for patch, color in zip(boxplot['boxes'], [colors[0],colors[2]]):
    patch.set_facecolor(color)

ax6.text(2, 1.2/6, 'no bifurcation', ha='center', va='center', fontsize=15, fontweight='bold', rotation=45)

ax6.set_xticks([1, 2, 3])
ax6.set_xticklabels(labels, ha='center', fontdict=font_x1)

ax6.set_ylim(-(1.2/44),1.2)
ax6.set_yticks([0,1])
ax6.set_yticklabels(['0', '1'])

ax6.tick_params(axis='x', direction='out')
ax6.tick_params(axis='y', direction='in')

for spine in ['top', 'right']:
    ax6.spines[spine].set_visible(False)
for spine in ['left', 'bottom']:
    ax6.spines[spine].set_linewidth(1)

ax6.grid(False)

ax6.set_ylabel('Prediction inaccuracy (NED)',font_y,labelpad=17)

ax6.text(-0.15, 1.1,'f',ha='left', transform=ax6.transAxes,fontdict={'family':'Arial','size':30,'weight':'bold'})

xlim = ax6.get_xlim()

#library6######################################################################################

# ax7

values = [0.6, 0, 0]
percent_text = ['60%', '0%', '0%']

bars = ax7.bar(labels, values, color=colors, width=0.5)

for bar, pct in zip(bars, percent_text):
    height = bar.get_height()
    ax7.text(bar.get_x() + 1.25*bar.get_width()/2, height + 0.05, pct,
             ha='center', va='bottom', fontsize=15, fontname='Arial', weight='bold')

ax7.set_xticks([0, 1, 2])
ax7.set_xticklabels(labels, ha='center', fontdict=font_x1)

for spine in ['top', 'right']:
    ax7.spines[spine].set_visible(False)
for spine in ['left', 'bottom']:
    ax7.spines[spine].set_linewidth(1)

ax7.set_ylim(-0.025,1.1)
ax7.set_yticks([0,1])

ax7.tick_params(axis='x', direction='out')
ax7.tick_params(axis='y', direction='in')

ax7.set_ylabel('Bifurcation detection rate',font_y)

ax7.text(-0.15, 1.1,'g',ha='left', transform=ax7.transAxes,fontdict={'family':'Arial','size':30,'weight':'bold'})

# ax8 cusp bif errror

box_data_ax8 = [
    {'med': 18.5, 'q1': 13.0, 'q3': 31.0, 'whislo': 1, 'whishi': 50, 'fliers': [78,103,63,126]}
]

boxplot = ax8.bxp(box_data_ax8, positions=[1], widths=0.6, showfliers=True,
                  patch_artist=True,
                  boxprops=dict(edgecolor='k', linewidth=1, alpha=0.8),
                  medianprops=dict(color='k', linewidth=1.5, alpha=0.8),
                  whiskerprops=dict(color='k', linewidth=1, alpha=0.8),
                  capprops=dict(color='k', linewidth=1, alpha=0.8),
                  flierprops=dict(marker='o', markersize=4, linestyle='none',
                                  markerfacecolor='dimgray', markeredgecolor='k'))

for patch, color in zip(boxplot['boxes'], [colors[0],colors[2]]):
    patch.set_facecolor(color)

ax8.text(2, 150/6, 'no bifurcation', ha='center', va='center', fontsize=15, fontweight='bold', rotation=45)
ax8.text(3, 150/6, 'no bifurcation', ha='center', va='center', fontsize=15, fontweight='bold', rotation=45)

ax8.set_xticks([1, 2, 3])
ax8.set_xticklabels(labels, ha='center', fontdict=font_x1)
ax8.set_xlim(xlim)

ax8.set_ylim(-(150/44),150)
ax8.set_yticks([0,120])

ax8.tick_params(axis='x', direction='out')
ax8.tick_params(axis='y', direction='in')

for spine in ['top', 'right']:
    ax8.spines[spine].set_visible(False)
for spine in ['left', 'bottom']:
    ax8.spines[spine].set_linewidth(1)

ax8.grid(False)

ax8.set_ylabel('Bifurcation inaccuracy (timepoint)',font_y)
ax8.yaxis.set_label_coords(-0.2, 0.5)

ax8.text(-0.15, 1.1,'h',ha='left', transform=ax8.transAxes,fontdict={'family':'Arial','size':30,'weight':'bold'})

# ax9 cusp NED

box_data_ax9 = [
    {'med': 0.002795616591624174, 'q1':  0.0013431631226988089, 'q3': 0.004858445512180334, 'whislo': 5.899334200037905e-05, 'whishi':0.009981968236733864, 'fliers': [0.0125806]}
]

boxplot = ax9.bxp(box_data_ax9, positions=[1], widths=0.6, showfliers=True,
                  patch_artist=True,
                  boxprops=dict(edgecolor='k', linewidth=1, alpha=0.8),
                  medianprops=dict(color='k', linewidth=1.5, alpha=0.8),
                  whiskerprops=dict(color='k', linewidth=1, alpha=0.8),
                  capprops=dict(color='k', linewidth=1, alpha=0.8),
                  flierprops=dict(marker='o', markersize=4, linestyle='none',
                                  markerfacecolor='dimgray', markeredgecolor='k'))

for patch, color in zip(boxplot['boxes'], colors):
    patch.set_facecolor(color)

ax9.text(2, 0.015/6, 'no bifurcation', ha='center', va='center', fontsize=15, fontweight='bold', rotation=45)
ax9.text(3, 0.015/6, 'no bifurcation', ha='center', va='center', fontsize=15, fontweight='bold', rotation=45)

ax9.set_xticks([1, 2, 3])
ax9.set_xticklabels(labels, ha='center', fontdict=font_x1)
ax9.set_xlim(xlim)

ax9.set_ylim(-(0.015/44),0.015)
ax9.set_yticks([0,0.012])
ax9.set_yticklabels(['0', '0.012'])

ax9.tick_params(axis='x', direction='out')
ax9.tick_params(axis='y', direction='in')

for spine in ['top', 'right']:
    ax9.spines[spine].set_visible(False)
for spine in ['left', 'bottom']:
    ax9.spines[spine].set_linewidth(1)

ax9.grid(False)

ax9.set_ylabel('Prediction inaccuracy (NED)',font_y)

ax9.text(-0.15, 1.1,'i',ha='left', transform=ax9.transAxes,fontdict={'family':'Arial','size':30,'weight':'bold'})

#######################################################################################

plt.subplots_adjust(top=0.88, bottom=0.03, left=0.05, right=0.98, hspace=0.6, wspace=0.4)

fig.text(0.5, 0.95, 'Degree-4 basis function library', ha='center', va='center',fontsize=25,fontname='Arial',color='navy')
fig.text(0.5, 0.63, 'Degree-5 basis function library', ha='center', va='center',fontsize=25,fontname='Arial',color='navy')
fig.text(0.5, 0.3, 'Degree-6 basis function library', ha='center', va='center',fontsize=25,fontname='Arial',color='navy')

legend_v = mlines.Line2D([], [], color='cornflowerblue', linestyle='-',linewidth=20)
legend_p = mlines.Line2D([], [], color='blueviolet', linestyle='-',linewidth=20)
legend_t = mlines.Line2D([], [], color='violet', linestyle='-',linewidth=20)

fig.legend(handles = [legend_v, legend_p, legend_t],
            labels = ['\u2460  Optimal driving variable','\u2461  Forcing parameters','\u2462  Time variable'],
              loc='upper center', bbox_to_anchor=(0.5,1.01), ncol=3, frameon=False, handlelength=1.5,
              prop=font_manager.FontProperties(family='Arial Unicode MS', size=20))

plt.savefig('../figures/SFIG2.pdf',format='pdf')