import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

font_x = {'family':'Arial','weight':'medium','size': 24}
font_y = {'family':'Arial','weight':'medium','size': 16}
font_title = {'family':'DejaVu Sans','weight':'normal','size': 15, 'style': 'italic'}

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15

# df for ax1
df_cusp_smape_1 = pd.read_csv('../results/sMAPE/library/cusp_smape_500.csv')
# df for ax2
df_cusp_smape_2 = pd.read_csv('../results/sMAPE/library/cusp_smape_550.csv')
# df for ax3
df_cusp_smape_3 = pd.read_csv('../results/sMAPE/library/cusp_smape_600.csv')
# df for ax4
df_cusp_smape_4 = pd.read_csv('../results/sMAPE/library/cusp_smape_650.csv')
# df for ax5
df_cusp_smape_5 = pd.read_csv('../results/sMAPE/library/cusp_smape_700.csv')
# df for ax6
df_Koscillators_smape_1 = pd.read_csv('../results/sMAPE/library/Koscillators_smape_300.csv')
# df for ax7
df_Koscillators_smape_2 = pd.read_csv('../results/sMAPE/library/Koscillators_smape_350.csv')
# df for ax8
df_Koscillators_smape_3 = pd.read_csv('../results/sMAPE/library/Koscillators_smape_400.csv')
# df for ax9
df_Koscillators_smape_4 = pd.read_csv('../results/sMAPE/library/Koscillators_smape_450.csv')
# df for ax10
df_Koscillators_smape_5 = pd.read_csv('../results/sMAPE/library/Koscillators_smape_500.csv')


fig, axs = plt.subplots(2, 5, figsize=(15,9))

ax1, ax2, ax3, ax4, ax5 = axs[0, 0], axs[0, 1], axs[0, 2], axs[0, 3], axs[0, 4]
ax6, ax7, ax8, ax9, ax10 = axs[1, 0], axs[1, 1], axs[1, 2], axs[1, 3], axs[1, 4]

cusp_cols = ['mean_smape', 'lower_smape', 'upper_smape', 'mean_smape_ab', 'lower_smape_ab', 'upper_smape_ab', 'mean_smape_t', 'lower_smape_t', 'upper_smape_t']
Koscillators_cols = ['mean_smape', 'lower_smape', 'upper_smape', 'mean_smape_p', 'lower_smape_p', 'upper_smape_p', 'mean_smape_t', 'lower_smape_t', 'upper_smape_t']

cusp_library = [3,4,5,6]
Koscillators_library = [2,3,4,5]

# ax1
cusp_smape = df_cusp_smape_1[cusp_cols].to_numpy()

smape_ax1 = cusp_smape[:,0]
lower_ax1 = cusp_smape[:,1]
upper_ax1 = cusp_smape[:,2]

smape_ab_ax1 = cusp_smape[:,3]
lower_ab_ax1 = cusp_smape[:,4]
upper_ab_ax1 = cusp_smape[:,5]

smape_t_ax1 = cusp_smape[:,6]
lower_t_ax1 = cusp_smape[:,7]
upper_t_ax1 = cusp_smape[:,8]

ax1.plot(cusp_library,smape_ax1,c='cornflowerblue',linewidth=3,alpha=0.9,zorder=2)
ax1.scatter(cusp_library,smape_ax1,c='cornflowerblue',marker='s',s=40,zorder=2)
ax1.fill_between(cusp_library,lower_ax1,upper_ax1,color='cornflowerblue',alpha=0.15,linewidth=0,zorder=2)

ax1.plot(cusp_library,smape_ab_ax1,c='blueviolet',linewidth=3,alpha=0.9,zorder=1)
ax1.scatter(cusp_library,smape_ab_ax1,c='blueviolet',marker='s',s=40,zorder=1)
ax1.fill_between(cusp_library,lower_ab_ax1,upper_ab_ax1,color='blueviolet',alpha=0.15,linewidth=0,zorder=1)

ax1.plot(cusp_library,smape_t_ax1,c='violet',linewidth=3,alpha=0.9,zorder=1)
ax1.scatter(cusp_library,smape_t_ax1,c='violet',marker='s',s=40,zorder=1)
ax1.fill_between(cusp_library,lower_t_ax1,upper_t_ax1,color='violet',alpha=0.15,linewidth=0,zorder=1)

ax1.set_xlim(2.7,6.3)
ax1.set_xticks([3,4,5,6])
ax1.set_ylim(-0.1,1.1)
ax1.set_yticks([0,1])
ax1.tick_params(direction='in')

ax1.set_ylabel('Inference inaccuracy (sMAPE)',font_y)
ax1.yaxis.set_label_coords(-0.1, 0.45)

legend_v = mlines.Line2D([], [], color='cornflowerblue', marker='s',markersize=3,linestyle='-',linewidth=2)
legend_p = mlines.Line2D([], [], color='blueviolet', marker='s',markersize=3,linestyle='-',linewidth=2)
legend_t = mlines.Line2D([], [], color='violet', marker='s',markersize=3,linestyle='-',linewidth=2)
ax1.legend(handles=[legend_v,legend_p,legend_t],
           labels=['Virtual variable','   Forcing \nparameters','Time variable'],
           loc='center',bbox_to_anchor=(0.6, 0.15),frameon=False, markerscale=2, prop={'size':12})

ax1.set_title('500 data points',fontdict=font_title)
ax1.text(-0.18, 1,'a',ha='left', transform=ax1.transAxes,fontdict={'family':'Arial','size':24,'weight':'bold'})

# ax2
cusp_smape = df_cusp_smape_2[cusp_cols].to_numpy()

smape_ax2 = cusp_smape[:,0]
lower_ax2 = cusp_smape[:,1]
upper_ax2 = cusp_smape[:,2]

smape_ab_ax2 = cusp_smape[:,3]
lower_ab_ax2 = cusp_smape[:,4]
upper_ab_ax2 = cusp_smape[:,5]

smape_t_ax2 = cusp_smape[:,6]
lower_t_ax2 = cusp_smape[:,7]
upper_t_ax2 = cusp_smape[:,8]

ax2.plot(cusp_library,smape_ax2,c='cornflowerblue',linewidth=3,alpha=0.9,zorder=2)
ax2.scatter(cusp_library,smape_ax2,c='cornflowerblue',marker='s',s=40,zorder=2)
ax2.fill_between(cusp_library,lower_ax2,upper_ax2,color='cornflowerblue',alpha=0.15,linewidth=0,zorder=2)

ax2.plot(cusp_library,smape_ab_ax2,c='blueviolet',linewidth=3,alpha=0.9,zorder=1)
ax2.scatter(cusp_library,smape_ab_ax2,c='blueviolet',marker='s',s=40,zorder=1)
ax2.fill_between(cusp_library,lower_ab_ax2,upper_ab_ax2,color='blueviolet',alpha=0.15,linewidth=0,zorder=1)

ax2.plot(cusp_library,smape_t_ax2,c='violet',linewidth=3,alpha=0.9,zorder=1)
ax2.scatter(cusp_library,smape_t_ax2,c='violet',marker='s',s=40,zorder=1)
ax2.fill_between(cusp_library,lower_t_ax2,upper_t_ax2,color='violet',alpha=0.15,linewidth=0,zorder=1)

ax2.set_xlim(2.7,6.3)
ax2.set_xticks([3,4,5,6])
ax2.set_ylim(-0.1,1.1)
ax2.set_yticks([0,1])
ax2.tick_params(direction='in')

ax2.set_ylabel('Inference inaccuracy (sMAPE)',font_y)
ax2.yaxis.set_label_coords(-0.1, 0.45)

ax2.set_title('550 data points',fontdict=font_title)
ax2.text(-0.18, 1,'b',ha='left', transform=ax2.transAxes,fontdict={'family':'Arial','size':24,'weight':'bold'})

# ax3
cusp_smape = df_cusp_smape_3[cusp_cols].to_numpy()

smape_ax3 = cusp_smape[:,0]
lower_ax3 = cusp_smape[:,1]
upper_ax3 = cusp_smape[:,2]

smape_ab_ax3 = cusp_smape[:,3]
lower_ab_ax3 = cusp_smape[:,4]
upper_ab_ax3 = cusp_smape[:,5]

smape_t_ax3 = cusp_smape[:,6]
lower_t_ax3 = cusp_smape[:,7]
upper_t_ax3 = cusp_smape[:,8]

ax3.plot(cusp_library,smape_ax3,c='cornflowerblue',linewidth=3,alpha=0.9,zorder=2)
ax3.scatter(cusp_library,smape_ax3,c='cornflowerblue',marker='s',s=40,zorder=2)
ax3.fill_between(cusp_library,lower_ax3,upper_ax3,color='cornflowerblue',alpha=0.15,linewidth=0,zorder=2)

ax3.plot(cusp_library,smape_ab_ax3,c='blueviolet',linewidth=3,alpha=0.9,zorder=1)
ax3.scatter(cusp_library,smape_ab_ax3,c='blueviolet',marker='s',s=40,zorder=1)
ax3.fill_between(cusp_library,lower_ab_ax3,upper_ab_ax3,color='blueviolet',alpha=0.15,linewidth=0,zorder=1)

ax3.plot(cusp_library,smape_t_ax3,c='violet',linewidth=3,alpha=0.9,zorder=1)
ax3.scatter(cusp_library,smape_t_ax3,c='violet',marker='s',s=40,zorder=1)
ax3.fill_between(cusp_library,lower_t_ax3,upper_t_ax3,color='violet',alpha=0.15,linewidth=0,zorder=1)

ax3.set_xlim(2.7,6.3)
ax3.set_xticks([3,4,5,6])
ax3.set_ylim(-0.1,1.1)
ax3.set_yticks([0,1])
ax3.tick_params(direction='in')

ax3.set_ylabel('Inference inaccuracy (sMAPE)',font_y)
ax3.yaxis.set_label_coords(-0.1, 0.45)

ax3.set_title('600 data points',fontdict=font_title)
ax3.text(-0.18, 1,'c',ha='left', transform=ax3.transAxes,fontdict={'family':'Arial','size':24,'weight':'bold'})

# ax4
cusp_smape = df_cusp_smape_4[cusp_cols].to_numpy()

smape_ax4 = cusp_smape[:,0]
lower_ax4 = cusp_smape[:,1]
upper_ax4 = cusp_smape[:,2]

smape_ab_ax4 = cusp_smape[:,3]
lower_ab_ax4 = cusp_smape[:,4]
upper_ab_ax4 = cusp_smape[:,5]

smape_t_ax4 = cusp_smape[:,6]
lower_t_ax4 = cusp_smape[:,7]
upper_t_ax4 = cusp_smape[:,8]

ax4.plot(cusp_library,smape_ax4,c='cornflowerblue',linewidth=3,alpha=0.9,zorder=2)
ax4.scatter(cusp_library,smape_ax4,c='cornflowerblue',marker='s',s=40,zorder=2)
ax4.fill_between(cusp_library,lower_ax4,upper_ax4,color='cornflowerblue',alpha=0.15,linewidth=0,zorder=2)

ax4.plot(cusp_library,smape_ab_ax4,c='blueviolet',linewidth=3,alpha=0.9,zorder=1)
ax4.scatter(cusp_library,smape_ab_ax4,c='blueviolet',marker='s',s=40,zorder=1)
ax4.fill_between(cusp_library,lower_ab_ax4,upper_ab_ax4,color='blueviolet',alpha=0.15,linewidth=0,zorder=1)

ax4.plot(cusp_library,smape_t_ax4,c='violet',linewidth=3,alpha=0.9,zorder=1)
ax4.scatter(cusp_library,smape_t_ax4,c='violet',marker='s',s=40,zorder=1)
ax4.fill_between(cusp_library,lower_t_ax4,upper_t_ax4,color='violet',alpha=0.15,linewidth=0,zorder=1)

ax4.set_xlim(2.7,6.3)
ax4.set_xticks([3,4,5,6])
ax4.set_ylim(-0.1,1.1)
ax4.set_yticks([0,1])
ax4.tick_params(direction='in')

ax4.set_ylabel('Inference inaccuracy (sMAPE)',font_y)
ax4.yaxis.set_label_coords(-0.1, 0.45)

ax4.set_title('650 data points',fontdict=font_title)
ax4.text(-0.18, 1,'d',ha='left', transform=ax4.transAxes,fontdict={'family':'Arial','size':24,'weight':'bold'})

# ax5
cusp_smape = df_cusp_smape_5[cusp_cols].to_numpy()

smape_ax5 = cusp_smape[:,0]
lower_ax5 = cusp_smape[:,1]
upper_ax5 = cusp_smape[:,2]

smape_ab_ax5 = cusp_smape[:,3]
lower_ab_ax5 = cusp_smape[:,4]
upper_ab_ax5 = cusp_smape[:,5]

smape_t_ax5 = cusp_smape[:,6]
lower_t_ax5 = cusp_smape[:,7]
upper_t_ax5 = cusp_smape[:,8]

ax5.plot(cusp_library,smape_ax5,c='cornflowerblue',linewidth=3,alpha=0.9,zorder=2)
ax5.scatter(cusp_library,smape_ax5,c='cornflowerblue',marker='s',s=40,zorder=2)
ax5.fill_between(cusp_library,lower_ax5,upper_ax5,color='cornflowerblue',alpha=0.15,linewidth=0,zorder=2)

ax5.plot(cusp_library,smape_ab_ax5,c='blueviolet',linewidth=3,alpha=0.9,zorder=1)
ax5.scatter(cusp_library,smape_ab_ax5,c='blueviolet',marker='s',s=40,zorder=1)
ax5.fill_between(cusp_library,lower_ab_ax5,upper_ab_ax5,color='blueviolet',alpha=0.15,linewidth=0,zorder=1)

ax5.plot(cusp_library,smape_t_ax5,c='violet',linewidth=3,alpha=0.9,zorder=1)
ax5.scatter(cusp_library,smape_t_ax5,c='violet',marker='s',s=40,zorder=1)
ax5.fill_between(cusp_library,lower_t_ax5,upper_t_ax5,color='violet',alpha=0.15,linewidth=0,zorder=1)

ax5.set_xlim(2.7,6.3)
ax5.set_xticks([3,4,5,6])
ax5.set_ylim(-0.1,1.1)
ax5.set_yticks([0,1])
ax5.tick_params(direction='in')

ax5.set_ylabel('Inference inaccuracy (sMAPE)',font_y)
ax5.yaxis.set_label_coords(-0.1, 0.45)

ax5.set_title('700 data points',fontdict=font_title)
ax5.text(-0.18, 1,'e',ha='left', transform=ax5.transAxes,fontdict={'family':'Arial','size':24,'weight':'bold'})

# ax6
Koscillators_smape = df_Koscillators_smape_1[Koscillators_cols].to_numpy()

smape_ax6 = Koscillators_smape[:,0]
lower_ax6 = Koscillators_smape[:,1]
upper_ax6 = Koscillators_smape[:,2]

smape_ab_ax6 = Koscillators_smape[:,3]
lower_ab_ax6 = Koscillators_smape[:,4]
upper_ab_ax6 = Koscillators_smape[:,5]

smape_t_ax6 = Koscillators_smape[:,6]
lower_t_ax6 = Koscillators_smape[:,7]
upper_t_ax6 = Koscillators_smape[:,8]

ax6.plot(Koscillators_library,smape_ax6,c='cornflowerblue',linewidth=3,alpha=0.9,zorder=2)
ax6.scatter(Koscillators_library,smape_ax6,c='cornflowerblue',marker='s',s=40,zorder=2)
ax6.fill_between(Koscillators_library,lower_ax6,upper_ax6,color='cornflowerblue',alpha=0.15,linewidth=0,zorder=2)

ax6.plot(Koscillators_library,smape_ab_ax6,c='blueviolet',linewidth=3,alpha=0.9,zorder=1)
ax6.scatter(Koscillators_library,smape_ab_ax6,c='blueviolet',marker='s',s=40,zorder=1)
ax6.fill_between(Koscillators_library,lower_ab_ax6,upper_ab_ax6,color='blueviolet',alpha=0.15,linewidth=0,zorder=1)

ax6.plot(Koscillators_library,smape_t_ax6,c='violet',linewidth=3,alpha=0.9,zorder=1)
ax6.scatter(Koscillators_library,smape_t_ax6,c='violet',marker='s',s=40,zorder=1)
ax6.fill_between(Koscillators_library,lower_t_ax6,upper_t_ax6,color='violet',alpha=0.15,linewidth=0,zorder=1)

ax6.set_xlim(1.7,5.3)
ax6.set_xticks([2,3,4,5])
ax6.set_ylim(-0.1,1.1)
ax6.set_yticks([0,1])
ax6.tick_params(direction='in')

ax6.set_ylabel('Inference inaccuracy (sMAPE)',font_y)
ax6.yaxis.set_label_coords(-0.1, 0.45)

legend_v = mlines.Line2D([], [], color='cornflowerblue', marker='s',markersize=3,linestyle='-',linewidth=2)
legend_p = mlines.Line2D([], [], color='blueviolet', marker='s',markersize=3,linestyle='-',linewidth=2)
legend_t = mlines.Line2D([], [], color='violet', marker='s',markersize=3,linestyle='-',linewidth=2)
ax6.legend(handles=[legend_v,legend_p,legend_t],
           labels=['Virtual variable','Forcing parameters','Time variable'],
           loc='center',bbox_to_anchor=(0.48, 0.85),frameon=False, markerscale=2, prop={'size':12})

ax6.set_title('300 data points',fontdict=font_title)
ax6.text(-0.18, 1,'f',ha='left', transform=ax6.transAxes,fontdict={'family':'Arial','size':24,'weight':'bold'})

# ax7
Koscillators_smape = df_Koscillators_smape_2[Koscillators_cols].to_numpy()

smape_ax7 = Koscillators_smape[:,0]
lower_ax7 = Koscillators_smape[:,1]
upper_ax7 = Koscillators_smape[:,2]

smape_ab_ax7 = Koscillators_smape[:,3]
lower_ab_ax7 = Koscillators_smape[:,4]
upper_ab_ax7 = Koscillators_smape[:,5]

smape_t_ax7 = Koscillators_smape[:,6]
lower_t_ax7 = Koscillators_smape[:,7]
upper_t_ax7 = Koscillators_smape[:,8]

ax7.plot(Koscillators_library,smape_ax7,c='cornflowerblue',linewidth=3,alpha=0.9,zorder=2)
ax7.scatter(Koscillators_library,smape_ax7,c='cornflowerblue',marker='s',s=40,zorder=2)
ax7.fill_between(Koscillators_library,lower_ax7,upper_ax7,color='cornflowerblue',alpha=0.15,linewidth=0,zorder=2)

ax7.plot(Koscillators_library,smape_ab_ax7,c='blueviolet',linewidth=3,alpha=0.9,zorder=1)
ax7.scatter(Koscillators_library,smape_ab_ax7,c='blueviolet',marker='s',s=40,zorder=1)
ax7.fill_between(Koscillators_library,lower_ab_ax7,upper_ab_ax7,color='blueviolet',alpha=0.15,linewidth=0,zorder=1)

ax7.plot(Koscillators_library,smape_t_ax7,c='violet',linewidth=3,alpha=0.9,zorder=1)
ax7.scatter(Koscillators_library,smape_t_ax7,c='violet',marker='s',s=40,zorder=1)
ax7.fill_between(Koscillators_library,lower_t_ax7,upper_t_ax7,color='violet',alpha=0.15,linewidth=0,zorder=1)

ax7.set_xlim(1.7,5.3)
ax7.set_xticks([2,3,4,5])
ax7.set_ylim(-0.1,1.1)
ax7.set_yticks([0,1])
ax7.tick_params(direction='in')

ax7.set_ylabel('Inference inaccuracy (sMAPE)',font_y)
ax7.yaxis.set_label_coords(-0.1, 0.45)

ax7.set_title('350 data points',fontdict=font_title)
ax7.text(-0.18, 1,'g',ha='left', transform=ax7.transAxes,fontdict={'family':'Arial','size':24,'weight':'bold'})

# ax8
Koscillators_smape = df_Koscillators_smape_3[Koscillators_cols].to_numpy()

smape_ax8 = Koscillators_smape[:,0]
lower_ax8 = Koscillators_smape[:,1]
upper_ax8 = Koscillators_smape[:,2]

smape_ab_ax8 = Koscillators_smape[:,3]
lower_ab_ax8 = Koscillators_smape[:,4]
upper_ab_ax8 = Koscillators_smape[:,5]

smape_t_ax8 = Koscillators_smape[:,6]
lower_t_ax8 = Koscillators_smape[:,7]
upper_t_ax8 = Koscillators_smape[:,8]

ax8.plot(Koscillators_library,smape_ax8,c='cornflowerblue',linewidth=3,alpha=0.9,zorder=2)
ax8.scatter(Koscillators_library,smape_ax8,c='cornflowerblue',marker='s',s=40,zorder=2)
ax8.fill_between(Koscillators_library,lower_ax8,upper_ax8,color='cornflowerblue',alpha=0.15,linewidth=0,zorder=2)

ax8.plot(Koscillators_library,smape_ab_ax8,c='blueviolet',linewidth=3,alpha=0.9,zorder=1)
ax8.scatter(Koscillators_library,smape_ab_ax8,c='blueviolet',marker='s',s=40,zorder=1)
ax8.fill_between(Koscillators_library,lower_ab_ax8,upper_ab_ax8,color='blueviolet',alpha=0.15,linewidth=0,zorder=1)

ax8.plot(Koscillators_library,smape_t_ax8,c='violet',linewidth=3,alpha=0.9,zorder=1)
ax8.scatter(Koscillators_library,smape_t_ax8,c='violet',marker='s',s=40,zorder=1)
ax8.fill_between(Koscillators_library,lower_t_ax8,upper_t_ax8,color='violet',alpha=0.15,linewidth=0,zorder=1)

ax8.set_xlim(1.7,5.3)
ax8.set_xticks([2,3,4,5])
ax8.set_ylim(-0.1,1.1)
ax8.set_yticks([0,1])
ax8.tick_params(direction='in')

ax8.set_ylabel('Inference inaccuracy (sMAPE)',font_y)
ax8.yaxis.set_label_coords(-0.1, 0.45)

ax8.set_title('400 data points',fontdict=font_title)
ax8.text(-0.18, 1,'h',ha='left', transform=ax8.transAxes,fontdict={'family':'Arial','size':24,'weight':'bold'})

# ax9
Koscillators_smape = df_Koscillators_smape_4[Koscillators_cols].to_numpy()

smape_ax9 = Koscillators_smape[:,0]
lower_ax9 = Koscillators_smape[:,1]
upper_ax9 = Koscillators_smape[:,2]

smape_ab_ax9 = Koscillators_smape[:,3]
lower_ab_ax9 = Koscillators_smape[:,4]
upper_ab_ax9 = Koscillators_smape[:,5]

smape_t_ax9 = Koscillators_smape[:,6]
lower_t_ax9 = Koscillators_smape[:,7]
upper_t_ax9 = Koscillators_smape[:,8]

ax9.plot(Koscillators_library,smape_ax9,c='cornflowerblue',linewidth=3,alpha=0.9,zorder=2)
ax9.scatter(Koscillators_library,smape_ax9,c='cornflowerblue',marker='s',s=40,zorder=2)
ax9.fill_between(Koscillators_library,lower_ax9,upper_ax9,color='cornflowerblue',alpha=0.15,linewidth=0,zorder=2)

ax9.plot(Koscillators_library,smape_ab_ax9,c='blueviolet',linewidth=3,alpha=0.9,zorder=1)
ax9.scatter(Koscillators_library,smape_ab_ax9,c='blueviolet',marker='s',s=40,zorder=1)
ax9.fill_between(Koscillators_library,lower_ab_ax9,upper_ab_ax9,color='blueviolet',alpha=0.15,linewidth=0,zorder=1)

ax9.plot(Koscillators_library,smape_t_ax9,c='violet',linewidth=3,alpha=0.9,zorder=1)
ax9.scatter(Koscillators_library,smape_t_ax9,c='violet',marker='s',s=40,zorder=1)
ax9.fill_between(Koscillators_library,lower_t_ax9,upper_t_ax9,color='violet',alpha=0.15,linewidth=0,zorder=1)

ax9.set_xlim(1.7,5.3)
ax9.set_xticks([2,3,4,5])
ax9.set_ylim(-0.1,1.1)
ax9.set_yticks([0,1])
ax9.tick_params(direction='in')

ax9.set_ylabel('Inference inaccuracy (sMAPE)',font_y)
ax9.yaxis.set_label_coords(-0.1, 0.45)

ax9.set_title('450 data points',fontdict=font_title)
ax9.text(-0.18, 1,'i',ha='left', transform=ax9.transAxes,fontdict={'family':'Arial','size':24,'weight':'bold'})

# ax10
Koscillators_smape = df_Koscillators_smape_5[Koscillators_cols].to_numpy()

smape_ax10 = Koscillators_smape[:,0]
lower_ax10 = Koscillators_smape[:,1]
upper_ax10 = Koscillators_smape[:,2]

smape_ab_ax10 = Koscillators_smape[:,3]
lower_ab_ax10 = Koscillators_smape[:,4]
upper_ab_ax10 = Koscillators_smape[:,5]

smape_t_ax10 = Koscillators_smape[:,6]
lower_t_ax10 = Koscillators_smape[:,7]
upper_t_ax10 = Koscillators_smape[:,8]

ax10.plot(Koscillators_library,smape_ax10,c='cornflowerblue',linewidth=3,alpha=0.9,zorder=2)
ax10.scatter(Koscillators_library,smape_ax10,c='cornflowerblue',marker='s',s=40,zorder=2)
ax10.fill_between(Koscillators_library,lower_ax10,upper_ax10,color='cornflowerblue',alpha=0.15,linewidth=0,zorder=2)

ax10.plot(Koscillators_library,smape_ab_ax10,c='blueviolet',linewidth=3,alpha=0.9,zorder=1)
ax10.scatter(Koscillators_library,smape_ab_ax10,c='blueviolet',marker='s',s=40,zorder=1)
ax10.fill_between(Koscillators_library,lower_ab_ax10,upper_ab_ax10,color='blueviolet',alpha=0.15,linewidth=0,zorder=1)

ax10.plot(Koscillators_library,smape_t_ax10,c='violet',linewidth=3,alpha=0.9,zorder=1)
ax10.scatter(Koscillators_library,smape_t_ax10,c='violet',marker='s',s=40,zorder=1)
ax10.fill_between(Koscillators_library,lower_t_ax10,upper_t_ax10,color='violet',alpha=0.15,linewidth=0,zorder=1)

ax10.set_xlim(1.7,5.3)
ax10.set_xticks([2,3,4,5])
ax10.set_ylim(-0.1,1.1)
ax10.set_yticks([0,1])
ax10.tick_params(direction='in')

ax10.set_ylabel('Inference inaccuracy (sMAPE)',font_y)
ax10.yaxis.set_label_coords(-0.1, 0.45)

ax10.set_title('500 data points',fontdict=font_title)
ax10.text(-0.18, 1,'j',ha='left', transform=ax10.transAxes,fontdict={'family':'Arial','size':24,'weight':'bold'})

fig.supxlabel(r'Basis function library with degree $k$',x=0.5, y=0, fontproperties=font_x)

plt.subplots_adjust(top=0.96, bottom=0.08, left=0.04, right=0.98, hspace=0.2, wspace=0.3)
plt.savefig('../figures/FIG2.pdf',format='pdf')
plt.savefig('/Users/zhugchzo/Desktop/3paper_fig/FIG2.png',format='png',dpi=600)