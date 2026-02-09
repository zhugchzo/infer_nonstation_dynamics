import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib import font_manager

font_x = {'family':'Arial','weight':'medium','size': 16}
font_y = {'family':'Arial','weight':'medium','size': 16}
font_title = {'family':'DejaVu Sans','weight':'normal','size': 15, 'style': 'italic'}

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15

# df for ax1
df_cusp_smape = pd.read_csv('../results/sMAPE/cusp_smape.csv')
# df for ax2
df_Koscillators_smape = pd.read_csv('../results/sMAPE/Koscillators_smape.csv')
# df for ax3
df_mitochondria_pred_ned = pd.read_csv('../results/NED/mitochondria/mitochondria_pred.csv')
df_mitochondria_pred_ned_t = pd.read_csv('../results/NED/mitochondria/mitochondria_pred_t.csv')
df_mitochondria_pred_ned_AIC = pd.read_csv('../results/NED/mitochondria/mitochondria_pred_AIC.csv')
# df for ax4
df_UAV_pred_ned = pd.read_csv('../results/NED/UAV/UAV_pred.csv')
df_UAV_pred_ned_t = pd.read_csv('../results/NED/UAV/UAV_pred_t.csv')
df_UAV_pred_ned_AIC = pd.read_csv('../results/NED/UAV/UAV_pred_AIC.csv')
# df for ax5
df_chick_220_pred_ned = pd.read_csv('../results/NED/chick_220/chick_220_pred.csv')
df_chick_220_pred_ned_t = pd.read_csv('../results/NED/chick_220/chick_220_pred_t.csv')
df_chick_220_pred_ned_AIC = pd.read_csv('../results/NED/chick_220/chick_220_pred_AIC.csv')
# df for ax6
df_chick_230_pred_ned = pd.read_csv('../results/NED/chick_230/chick_230_pred.csv')
df_chick_230_pred_ned_t = pd.read_csv('../results/NED/chick_230/chick_230_pred_t.csv')
df_chick_230_pred_ned_AIC = pd.read_csv('../results/NED/chick_230/chick_230_pred_AIC.csv')
# df for ax7
df_chick_335_pred_ned = pd.read_csv('../results/NED/chick_335/chick_335_pred.csv')
df_chick_335_pred_ned_t = pd.read_csv('../results/NED/chick_335/chick_335_pred_t.csv')
df_chick_335_pred_ned_AIC = pd.read_csv('../results/NED/chick_335/chick_335_pred_AIC.csv')
# df for ax8
df_fish_pred_ned = pd.read_csv('../results/NED/fish/fish_pred.csv')
df_fish_pred_ned_p = pd.read_csv('../results/NED/fish/fish_pred_p.csv')
df_fish_pred_ned_t = pd.read_csv('../results/NED/fish/fish_pred_t.csv')
df_fish_pred_ned_AIC = pd.read_csv('../results/NED/fish/fish_pred_AIC.csv')

fig, axs = plt.subplots(2, 4, figsize=(15,10.5))

ax1, ax2, ax3, ax4 = axs[0, 0], axs[0, 1], axs[0, 2], axs[0, 3]
ax5, ax6, ax7, ax8 = axs[1, 0], axs[1, 1], axs[1, 2], axs[1, 3]

# ax1
cusp_smape = df_cusp_smape[['mean_smape','mean_smape_ab','mean_smape_t','mean_smape_AIC']].values
cusp_lower = df_cusp_smape[['lower_smape','lower_smape_ab','lower_smape_t','lower_smape_AIC']].values
cusp_upper = df_cusp_smape[['upper_smape','upper_smape_ab','upper_smape_t','upper_smape_AIC']].values

len_ax1 = len(cusp_smape)
t_ax1 = np.arange(0,len_ax1)

smape_ax1 = cusp_smape[:,0]
smape_ab_ax1 = cusp_smape[:,1]
smape_t_ax1 = cusp_smape[:,2]
smape_AIC_ax1 = cusp_smape[:,3]

lower_ax1 = cusp_lower[:,0]
lower_ab_ax1 = cusp_lower[:,1]
lower_t_ax1 = cusp_lower[:,2]
lower_AIC_ax1 = cusp_lower[:,3]

upper_ax1 = cusp_upper[:,0]
upper_ab_ax1 = cusp_upper[:,1]
upper_t_ax1 = cusp_upper[:,2]
upper_AIC_ax1 = cusp_upper[:,3]

ax1.plot(t_ax1,smape_ax1,c='cornflowerblue',linewidth=3,alpha=0.9,zorder=2)
ax1.fill_between(t_ax1,lower_ax1,upper_ax1,color='cornflowerblue',alpha=0.15,linewidth=0,zorder=2)

ax1.plot(t_ax1,smape_ab_ax1,c='blueviolet',linewidth=3,alpha=0.9,zorder=1)
ax1.fill_between(t_ax1,lower_ab_ax1,upper_ab_ax1,color='blueviolet',alpha=0.15,linewidth=0,zorder=1)

ax1.plot(t_ax1,smape_t_ax1,c='violet',linewidth=3,alpha=0.9,zorder=1)
ax1.fill_between(t_ax1,lower_t_ax1,upper_t_ax1,color='violet',alpha=0.15,linewidth=0,zorder=1)

ax1.plot(t_ax1,smape_AIC_ax1,c='turquoise',linewidth=3,alpha=0.9,zorder=0)
ax1.fill_between(t_ax1,lower_AIC_ax1,upper_AIC_ax1,color='turquoise',alpha=0.15,linewidth=0,zorder=0)

ax1.set_xticks([0,500,1000])
ax1.set_ylim(-0.05,0.55)
ax1.set_yticks([0,0.5])
ax1.set_yticklabels(['0','0.5'])
ax1.tick_params(direction='in')

ax1.set_ylabel('Inference inaccuracy (sMAPE)',font_y)
ax1.yaxis.set_label_coords(-0.1, 0.45)

ax1.set_title('Cusp bifurcation',fontdict=font_title)
ax1.text(-0.18, 1,'a',ha='left', transform=ax1.transAxes,fontdict={'family':'Arial','size':30,'weight':'bold'})

# ax2
Koscillators_smape = df_Koscillators_smape[['mean_smape','mean_smape_p','mean_smape_t','mean_smape_AIC']].values
Koscillators_lower = df_Koscillators_smape[['lower_smape','lower_smape_p','lower_smape_t','lower_smape_AIC']].values
Koscillators_upper = df_Koscillators_smape[['upper_smape','upper_smape_p','upper_smape_t','upper_smape_AIC']].values

len_ax2 = len(Koscillators_smape)
t_ax2 = np.arange(0,len_ax2)

smape_ax2 = Koscillators_smape[:,0]
smape_p_ax2 = Koscillators_smape[:,1]
smape_t_ax2 = Koscillators_smape[:,2]
smape_AIC_ax2 = Koscillators_smape[:,3]

lower_ax2 = Koscillators_lower[:,0]
lower_p_ax2 = Koscillators_lower[:,1]
lower_t_ax2 = Koscillators_lower[:,2]
lower_AIC_ax2 = Koscillators_lower[:,3]

upper_ax2 = Koscillators_upper[:,0]
upper_p_ax2 = Koscillators_upper[:,1]
upper_t_ax2 = Koscillators_upper[:,2]
upper_AIC_ax2 = Koscillators_upper[:,3]

ax2.plot(t_ax2,smape_ax2,c='cornflowerblue',linewidth=3,alpha=0.9,zorder=2)
ax2.fill_between(t_ax2,lower_ax2,upper_ax2,color='cornflowerblue',alpha=0.15,linewidth=0,zorder=2)

ax2.plot(t_ax2,smape_p_ax2,c='blueviolet',linewidth=3,alpha=0.9,zorder=1)
ax2.fill_between(t_ax2,lower_p_ax2,upper_p_ax2,color='blueviolet',alpha=0.15,linewidth=0,zorder=1)

ax2.plot(t_ax2,smape_t_ax2,c='violet',linewidth=3,alpha=0.9,zorder=1)
ax2.fill_between(t_ax2,lower_t_ax2,upper_t_ax2,color='violet',alpha=0.15,linewidth=0,zorder=1)

ax2.plot(t_ax2,smape_AIC_ax2,c='turquoise',linewidth=3,alpha=0.9,zorder=0)
ax2.fill_between(t_ax2,lower_AIC_ax2,upper_AIC_ax2,color='turquoise',alpha=0.15,linewidth=0,zorder=0)

ax2.set_xticks([0,500,1000])
ax2.set_ylim(-0.055,0.605)
ax2.set_yticks([0,0.55])
ax2.set_yticklabels(['0','0.55'])
ax2.tick_params(direction='in')

ax2.set_ylabel('Inference inaccuracy (sMAPE)',font_y)
ax2.yaxis.set_label_coords(-0.1, 0.45)

ax2.set_title('Kuramoto oscillators',fontdict=font_title)
ax2.text(-0.18, 1,'b',ha='left', transform=ax2.transAxes,fontdict={'family':'Arial','size':30,'weight':'bold'})

# ax3
mitochondria_pred_ned = df_mitochondria_pred_ned[['normal_Eucli_d']].values
mitochondria_pred_ned_t = df_mitochondria_pred_ned_t[['normal_Eucli_d']].values
mitochondria_pred_ned_AIC = df_mitochondria_pred_ned_AIC[['normal_Eucli_d']].values

len_ax3 = len(mitochondria_pred_ned)
t_ax3 = np.arange(0,len_ax3)
len_ax3_t = len(mitochondria_pred_ned_t)
t_ax3_t = np.arange(0,len_ax3_t)
len_ax3_AIC = len(mitochondria_pred_ned_AIC)
t_ax3_AIC = np.arange(0,len_ax3_AIC)

ned_ax3 = mitochondria_pred_ned[:,0]
ned_t_ax3 = mitochondria_pred_ned_t[:,0]
ned_AIC_ax3 = mitochondria_pred_ned_AIC[:,0]

ax3.plot(t_ax3,ned_ax3,c='cornflowerblue',linewidth=3,alpha=0.9,zorder=3)
ax3.scatter(t_ax3[-1],ned_ax3[-1],c='crimson',marker='*',zorder=3,s=150,alpha=0.9)
ax3.plot(t_ax3_t,ned_t_ax3,c='violet',linewidth=3,alpha=0.9,zorder=2)
ax3.plot(t_ax3_AIC,ned_AIC_ax3,c='turquoise',linewidth=3,alpha=0.9,zorder=1)
ax3.scatter(t_ax3_AIC[-1],ned_AIC_ax3[-1],c='crimson',marker='*',zorder=3,s=150,alpha=0.9)
ax3.axvline(x=t_ax3[-1], color='crimson', linestyle='--', linewidth=1.5, zorder=0,alpha=0.9)

ax3.annotate('collapse',xy=(68, 0),xytext=(30, 0),fontsize=16,color='crimson',
             arrowprops=dict(arrowstyle='->',color='crimson',lw=1))

ax3.set_xlim(-5,95)
ax3.set_xticks([0,45,90])

ax3.set_ylim(-0.0026,0.0286)
ax3.set_yticks([0,0.025])
ax3.set_yticklabels(['0','0.025'])
ax3.tick_params(direction='in')

ax3.set_ylabel('Prediction inaccuracy (NED)',font_y)
ax3.yaxis.set_label_coords(-0.1, 0.45)

ax3.set_title('Cellular energy depletion',fontdict=font_title)
ax3.text(-0.18, 1,'c',ha='left', transform=ax3.transAxes,fontdict={'family':'Arial','size':30,'weight':'bold'})

# ax4
UAV_pred_ned = df_UAV_pred_ned[['normal_Eucli_d']].values
UAV_pred_ned_t = df_UAV_pred_ned_t[['normal_Eucli_d']].values
UAV_pred_ned_AIC = df_UAV_pred_ned_AIC[['normal_Eucli_d']].values

len_ax4 = len(UAV_pred_ned)
t_ax4 = np.arange(0,len_ax4)
len_ax4_t = len(UAV_pred_ned_t)
t_ax4_t = np.arange(0,len_ax4_t)
len_ax4_AIC = len(UAV_pred_ned_AIC)
t_ax4_AIC = np.arange(0,len_ax4_AIC)

ned_ax4 = UAV_pred_ned[:,0]
ned_t_ax4 = UAV_pred_ned_t[:,0]
ned_AIC_ax4 = UAV_pred_ned_AIC[:,0]

ax4.plot(t_ax4,ned_ax4,c='cornflowerblue',linewidth=3,alpha=0.9,zorder=2)
ax4.plot(t_ax4_t,ned_t_ax4,c='violet',linewidth=3,alpha=0.9,zorder=1)
ax4.scatter(t_ax4_t[-1],ned_t_ax4[-1],c='crimson',marker=r'$\times$',zorder=1,s=120,alpha=0.9)
ax4.plot(t_ax4_AIC,ned_AIC_ax4,c='turquoise',linewidth=3,alpha=0.9,zorder=0)

ax4.set_xticks([0,55,110])

ax4.set_ylim(-0.007,0.077)
ax4.set_yticks([0,0.07])
ax4.set_yticklabels(['0','0.07'])
ax4.tick_params(direction='in')

ax4.set_ylabel('Prediction inaccuracy (NED)',font_y)
ax4.yaxis.set_label_coords(-0.1, 0.5)
ax4.set_title('UAV autonomous flight',fontdict=font_title)
ax4.text(-0.18, 1,'d',ha='left', transform=ax4.transAxes,fontdict={'family':'Arial','size':30,'weight':'bold'})

# ax5
chick_220_pred_ned = df_chick_220_pred_ned[['normal_Eucli_d']].values
chick_220_pred_ned_t = df_chick_220_pred_ned_t[['normal_Eucli_d']].values
chick_220_pred_ned_AIC = df_chick_220_pred_ned_AIC[['normal_Eucli_d']].values

len_ax5 = len(chick_220_pred_ned)
t_ax5 = np.arange(0,len_ax5)
len_ax5_t = len(chick_220_pred_ned_t)
t_ax5_t = np.arange(0,len_ax5_t)
len_ax5_AIC = len(chick_220_pred_ned_AIC)
t_ax5_AIC = np.arange(0,len_ax5_AIC)

ned_ax5 = chick_220_pred_ned[:,0]
ned_t_ax5 = chick_220_pred_ned_t[:,0]
ned_AIC_ax5 = chick_220_pred_ned_AIC[:,0]

ax5.plot(t_ax5,ned_ax5,c='cornflowerblue',linewidth=3,alpha=0.9,zorder=3)
ax5.scatter(t_ax5[-1],ned_ax5[-1],c='chocolate',marker='*',zorder=3,s=150)
ax5.plot(t_ax5_t,ned_t_ax5,c='violet',linewidth=3,alpha=0.9,zorder=2)
ax5.scatter(t_ax5_t[-1],ned_t_ax5[-1],c='crimson',marker=r'$\times$',zorder=3,s=150,alpha=0.9)
ax5.plot(t_ax5_AIC,ned_AIC_ax5,c='turquoise',linewidth=3,alpha=0.9,zorder=1)
ax5.axvline(x=190-150, color='chocolate', linestyle='--', linewidth=1.5, zorder=0,alpha=0.9)

ax5.annotate('period-2',xy=(41, 0),xytext=(45, 0.01),fontsize=16,color='chocolate',
             arrowprops=dict(arrowstyle='->',color='chocolate',lw=1))

ax5.set_xlim(-4,75)
ax5.set_xticks([0,35,70])

ax5.set_ylim(-0.015,0.165)
ax5.set_yticks([0,0.15])
ax5.set_yticklabels(['0','0.15'])
ax5.tick_params(direction='in')

ax5.set_xlabel('Timepoints',font_x)
ax5.set_ylabel('Prediction inaccuracy (NED)',font_y)
ax5.yaxis.set_label_coords(-0.1, 0.5)

ax5.set_title('Beating chick-heart (I)',fontdict=font_title)
ax5.text(-0.18, 1,'e',ha='left', transform=ax5.transAxes,fontdict={'family':'Arial','size':30,'weight':'bold'})

# ax6
chick_230_pred_ned = df_chick_230_pred_ned[['normal_Eucli_d']].values
chick_230_pred_ned_t = df_chick_230_pred_ned_t[['normal_Eucli_d']].values
chick_230_pred_ned_AIC = df_chick_230_pred_ned_AIC[['normal_Eucli_d']].values

len_ax6 = len(chick_230_pred_ned)
t_ax6 = np.arange(0,len_ax6)
len_ax6_t = len(chick_230_pred_ned_t)
t_ax6_t = np.arange(0,len_ax6_t)
len_ax6_AIC = len(chick_230_pred_ned_AIC)
t_ax6_AIC = np.arange(0,len_ax6_AIC)

ned_ax6 = chick_230_pred_ned[:,0]
ned_t_ax6 = chick_230_pred_ned_t[:,0]
ned_AIC_ax6 = chick_230_pred_ned_AIC[:,0]

ax6.plot(t_ax6,ned_ax6,c='cornflowerblue',linewidth=3,alpha=0.9,zorder=3)
ax6.scatter(t_ax6[-1],ned_ax6[-1],c='chocolate',marker='*',zorder=3,s=150)
ax6.plot(t_ax6_t,ned_t_ax6,c='violet',linewidth=3,alpha=0.9,zorder=2)
ax6.scatter(t_ax6_t[-1],ned_t_ax6[-1],c='chocolate',marker='*',zorder=3,s=150)
ax6.plot(t_ax6_AIC,ned_AIC_ax6,c='turquoise',linewidth=3,alpha=0.9,zorder=1)
ax6.axvline(x=194-150, color='chocolate', linestyle='--', linewidth=1.5, zorder=0,alpha=0.9)

ax6.annotate('period-2',xy=(45, 0),xytext=(50, 0.005),fontsize=16,color='chocolate',
             arrowprops=dict(arrowstyle='->',color='chocolate',lw=1))

ax6.set_xlim(-4.5,85)
ax6.set_xticks([0,40,80])

ax6.set_ylim(-0.008,0.088)
ax6.set_yticks([0,0.08])
ax6.set_yticklabels(['0','0.08'])
ax6.tick_params(direction='in')

ax6.set_xlabel('Timepoints',font_x)
ax6.set_ylabel('Prediction inaccuracy (NED)',font_y)
ax6.yaxis.set_label_coords(-0.1, 0.5)

ax6.set_title('Beating chick-heart (II)',fontdict=font_title)
ax6.text(-0.18, 1,'f',ha='left', transform=ax6.transAxes,fontdict={'family':'Arial','size':30,'weight':'bold'})

# ax7
chick_335_pred_ned = df_chick_335_pred_ned[['normal_Eucli_d']].values
chick_335_pred_ned_t = df_chick_335_pred_ned_t[['normal_Eucli_d']].values
chick_335_pred_ned_AIC = df_chick_335_pred_ned_AIC[['normal_Eucli_d']].values

len_ax7 = len(chick_335_pred_ned)
t_ax7 = np.arange(0,len_ax7)
len_ax7_t = len(chick_335_pred_ned_t)
t_ax7_t = np.arange(0,len_ax7_t)
len_ax7_AIC = len(chick_335_pred_ned_AIC)
t_ax7_AIC = np.arange(0,len_ax7_AIC)

ned_ax7 = chick_335_pred_ned[:,0]
ned_t_ax7 = chick_335_pred_ned_t[:,0]
ned_AIC_ax7 = chick_335_pred_ned_AIC[:,0]

ax7.plot(t_ax7,ned_ax7,c='cornflowerblue',linewidth=3,alpha=0.9,zorder=3)
ax7.scatter(t_ax7[-1],ned_ax7[-1],c='chocolate',marker='*',zorder=3,s=150)
ax7.plot(t_ax7_t,ned_t_ax7,c='violet',linewidth=3,alpha=0.9,zorder=2)
ax7.scatter(t_ax7_t[-1],ned_t_ax7[-1],c='crimson',marker=r'$\times$',zorder=3,s=150,alpha=0.9)
ax7.plot(t_ax7_AIC,ned_AIC_ax7,c='turquoise',linewidth=3,alpha=0.9,zorder=1)
ax7.axvline(x=296-200, color='chocolate', linestyle='--', linewidth=1.5, zorder=0,alpha=0.9)

ax7.annotate('period-2',xy=(94, 0),xytext=(40, 0.003),fontsize=16,color='chocolate',
             arrowprops=dict(arrowstyle='->',color='chocolate',lw=1))

ax7.set_xlim(-7,135)
ax7.set_xticks([0,65,130])

ax7.set_ylim(-0.022,0.242)
ax7.set_yticks([0,0.22])
ax7.set_yticklabels(['0','0.22'])
ax7.tick_params(direction='in')

ax7.set_xlabel('Timepoints',font_x)
ax7.set_ylabel('Prediction inaccuracy (NED)',font_y)
ax7.yaxis.set_label_coords(-0.1, 0.5)

ax7.set_title('Beating chick-heart (III)',fontdict=font_title)
ax7.text(-0.18, 1,'g',ha='left', transform=ax7.transAxes,fontdict={'family':'Arial','size':30,'weight':'bold'})

# ax8
fish_pred_ned = df_fish_pred_ned[['normal_Eucli_d']].values
fish_pred_ned_p = df_fish_pred_ned_p[['normal_Eucli_d']].values
fish_pred_ned_t = df_fish_pred_ned_t[['normal_Eucli_d']].values
fish_pred_ned_AIC = df_fish_pred_ned_AIC[['normal_Eucli_d']].values

len_ax8 = len(fish_pred_ned)
t_ax8 = np.arange(0,len_ax8)
len_ax8_p = len(fish_pred_ned_p)
t_ax8_p = np.arange(0,len_ax8_p)
len_ax8_t = len(fish_pred_ned_t)
t_ax8_t = np.arange(0,len_ax8_t)
len_ax8_AIC = len(fish_pred_ned_AIC)
t_ax8_AIC = np.arange(0,len_ax8_AIC)

ned_ax8 = fish_pred_ned[:,0]
ned_p_ax8 = fish_pred_ned_p[:,0]
ned_t_ax8 = fish_pred_ned_t[:,0]
ned_AIC_ax8 = fish_pred_ned_AIC[:,0]

ax8.plot(t_ax8,ned_ax8,c='cornflowerblue',linewidth=3,alpha=0.9,zorder=2)
ax8.plot(t_ax8_p,ned_p_ax8,c='blueviolet',linewidth=3,alpha=0.9,zorder=1)
ax8.plot(t_ax8_t,ned_t_ax8,c='violet',linewidth=3,alpha=0.9,zorder=1)
ax8.plot(t_ax8_AIC,ned_AIC_ax8,c='turquoise',linewidth=3,alpha=0.9,zorder=0)

ax8.set_xticks([0,15,30])
ax8.set_xlim(-3,33)
ax8.set_ylim(0.35,1)
ax8.set_yticks([0.4,0.95])
ax8.set_yticklabels(['0.4','0.95'])
ax8.tick_params(direction='in')

ax8.set_xlabel('Timepoints',font_x)
ax8.set_ylabel('Prediction inaccuracy (NED)',font_y)
ax8.yaxis.set_label_coords(-0.1, 0.5)
ax8.set_title('Fish community',fontdict=font_title)
ax8.text(-0.18, 1,'h',ha='left', transform=ax8.transAxes,fontdict={'family':'Arial','size':30,'weight':'bold'})

# legend
legend_v = mlines.Line2D([], [], color='cornflowerblue', marker='none', linestyle='-', linewidth=3,alpha=0.9)
legend_p = mlines.Line2D([], [], color='blueviolet', marker='none', linestyle='-', linewidth=3,alpha=0.9)
legend_t = mlines.Line2D([], [], color='violet', marker='none', linestyle='-', linewidth=3,alpha=0.9)
legend_AIC = mlines.Line2D([], [], color='turquoise', marker='none', linestyle='-', linewidth=3,alpha=0.9)
legend_tclsp = mlines.Line2D([], [], color='crimson', marker='*', linestyle='none', markersize=10,alpha=0.9)
legend_pd2 = mlines.Line2D([], [], color='chocolate', marker='*', linestyle='none', markersize=10,alpha=1)
legend_fclsp = mlines.Line2D([], [], color='crimson', marker=r'$\times$', linestyle='none', markersize=8,alpha=0.9)

fig.legend(
    handles=[legend_v, legend_p, legend_t, legend_AIC],
    labels=['Optimal driving variable', 'Forcing parameters', 'Time variable', r'Use AIC instead of $\epsilon$AIC'],
    loc='upper center',
    bbox_to_anchor=(0.5, 1.01),
    ncol=4,
    frameon=False,
    markerscale=1.5,
    prop=font_manager.FontProperties(family='Arial Unicode MS', size=16)
)

fig.legend(
    handles=[legend_tclsp, legend_pd2, legend_fclsp],
    labels=['Equations collapse (match)', 'Equations enter period-2 (match)', 'Equations collapse (mismatch)'],
    loc='upper center',
    bbox_to_anchor=(0.5, 0.97),
    ncol=3,
    frameon=False,
    markerscale=1.5,
    prop=font_manager.FontProperties(family='Arial Unicode MS', size=16)
)

plt.subplots_adjust(top=0.86, bottom=0.05, left=0.04, right=0.98, hspace=0.23, wspace=0.3)
plt.savefig('../figures/FIG6.pdf',format='pdf')
plt.savefig('/Users/zhugchzo/Desktop/3paper_fig/FIG6.png',format='png',dpi=600)
# plt.show()