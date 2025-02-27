import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

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
df_mitochondria_gen_ned = pd.read_csv('../results/NED/mitochondria/mitochondria_gen.csv')
df_mitochondria_gen_ned_t = pd.read_csv('../results/NED/mitochondria/mitochondria_gen_t.csv')
df_mitochondria_gen_ned_AIC = pd.read_csv('../results/NED/mitochondria/mitochondria_gen_AIC.csv')
# df for ax4
df_UAV_gen_ned = pd.read_csv('../results/NED/UAV/UAV_gen.csv')
df_UAV_gen_ned_t = pd.read_csv('../results/NED/UAV/UAV_gen_t.csv')
df_UAV_gen_ned_AIC = pd.read_csv('../results/NED/UAV/UAV_gen_AIC.csv')
# df for ax6
df_cusp_ned = pd.read_csv('../results/NED/cusp_NED.csv')
# df for ax7
df_Koscillators_ned = pd.read_csv('../results/NED/Koscillators_NED.csv')
# df for ax8
df_mitochondria_pred_ned = pd.read_csv('../results/NED/mitochondria/mitochondria_pred.csv')
df_mitochondria_pred_ned_t = pd.read_csv('../results/NED/mitochondria/mitochondria_pred_t.csv')
df_mitochondria_pred_ned_AIC = pd.read_csv('../results/NED/mitochondria/mitochondria_pred_AIC.csv')
# df for ax9
df_UAV_pred_ned = pd.read_csv('../results/NED/UAV/UAV_pred.csv')
df_UAV_pred_ned_t = pd.read_csv('../results/NED/UAV/UAV_pred_t.csv')
df_UAV_pred_ned_AIC = pd.read_csv('../results/NED/UAV/UAV_pred_AIC.csv')
# df for ax10
df_fish_pred_ned = pd.read_csv('../results/NED/fish/fish_pred.csv')
df_fish_pred_ned_p = pd.read_csv('../results/NED/fish/fish_pred_p.csv')
df_fish_pred_ned_t = pd.read_csv('../results/NED/fish/fish_pred_t.csv')
df_fish_pred_ned_AIC = pd.read_csv('../results/NED/fish/fish_pred_AIC.csv')

fig, axs = plt.subplots(2, 5, figsize=(15,9))

ax1, ax2, ax3, ax4, ax5 = axs[0, 0], axs[0, 1], axs[0, 2], axs[0, 3], axs[0, 4]
ax6, ax7, ax8, ax9, ax10 = axs[1, 0], axs[1, 1], axs[1, 2], axs[1, 3], axs[1, 4]

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
ax1.text(-0.18, 1,'a',ha='left', transform=ax1.transAxes,fontdict={'family':'Arial','size':24,'weight':'bold'})

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

ax2.set_xticks([0,450,900])
ax2.set_xticklabels(['100','550','1000'])
ax2.set_ylim(-0.007,0.077)
ax2.set_yticks([0,0.07])
ax2.set_yticklabels(['0','0.07'])
ax2.tick_params(direction='in')

ax2.set_ylabel('Inference inaccuracy (sMAPE)',font_y)
ax2.yaxis.set_label_coords(-0.1, 0.45)

ax2.set_title('Kuramoto oscillators',fontdict=font_title)
ax2.text(-0.18, 1,'b',ha='left', transform=ax2.transAxes,fontdict={'family':'Arial','size':24,'weight':'bold'})

# ax3
mitochondria_gen_ned = df_mitochondria_gen_ned[['normal_Eucli_d']].values
mitochondria_gen_ned_t = df_mitochondria_gen_ned_t[['normal_Eucli_d']].values
mitochondria_gen_ned_AIC = df_mitochondria_gen_ned_AIC[['normal_Eucli_d']].values

len_ax3 = len(mitochondria_gen_ned)
t_ax3 = np.arange(0,len_ax3)
len_ax3_t = len(mitochondria_gen_ned_t)
t_ax3_t = np.arange(0,len_ax3_t)
len_ax3_AIC = len(mitochondria_gen_ned_AIC)
t_ax3_AIC = np.arange(0,len_ax3_AIC)

ned_ax3 = mitochondria_gen_ned[:,0]
ned_t_ax3 = mitochondria_gen_ned_t[:,0]
ned_AIC_ax3 = mitochondria_gen_ned_AIC[:,0]

ax3.plot(t_ax3,ned_ax3,c='cornflowerblue',linewidth=3,alpha=0.9,zorder=2)
ax3.scatter(t_ax3[-1],ned_ax3[-1],c='crimson',marker='*',zorder=2,s=150,alpha=0.9)
ax3.plot(t_ax3_t,ned_t_ax3,c='violet',linewidth=3,alpha=0.9,zorder=1)
ax3.plot(t_ax3_AIC,ned_AIC_ax3,c='turquoise',linewidth=3,alpha=0.9,zorder=0)
ax3.scatter(t_ax3_AIC[-1],ned_AIC_ax3[-1],c='crimson',marker='*',zorder=1,s=150,alpha=0.9)
ax3.axvline(x=t_ax3[-1], color='crimson', linestyle='--', linewidth=1.5, zorder=0)

ax3.annotate('collapse',xy=(240, 0),xytext=(120, 0.001),fontsize=16,color='crimson',
             arrowprops=dict(arrowstyle='->',color='crimson',lw=1))

ax3.set_xticks([0,130,260])

ax3.set_ylim(-0.002,0.022)
ax3.set_yticks([0,0.02])
ax3.set_yticklabels(['0','0.02'])
ax3.tick_params(direction='in')

ax3.set_ylabel('Inference inaccuracy (NED)',font_y)
ax3.yaxis.set_label_coords(-0.1, 0.5)
ax3.set_title('Cellular energy depletion',fontdict=font_title)
ax3.text(-0.18, 1,'c',ha='left', transform=ax3.transAxes,fontdict={'family':'Arial','size':24,'weight':'bold'})

# ax4
UAV_gen_ned = df_UAV_gen_ned[['normal_Eucli_d']].values
UAV_gen_ned_t = df_UAV_gen_ned_t[['normal_Eucli_d']].values
UAV_gen_ned_AIC = df_UAV_gen_ned_AIC[['normal_Eucli_d']].values

len_ax4 = len(UAV_gen_ned)
t_ax4 = np.arange(0,len_ax4)
len_ax4_t = len(UAV_gen_ned_t)
t_ax4_t = np.arange(0,len_ax4_t)
len_ax4_AIC = len(UAV_gen_ned_AIC)
t_ax4_AIC = np.arange(0,len_ax4_AIC)

ned_ax4 = UAV_gen_ned[:,0]
ned_t_ax4 = UAV_gen_ned_t[:,0]
ned_AIC_ax4 = UAV_gen_ned_AIC[:,0]

ax4.plot(t_ax4,ned_ax4,c='cornflowerblue',linewidth=3,alpha=0.9,zorder=2)
ax4.plot(t_ax4_t,ned_t_ax4,c='violet',linewidth=3,alpha=0.9,zorder=1)
ax4.scatter(t_ax4_t[-1],ned_t_ax4[-1],c='crimson',marker=r'$\times$',zorder=1,s=120,alpha=0.9)
ax4.plot(t_ax4_AIC,ned_AIC_ax4,c='turquoise',linewidth=3,alpha=0.9,zorder=0)

ax4.set_xticks([0,135,270])

ax4.set_ylim(-0.06,0.66)
ax4.set_yticks([0,0.6])
ax4.set_yticklabels(['0','0.6'])
ax4.tick_params(direction='in')

ax4.set_ylabel('Inference inaccuracy (NED)',font_y)
ax4.yaxis.set_label_coords(-0.1, 0.5)
ax4.set_title('UAV obstacle avoidance',fontdict=font_title)
ax4.text(-0.18, 1,'d',ha='left', transform=ax4.transAxes,fontdict={'family':'Arial','size':24,'weight':'bold'})

# legend
ax5.axis('off')
legend_v = mlines.Line2D([], [], color='cornflowerblue', marker='none', linestyle='-', linewidth=3,alpha=0.9)
legend_p = mlines.Line2D([], [], color='blueviolet', marker='none', linestyle='-', linewidth=3,alpha=0.9)
legend_t = mlines.Line2D([], [], color='violet', marker='none', linestyle='-', linewidth=3,alpha=0.9)
legend_AIC = mlines.Line2D([], [], color='turquoise', marker='none', linestyle='-', linewidth=3,alpha=0.9)
legend_tclsp = mlines.Line2D([], [], color='crimson', marker='*', linestyle='none', markersize=10,alpha=0.9)
legend_fclsp = mlines.Line2D([], [], color='crimson', marker=r'$\times$', linestyle='none', markersize=8,alpha=0.9)
ax5.legend(handles=[legend_v,legend_p,legend_t,legend_AIC,legend_tclsp,legend_fclsp],
           labels=['Virtual variable','Forcing parameters','Time variable',r'Use AIC instead of $\epsilon$AIC','Equations collapse\n       (match)','Equations collapse\n     (mismatch)'],
           loc='center',bbox_to_anchor=(0.425, 0.5),frameon=False, markerscale=1.5, prop={'size':16})

# ax6
cusp_ned = df_cusp_ned[['mean_NED','mean_NED_ab','mean_NED_t','mean_NED_AIC']].values
cusp_lower = df_cusp_ned[['lower_NED','lower_NED_ab','lower_NED_t','lower_NED_AIC']].values
cusp_upper = df_cusp_ned[['upper_NED','upper_NED_ab','upper_NED_t','upper_NED_AIC']].values

len_ax6 = len(cusp_ned)
t_ax6 = np.arange(0,len_ax6)

ned_ax6 = cusp_ned[:,0]
ned_ab_ax6 = cusp_ned[:,1]
ned_t_ax6 = cusp_ned[:,2]
ned_AIC_ax6 = cusp_ned[:,3]

lower_ax6 = cusp_lower[:,0]
lower_ab_ax6 = cusp_lower[:,1]
lower_t_ax6 = cusp_lower[:,2]
lower_AIC_ax6 = cusp_lower[:,3]

upper_ax6 = cusp_upper[:,0]
upper_ab_ax6 = cusp_upper[:,1]
upper_t_ax6 = cusp_upper[:,2]
upper_AIC_ax6 = cusp_upper[:,3]

ax6.plot(t_ax6,ned_ax6,c='cornflowerblue',linewidth=3,alpha=0.9,zorder=2)
ax6.fill_between(t_ax6,lower_ax6,upper_ax6,color='cornflowerblue',alpha=0.15,linewidth=0,zorder=2)

ax6.plot(t_ax6,ned_ab_ax6,c='blueviolet',linewidth=3,alpha=0.9,zorder=1)
ax6.fill_between(t_ax6,lower_ab_ax6,upper_ab_ax6,color='blueviolet',alpha=0.15,linewidth=0,zorder=1)

ax6.plot(t_ax6,ned_t_ax6,c='violet',linewidth=3,alpha=0.9,zorder=1)
ax6.fill_between(t_ax6,lower_t_ax6,upper_t_ax6,color='violet',alpha=0.15,linewidth=0,zorder=1)

ax6.plot(t_ax6,ned_AIC_ax6,c='turquoise',linewidth=3,alpha=0.9,zorder=0)
ax6.fill_between(t_ax6,lower_AIC_ax6,upper_AIC_ax6,color='turquoise',alpha=0.15,linewidth=0,zorder=0)

ax6.set_xticks([0,250,500])
ax6.set_ylim(-0.035,0.385)
ax6.set_yticks([0,0.35])
ax6.set_yticklabels(['0','0.35'])
ax6.tick_params(direction='in')

ax6.set_xlabel('Timepoints',font_x)
ax6.set_ylabel('Prediction inaccuracy (NED)',font_y)
ax6.yaxis.set_label_coords(-0.1, 0.5)

ax6.set_title('Cusp bifurcation',fontdict=font_title)
ax6.text(-0.18, 1,'e',ha='left', transform=ax6.transAxes,fontdict={'family':'Arial','size':24,'weight':'bold'})

# ax7
Koscillators_ned = df_Koscillators_ned[['mean_NED','mean_NED_p','mean_NED_t','mean_NED_AIC']].values
Koscillators_lower = df_Koscillators_ned[['lower_NED','lower_NED_p','lower_NED_t','lower_NED_AIC']].values
Koscillators_upper = df_Koscillators_ned[['upper_NED','upper_NED_p','upper_NED_t','upper_NED_AIC']].values

len_ax7 = len(Koscillators_ned)
t_ax7 = np.arange(0,len_ax7)

ned_ax7 = Koscillators_ned[:,0]
ned_ab_ax7 = Koscillators_ned[:,1]
ned_t_ax7 = Koscillators_ned[:,2]
ned_AIC_ax7 = Koscillators_ned[:,3]

lower_ax7 = Koscillators_lower[:,0]
lower_ab_ax7 = Koscillators_lower[:,1]
lower_t_ax7 = Koscillators_lower[:,2]
lower_AIC_ax7 = Koscillators_lower[:,3]

upper_ax7 = Koscillators_upper[:,0]
upper_ab_ax7 = Koscillators_upper[:,1]
upper_t_ax7 = Koscillators_upper[:,2]
upper_AIC_ax7 = Koscillators_upper[:,3]

ax7.plot(t_ax7,ned_ax7,c='cornflowerblue',linewidth=3,alpha=0.9,zorder=2)
ax7.fill_between(t_ax7,lower_ax7,upper_ax7,color='cornflowerblue',alpha=0.15,linewidth=0,zorder=2)

ax7.plot(t_ax7,ned_ab_ax7,c='blueviolet',linewidth=3,alpha=0.9,zorder=1)
ax7.fill_between(t_ax7,lower_ab_ax7,upper_ab_ax7,color='blueviolet',alpha=0.15,linewidth=0,zorder=1)

ax7.plot(t_ax7,ned_t_ax7,c='violet',linewidth=3,alpha=0.9,zorder=1)
ax7.fill_between(t_ax7,lower_t_ax7,upper_t_ax7,color='violet',alpha=0.15,linewidth=0,zorder=1)

ax7.plot(t_ax7,ned_AIC_ax7,c='turquoise',linewidth=3,alpha=0.9,zorder=0)
ax7.fill_between(t_ax7,lower_AIC_ax7,upper_AIC_ax7,color='turquoise',alpha=0.15,linewidth=0,zorder=0)

ax7.set_xticks([0,350,700])
ax7.set_ylim(-8e-6,4.5e-5)
ax7.set_yticks([0,4e-5])
ax7.set_yticklabels(['0','4e-5'])
ax7.tick_params(direction='in')

ax7.set_xlabel('Timepoints',font_x)
ax7.set_ylabel(r'Prediction inaccuracy ($\Delta$NED)',font_y)
ax7.yaxis.set_label_coords(-0.1, 0.45)

ax7.set_title('Kuramoto oscillators',fontdict=font_title)
ax7.text(-0.18, 1,'f',ha='left', transform=ax7.transAxes,fontdict={'family':'Arial','size':24,'weight':'bold'})

# ax8
mitochondria_pred_ned = df_mitochondria_pred_ned[['normal_Eucli_d']].values
mitochondria_pred_ned_t = df_mitochondria_pred_ned_t[['normal_Eucli_d']].values
mitochondria_pred_ned_AIC = df_mitochondria_pred_ned_AIC[['normal_Eucli_d']].values

len_ax8 = len(mitochondria_pred_ned)
t_ax8 = np.arange(0,len_ax8)
len_ax8_t = len(mitochondria_pred_ned_t)
t_ax8_t = np.arange(0,len_ax8_t)
len_ax8_AIC = len(mitochondria_pred_ned_AIC)
t_ax8_AIC = np.arange(0,len_ax8_AIC)

ned_ax8 = mitochondria_pred_ned[:,0]
ned_t_ax8 = mitochondria_pred_ned_t[:,0]
ned_AIC_ax8 = mitochondria_pred_ned_AIC[:,0]

ax8.plot(t_ax8,ned_ax8,c='cornflowerblue',linewidth=3,alpha=0.9,zorder=2)
ax8.scatter(t_ax8[-1],ned_ax8[-1],c='crimson',marker='*',zorder=2,s=150,alpha=0.9)
ax8.plot(t_ax8_t,ned_t_ax8,c='violet',linewidth=3,alpha=0.9,zorder=1)
ax8.plot(t_ax8_AIC,ned_AIC_ax8,c='turquoise',linewidth=3,alpha=0.9,zorder=0)
ax8.scatter(t_ax8_AIC[-1],ned_AIC_ax8[-1],c='crimson',marker='*',zorder=1,s=150,alpha=0.9)
ax8.axvline(x=t_ax8[-1], color='crimson', linestyle='--', linewidth=1.5, zorder=0,alpha=0.9)

ax8.annotate('collapse',xy=(68, 0),xytext=(21, 0),fontsize=16,color='crimson',
             arrowprops=dict(arrowstyle='->',color='crimson',lw=1))

ax8.set_xlim(-5,95)
ax8.set_xticks([0,45,90])

ax8.set_ylim(-0.0026,0.0286)
ax8.set_yticks([0,0.025])
ax8.set_yticklabels(['0','0.025'])
ax8.tick_params(direction='in')

ax8.set_xlabel('Timepoints',font_x)
ax8.set_ylabel('Prediction inaccuracy (NED)',font_y)
ax8.yaxis.set_label_coords(-0.1, 0.45)

ax8.set_title('Cellular energy depletion',fontdict=font_title)
ax8.text(-0.18, 1,'g',ha='left', transform=ax8.transAxes,fontdict={'family':'Arial','size':24,'weight':'bold'})

# ax9
UAV_pred_ned = df_UAV_pred_ned[['normal_Eucli_d']].values
UAV_pred_ned_t = df_UAV_pred_ned_t[['normal_Eucli_d']].values
UAV_pred_ned_AIC = df_UAV_pred_ned_AIC[['normal_Eucli_d']].values

len_ax9 = len(UAV_pred_ned)
t_ax9 = np.arange(0,len_ax9)
len_ax9_t = len(UAV_pred_ned_t)
t_ax9_t = np.arange(0,len_ax9_t)
len_ax9_AIC = len(UAV_pred_ned_AIC)
t_ax9_AIC = np.arange(0,len_ax9_AIC)

ned_ax9 = UAV_pred_ned[:,0]
ned_t_ax9 = UAV_pred_ned_t[:,0]
ned_AIC_ax9 = UAV_pred_ned_AIC[:,0]

ax9.plot(t_ax9,ned_ax9,c='cornflowerblue',linewidth=3,alpha=0.9,zorder=2)
ax9.plot(t_ax9_t,ned_t_ax9,c='violet',linewidth=3,alpha=0.9,zorder=1)
ax9.scatter(t_ax9_t[-1],ned_t_ax9[-1],c='crimson',marker=r'$\times$',zorder=1,s=120,alpha=0.9)
ax9.plot(t_ax9_AIC,ned_AIC_ax9,c='turquoise',linewidth=3,alpha=0.9,zorder=0)

ax9.set_xticks([0,55,110])

ax9.set_ylim(-0.007,0.077)
ax9.set_yticks([0,0.07])
ax9.set_yticklabels(['0','0.07'])
ax9.tick_params(direction='in')

ax9.set_xlabel('Timepoints',font_x)
ax9.set_ylabel('Prediction inaccuracy (NED)',font_y)
ax9.yaxis.set_label_coords(-0.1, 0.45)
ax9.set_title('UAV obstacle avoidance',fontdict=font_title)
ax9.text(-0.18, 1,'h',ha='left', transform=ax9.transAxes,fontdict={'family':'Arial','size':24,'weight':'bold'})

# ax10
fish_pred_ned = df_fish_pred_ned[['normal_Eucli_d']].values
fish_pred_ned_p = df_fish_pred_ned_p[['normal_Eucli_d']].values
fish_pred_ned_t = df_fish_pred_ned_t[['normal_Eucli_d']].values
fish_pred_ned_AIC = df_fish_pred_ned_AIC[['normal_Eucli_d']].values

len_ax10 = len(fish_pred_ned)
t_ax10 = np.arange(0,len_ax10)
len_ax10_p = len(fish_pred_ned_p)
t_ax10_p = np.arange(0,len_ax10_p)
len_ax10_t = len(fish_pred_ned_t)
t_ax10_t = np.arange(0,len_ax10_t)
len_ax10_AIC = len(fish_pred_ned_AIC)
t_ax10_AIC = np.arange(0,len_ax10_AIC)

ned_ax10 = fish_pred_ned[:,0]
ned_p_ax10 = fish_pred_ned_p[:,0]
ned_t_ax10 = fish_pred_ned_t[:,0]
ned_AIC_ax10 = fish_pred_ned_AIC[:,0]

ax10.plot(t_ax10,ned_ax10,c='cornflowerblue',linewidth=3,alpha=0.9,zorder=2)
ax10.plot(t_ax10_p,ned_p_ax10,c='blueviolet',linewidth=3,alpha=0.9,zorder=1)
ax10.plot(t_ax10_t,ned_t_ax10,c='violet',linewidth=3,alpha=0.9,zorder=1)
ax10.plot(t_ax10_AIC,ned_AIC_ax10,c='turquoise',linewidth=3,alpha=0.9,zorder=0)

ax10.set_xticks([0,15,30])
ax10.set_xlim(-3,33)
ax10.set_ylim(0.35,1)
ax10.set_yticks([0.4,0.95])
ax10.set_yticklabels(['0.4','0.95'])
ax10.tick_params(direction='in')

ax10.set_xlabel('Timepoints',font_x)
ax10.set_ylabel('Prediction inaccuracy (NED)',font_y)
ax10.yaxis.set_label_coords(-0.1, 0.5)
ax10.set_title('Fish community',fontdict=font_title)
ax10.text(-0.18, 1,'i',ha='left', transform=ax10.transAxes,fontdict={'family':'Arial','size':24,'weight':'bold'})

plt.subplots_adjust(top=0.96, bottom=0.06, left=0.04, right=0.98, hspace=0.25, wspace=0.3)
plt.savefig('../figures/FIG5.pdf',format='pdf')
plt.savefig('/Users/zhugchzo/Desktop/3paper_fig/FIG5.png',format='png',dpi=600)
# plt.show()




