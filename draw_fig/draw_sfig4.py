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
df_mitochondria_gen_ned = pd.read_csv('../results/NED/mitochondria/mitochondria_gen.csv')
df_mitochondria_gen_ned_t = pd.read_csv('../results/NED/mitochondria/mitochondria_gen_t.csv')
df_mitochondria_gen_ned_AIC = pd.read_csv('../results/NED/mitochondria/mitochondria_gen_AIC.csv')
# df for ax2
df_UAV_gen_ned = pd.read_csv('../results/NED/UAV/UAV_gen.csv')
df_UAV_gen_ned_t = pd.read_csv('../results/NED/UAV/UAV_gen_t.csv')
df_UAV_gen_ned_AIC = pd.read_csv('../results/NED/UAV/UAV_gen_AIC.csv')
# df for ax3
df_chick_220_gen_ned = pd.read_csv('../results/NED/chick_220/chick_220_gen.csv')
df_chick_220_gen_ned_t = pd.read_csv('../results/NED/chick_220/chick_220_gen_t.csv')
df_chick_220_gen_ned_AIC = pd.read_csv('../results/NED/chick_220/chick_220_gen_AIC.csv')
# df for ax4
df_chick_230_gen_ned = pd.read_csv('../results/NED/chick_230/chick_230_gen.csv')
df_chick_230_gen_ned_t = pd.read_csv('../results/NED/chick_230/chick_230_gen_t.csv')
df_chick_230_gen_ned_AIC = pd.read_csv('../results/NED/chick_230/chick_230_gen_AIC.csv')
# df for ax5
df_chick_335_gen_ned = pd.read_csv('../results/NED/chick_335/chick_335_gen.csv')
df_chick_335_gen_ned_t = pd.read_csv('../results/NED/chick_335/chick_335_gen_t.csv')
df_chick_335_gen_ned_AIC = pd.read_csv('../results/NED/chick_335/chick_335_gen_AIC.csv')

fig, axs = plt.subplots(1, 5, figsize=(15,5.5))

ax1, ax2, ax3, ax4, ax5 = axs[0], axs[1], axs[2], axs[3], axs[4]

# ax1
mitochondria_gen_ned = df_mitochondria_gen_ned[['normal_Eucli_d']].values
mitochondria_gen_ned_t = df_mitochondria_gen_ned_t[['normal_Eucli_d']].values
mitochondria_gen_ned_AIC = df_mitochondria_gen_ned_AIC[['normal_Eucli_d']].values

len_ax1 = len(mitochondria_gen_ned)
t_ax1 = np.arange(0,len_ax1)
len_ax1_t = len(mitochondria_gen_ned_t)
t_ax1_t = np.arange(0,len_ax1_t)
len_ax1_AIC = len(mitochondria_gen_ned_AIC)
t_ax1_AIC = np.arange(0,len_ax1_AIC)

ned_ax1 = mitochondria_gen_ned[:,0]
ned_t_ax1 = mitochondria_gen_ned_t[:,0]
ned_AIC_ax1 = mitochondria_gen_ned_AIC[:,0]

ax1.plot(t_ax1,ned_ax1,c='cornflowerblue',linewidth=3,alpha=0.9,zorder=3)
ax1.scatter(t_ax1[-1],ned_ax1[-1],c='crimson',marker='*',zorder=3,s=150,alpha=0.9)
ax1.plot(t_ax1_t,ned_t_ax1,c='violet',linewidth=3,alpha=0.9,zorder=2)
ax1.plot(t_ax1_AIC,ned_AIC_ax1,c='turquoise',linewidth=3,alpha=0.9,zorder=1)
ax1.scatter(t_ax1_AIC[-1],ned_AIC_ax1[-1],c='crimson',marker='*',zorder=3,s=150,alpha=0.9)
ax1.axvline(x=t_ax1[-1], color='crimson', linestyle='--', linewidth=1.5, zorder=0,alpha=0.9)

ax1.annotate('collapse',xy=(240, 0),xytext=(120, 0.001),fontsize=16,color='crimson',
             arrowprops=dict(arrowstyle='->',color='crimson',lw=1))

ax1.set_xticks([0,130,260])

ax1.set_ylim(-0.002,0.022)
ax1.set_yticks([0,0.02])
ax1.set_yticklabels(['0','0.02'])
ax1.tick_params(direction='in')

ax1.set_xlabel('Timepoints',font_x)
ax1.set_ylabel('Inference inaccuracy (NED)',font_y)
ax1.yaxis.set_label_coords(-0.1, 0.45)

ax1.set_title('Cellular energy depletion',fontdict=font_title)
ax1.text(-0.18, 1,'A',ha='left', transform=ax1.transAxes,fontdict={'family':'Arial','size':22,'weight':'bold'})

# ax2
UAV_gen_ned = df_UAV_gen_ned[['normal_Eucli_d']].values
UAV_gen_ned_t = df_UAV_gen_ned_t[['normal_Eucli_d']].values
UAV_gen_ned_AIC = df_UAV_gen_ned_AIC[['normal_Eucli_d']].values

len_ax2 = len(UAV_gen_ned)
t_ax2 = np.arange(0,len_ax2)
len_ax2_t = len(UAV_gen_ned_t)
t_ax2_t = np.arange(0,len_ax2_t)
len_ax2_AIC = len(UAV_gen_ned_AIC)
t_ax2_AIC = np.arange(0,len_ax2_AIC)

ned_ax2 = UAV_gen_ned[:,0]
ned_t_ax2 = UAV_gen_ned_t[:,0]
ned_AIC_ax2 = UAV_gen_ned_AIC[:,0]

ax2.plot(t_ax2,ned_ax2,c='cornflowerblue',linewidth=3,alpha=0.9,zorder=2)
ax2.plot(t_ax2_t,ned_t_ax2,c='violet',linewidth=3,alpha=0.9,zorder=1)
ax2.scatter(t_ax2_t[-1],ned_t_ax2[-1],c='crimson',marker=r'$\times$',zorder=1,s=120,alpha=0.9)
ax2.plot(t_ax2_AIC,ned_AIC_ax2,c='turquoise',linewidth=3,alpha=0.9,zorder=0)

ax2.set_xticks([0,135,270])

ax2.set_ylim(-0.06,0.66)
ax2.set_yticks([0,0.6])
ax2.set_yticklabels(['0','0.6'])
ax2.tick_params(direction='in')

ax2.set_xlabel('Timepoints',font_x)
ax2.set_ylabel('Inference inaccuracy (NED)',font_y)
ax2.yaxis.set_label_coords(-0.1, 0.45)
ax2.set_title('UAV obstacle avoidance',fontdict=font_title)
ax2.text(-0.18, 1,'B',ha='left', transform=ax2.transAxes,fontdict={'family':'Arial','size':22,'weight':'bold'})

# ax3
chick_220_gen_ned = df_chick_220_gen_ned[['normal_Eucli_d']].values
chick_220_gen_ned_t = df_chick_220_gen_ned_t[['normal_Eucli_d']].values
chick_220_gen_ned_AIC = df_chick_220_gen_ned_AIC[['normal_Eucli_d']].values

len_ax3 = len(chick_220_gen_ned)
t_ax3 = np.arange(0,len_ax3)
len_ax3_t = len(chick_220_gen_ned_t)
t_ax3_t = np.arange(0,len_ax3_t)
len_ax3_AIC = len(chick_220_gen_ned_AIC)
t_ax3_AIC = np.arange(0,len_ax3_AIC)

ned_ax3 = chick_220_gen_ned[:,0]
ned_t_ax3 = chick_220_gen_ned_t[:,0]
ned_AIC_ax3 = chick_220_gen_ned_AIC[:,0]

ax3.plot(t_ax3,ned_ax3,c='cornflowerblue',linewidth=3,alpha=0.9,zorder=3)
ax3.scatter(t_ax3[-1],ned_ax3[-1],c='chocolate',marker='*',zorder=3,s=150)
ax3.plot(t_ax3_t,ned_t_ax3,c='violet',linewidth=3,alpha=0.9,zorder=2)
ax3.scatter(t_ax3_t[-1],ned_t_ax3[-1],c='crimson',marker=r'$\times$',zorder=3,s=150,alpha=0.9)
ax3.plot(t_ax3_AIC,ned_AIC_ax3,c='turquoise',linewidth=3,alpha=0.9,zorder=1)
ax3.axvline(x=190, color='chocolate', linestyle='--', linewidth=1.5, zorder=0,alpha=0.9)

ax3.annotate('period-2',xy=(189, 0.2),xytext=(100, 0.22),fontsize=16,color='chocolate',
             arrowprops=dict(arrowstyle='->',color='chocolate',lw=1))

ax3.set_xlim(-10,230)
ax3.set_xticks([0,110,220])

ax3.set_ylim(-0.035,0.385)
ax3.set_yticks([0,0.35])
ax3.set_yticklabels(['0','0.35'])
ax3.tick_params(direction='in')

ax3.set_xlabel('Timepoints',font_x)
ax3.set_ylabel('Inference inaccuracy (NED)',font_y)
ax3.yaxis.set_label_coords(-0.1, 0.45)

ax3.set_title('Beating chick-heart (I)',fontdict=font_title)
ax3.text(-0.18, 1,'C',ha='left', transform=ax3.transAxes,fontdict={'family':'Arial','size':22,'weight':'bold'})

# ax4
chick_230_gen_ned = df_chick_230_gen_ned[['normal_Eucli_d']].values
chick_230_gen_ned_t = df_chick_230_gen_ned_t[['normal_Eucli_d']].values
chick_230_gen_ned_AIC = df_chick_230_gen_ned_AIC[['normal_Eucli_d']].values

len_ax4 = len(chick_230_gen_ned)
t_ax4 = np.arange(0,len_ax4)
len_ax4_t = len(chick_230_gen_ned_t)
t_ax4_t = np.arange(0,len_ax4_t)
len_ax4_AIC = len(chick_230_gen_ned_AIC)
t_ax4_AIC = np.arange(0,len_ax4_AIC)

ned_ax4 = chick_230_gen_ned[:,0]
ned_t_ax4 = chick_230_gen_ned_t[:,0]
ned_AIC_ax4 = chick_230_gen_ned_AIC[:,0]

ax4.plot(t_ax4,ned_ax4,c='cornflowerblue',linewidth=3,alpha=0.9,zorder=3)
ax4.scatter(t_ax4[-1],ned_ax4[-1],c='chocolate',marker='*',zorder=3,s=150)
ax4.plot(t_ax4_t,ned_t_ax4,c='violet',linewidth=3,alpha=0.9,zorder=2)
ax4.scatter(t_ax4_t[-1],ned_t_ax4[-1],c='crimson',marker=r'$\times$',zorder=3,s=150,alpha=0.9)
ax4.plot(t_ax4_AIC,ned_AIC_ax4,c='turquoise',linewidth=3,alpha=0.9,zorder=1)
ax4.axvline(x=194, color='chocolate', linestyle='--', linewidth=1.5, zorder=0,alpha=0.9)

ax4.annotate('period-2',xy=(193, 0.053),xytext=(100, 0.06),fontsize=16,color='chocolate',
             arrowprops=dict(arrowstyle='->',color='chocolate',lw=1))

ax4.set_xlim(-11,241)
ax4.set_xticks([0,115,230])

ax4.set_ylim(-0.011,0.11)
ax4.set_yticks([0,0.1])
ax4.set_yticklabels(['0','0.1'])
ax4.tick_params(direction='in')

ax4.set_xlabel('Timepoints',font_x)
ax4.set_ylabel('Inference inaccuracy (NED)',font_y)
ax4.yaxis.set_label_coords(-0.1, 0.45)

ax4.set_title('Beating chick-heart (II)',fontdict=font_title)
ax4.text(-0.18, 1,'D',ha='left', transform=ax4.transAxes,fontdict={'family':'Arial','size':22,'weight':'bold'})

# ax5
chick_335_gen_ned = df_chick_335_gen_ned[['normal_Eucli_d']].values
chick_335_gen_ned_t = df_chick_335_gen_ned_t[['normal_Eucli_d']].values
chick_335_gen_ned_AIC = df_chick_335_gen_ned_AIC[['normal_Eucli_d']].values

len_ax5 = len(chick_335_gen_ned)
t_ax5 = np.arange(0,len_ax5)
len_ax5_t = len(chick_335_gen_ned_t)
t_ax5_t = np.arange(0,len_ax5_t)
len_ax5_AIC = len(chick_335_gen_ned_AIC)
t_ax5_AIC = np.arange(0,len_ax5_AIC)

ned_ax5 = chick_335_gen_ned[:,0]
ned_t_ax5 = chick_335_gen_ned_t[:,0]
ned_AIC_ax5 = chick_335_gen_ned_AIC[:,0]

ax5.plot(t_ax5,ned_ax5,c='cornflowerblue',linewidth=3,alpha=0.9,zorder=3)
ax5.scatter(t_ax5[-1],ned_ax5[-1],c='chocolate',marker='*',zorder=3,s=150)
ax5.plot(t_ax5_t,ned_t_ax5,c='violet',linewidth=3,alpha=0.9,zorder=2)
ax5.scatter(t_ax5_t[-1],ned_t_ax5[-1],c='crimson',marker=r'$\times$',zorder=3,s=150,alpha=0.9)
ax5.plot(t_ax5_AIC,ned_AIC_ax5,c='turquoise',linewidth=3,alpha=0.9,zorder=1)
ax5.axvline(x=296, color='chocolate', linestyle='--', linewidth=1.5, zorder=0,alpha=0.9)

ax5.annotate('period-2',xy=(290, 0.35),xytext=(140, 0.38),fontsize=16,color='chocolate',
             arrowprops=dict(arrowstyle='->',color='chocolate',lw=1))

ax5.set_xlim(-16,350)
ax5.set_xticks([0,170,335])

ax5.set_ylim(-0.055,0.605)
ax5.set_yticks([0,0.55])
ax5.set_yticklabels(['0','0.55'])
ax5.tick_params(direction='in')

ax5.set_xlabel('Timepoints',font_x)
ax5.set_ylabel('Inference inaccuracy (NED)',font_y)
ax5.yaxis.set_label_coords(-0.1, 0.45)

ax5.set_title('Beating chick-heart (III)',fontdict=font_title)
ax5.text(-0.18, 1,'E',ha='left', transform=ax5.transAxes,fontdict={'family':'Arial','size':22,'weight':'bold'})

# legend
legend_v = mlines.Line2D([], [], color='cornflowerblue', marker='none', linestyle='-', linewidth=3,alpha=0.9)
legend_t = mlines.Line2D([], [], color='violet', marker='none', linestyle='-', linewidth=3,alpha=0.9)
legend_AIC = mlines.Line2D([], [], color='turquoise', marker='none', linestyle='-', linewidth=3,alpha=0.9)
legend_tclsp = mlines.Line2D([], [], color='crimson', marker='*', linestyle='none', markersize=10,alpha=0.9)
legend_pd2 = mlines.Line2D([], [], color='chocolate', marker='*', linestyle='none', markersize=10,alpha=1)
legend_fclsp = mlines.Line2D([], [], color='crimson', marker=r'$\times$', linestyle='none', markersize=8,alpha=0.9)

fig.legend(
    handles=[legend_v, legend_tclsp,  legend_t, legend_pd2, legend_AIC, legend_fclsp],
    labels=['Optimal driving variable', 'Equations collapse (match)', 'Time variable',
             'Equations enter period-2 (match)', r'Use AIC instead of $\epsilon$AIC', 'Equations collapse (mismatch)'],
    loc='upper center',
    bbox_to_anchor=(0.5, 1.02),
    ncol=3,
    frameon=False,
    markerscale=1.5,
    prop=font_manager.FontProperties(family='Arial Unicode MS', size=16)
)

plt.subplots_adjust(top=0.75, bottom=0.1, left=0.04, right=0.98, wspace=0.3)
plt.savefig('../figures/SFIG4.pdf',format='pdf')
plt.savefig('/Users/zhugchzo/Desktop/3paper_fig/SFIG4.png',format='png',dpi=600)
# plt.show()




