import pandas
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

font_x = {'family':'Arial','weight':'normal','size': 20}
font_y = {'family':'Arial','weight':'normal','size': 20}
font_title = {'family':'DejaVu Sans','weight':'normal','size': 20, 'style': 'italic'}

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.labelweight'] = 'bold'

df_mitochondria_tseries = pandas.read_csv('../mitochondria/mitochondria_data.csv')
df_mitochondria_gen = pandas.read_csv('../results/mitochondria/mitochondria_gen.csv')

df_UAV_tseries = pandas.read_csv('../UAV/UAV_data.csv')
df_UAV_gen = pandas.read_csv('../results/UAV/UAV_gen.csv')

df_chick_220_tseries = pandas.read_csv('../chick/chick_data_220.csv')
df_chick_220_pred_2 = pandas.read_csv('../results/chick/chick_220_pred_2.csv')
df_chick_220_gen = pandas.read_csv('../results/chick/chick_220_gen.csv')

df_chick_230_tseries = pandas.read_csv('../chick/chick_data_230.csv')
df_chick_230_pred_2 = pandas.read_csv('../results/chick/chick_230_pred_2.csv')
df_chick_230_gen = pandas.read_csv('../results/chick/chick_230_gen.csv')

df_chick_335_tseries = pandas.read_csv('../chick/chick_data_335.csv')
df_chick_335_pred_2 = pandas.read_csv('../results/chick/chick_335_pred_2.csv')
df_chick_335_gen = pandas.read_csv('../results/chick/chick_335_gen.csv')

fig, axs = plt.subplots(2, 3, figsize=(15,10))

ax1, ax2, ax3 = axs[0, 0], axs[0, 1], axs[0, 2]
ax4, ax5, ax6 = axs[1, 0], axs[1, 1], axs[1, 2]

# ax1

mitochondria_t1 = df_mitochondria_tseries['Time']

rfr = df_mitochondria_tseries['Relative fluorescence ratio']
gen_rfr = df_mitochondria_gen['gen']

initial_t = mitochondria_t1.iloc[0]
end_t = mitochondria_t1.iloc[-1]
initial_theta = df_mitochondria_gen['theta'].iloc[0] - (df_mitochondria_gen['theta'].iloc[1] - df_mitochondria_gen['theta'].iloc[0])
end_theta =  df_mitochondria_gen['theta'].iloc[-1]

# fold bifurcation

t_fold = initial_t + (1.24329-initial_theta)/(end_theta-initial_theta)*(end_t-initial_t)
df_mitochondria_gen['distance'] = (df_mitochondria_gen['Time'] - t_fold).abs()
closest_row = df_mitochondria_gen.loc[df_mitochondria_gen['distance'].idxmin()]
x_fold = closest_row['gen']

ax1.plot(mitochondria_t1,rfr,c='black',linewidth=1,zorder=2)
ax1.scatter(mitochondria_t1,rfr,s=10,c='black',marker='o',zorder=2)
ax1.scatter(mitochondria_t1[1:244][::2],gen_rfr[:243][::2],s=50,marker='o',facecolors='none',edgecolors='darkorange',zorder=3)
ax1.scatter(mitochondria_t1[244:],gen_rfr[243:],s=50,marker='o',facecolors='none',edgecolors='darkorange',zorder=3)

ax1.scatter(t_fold,x_fold,s=180, marker='o',facecolors='white', edgecolors='black',zorder=5)

ax1.set_xlabel('Time (min)',font_x,labelpad=-5)
ax1.set_ylabel('Relative fluorescence ratio (%)',font_y)
ax1.set_xlim(1.8,5.1)
ax1.set_xticks([2,5])
ax1.set_ylim(0.71,0.9)
ax1.set_yticks([0.72,0.88])
ax1.set_yticklabels(['72','88'])
ax1.tick_params(direction='in')

ax1.yaxis.set_label_coords(-0.1, 0.48)

ax1.set_title('Cellular energy depletion',y=1.02,fontdict=font_title)
ax1.text(-0.125, 1,'a',ha='left', transform=ax1.transAxes,fontdict={'family':'DejaVu Sans','size':30,'weight':'bold'})

ax1.tick_params(axis='x', labelsize=18)
ax1.tick_params(axis='y', labelsize=18)

# ax2

UAV_x = df_UAV_tseries['x']
UAV_y = df_UAV_tseries['y']
gen_UAV_x = df_UAV_gen['gen_x']
gen_UAV_y = df_UAV_gen['gen_y']

ax2.plot(UAV_x,UAV_y,c='black',linewidth=1,zorder=2)
ax2.scatter(UAV_x,UAV_y,s=10,c='black',marker='o',zorder=2)
# ax2.scatter(UAV_x.iloc[0], UAV_y.iloc[0], color='royalblue',s=180,zorder=4,marker=(5,1))  # Mark the start point
ax2.scatter(gen_UAV_x[::4],gen_UAV_y[::4],s=50,marker='o',facecolors='none',edgecolors='darkorange',zorder=3)

# ax2.annotate(
#     text='',
#     xy=(2.3, 0.82),
#     xytext=(2, 0.7),
#     arrowprops=dict(arrowstyle='->', color='black', lw=3)
# )

# ax2.annotate(
#     text='',
#     xy=(6.3, 1.2),
#     xytext=(6, 1.3),
#     arrowprops=dict(arrowstyle='->', color='black', lw=3)
# )

# ax2.annotate(
#     text='',
#     xy=(8.7, 1.2),
#     xytext=(8.1, 1.1),
#     arrowprops=dict(arrowstyle='->', color='black', lw=3)
# )

ax2.annotate(
    text='',
    xy=(2.8, -0.18),
    xytext=(1.5, -0.3),
    arrowprops=dict(arrowstyle='->', color='black', lw=3)
)

ax2.annotate(
    text='',
    xy=(6.3, 1.6),
    xytext=(5, 1.9),
    arrowprops=dict(arrowstyle='->', color='black', lw=3)
)

ax2.annotate(
    text='',
    xy=(9.3, 0.5),
    xytext=(8, 0.3),
    arrowprops=dict(arrowstyle='->', color='black', lw=3)
)

ax2.text(
    x=0, y=-0.1,             
    s=r'$\text{P}_{\text{s}}$',       
    fontsize=18,
    color='black',
    ha='center', va='top'      
)

ax2.text(
    x=11, y=1,             
    s=r'$\text{P}_{\text{g}}$',       
    fontsize=18,
    color='black',
    ha='center', va='top'      
)

ax2.set_xlabel('X Position (m)',font_x,labelpad=-5)
ax2.set_ylabel('Y Position (m)',font_y,labelpad=-15)
ax2.set_xlim(-1,11.5)
ax2.set_xticks([0,10])
# ax2.set_ylim(0.1,1.5)
# ax2.set_yticks([0.2,1.4])
ax2.set_ylim(-5.25,7.25)
ax2.set_yticks([-4,6])

ax2.set_title('UAV autonomous flight',y=1.02,fontdict=font_title)
ax2.text(-0.125, 1,'b',ha='left', transform=ax2.transAxes,fontdict={'family':'DejaVu Sans','size':30,'weight':'bold'})

ax2.tick_params(axis='x', labelsize=18)
ax2.tick_params(axis='y', labelsize=18)
ax2.tick_params(direction='in')

# ax3
ax3.axis('off')

legend_state = mlines.Line2D([], [], color='black', marker='o', markersize=3, linestyle='-', markeredgewidth=1.5)
legend_state_null = mlines.Line2D([0], [0], color='none')
legend_pstate = mlines.Line2D([], [], markerfacecolor='none',color='darkorange', marker='o', markersize=5, linestyle='None', markeredgewidth=1.5)
legend_fold = mlines.Line2D([], [], markerfacecolor='white',color='black', marker='o', markersize=5, linestyle='None', markeredgewidth=1.5)
legend_pd = mlines.Line2D([], [], markerfacecolor='white',color='black', marker='h', markersize=5, linestyle='None', markeredgewidth=1.5)
# legend_start = mlines.Line2D([], [], color='royalblue', marker=(5,1), markersize=6, linestyle='None', markeredgewidth=1.5)

legend = ax3.legend(handles=[legend_state,legend_state_null,legend_state_null,legend_state_null,legend_pstate,legend_fold,legend_pd],
           labels=['System state','(a) ATP concentration','(b) Odometry path','(c-e) Inter-beat intervals',
                   'Generation','Predicted fold bifurcation','Predicted period-doubling bifurcation'],
                   loc='center',frameon=False, bbox_to_anchor=(0.45, 0.5), markerscale=2.5,prop={'size':18})

legend.get_texts()[1].set_fontsize(15)
legend.get_texts()[2].set_fontsize(15)
legend.get_texts()[3].set_fontsize(15)

# ax4

chick_220_t1 = df_chick_220_tseries['Beat number']
chick_220_t2 = df_chick_220_pred_2['Time']

ibi = df_chick_220_tseries['IBI (s)']
gen_ibi = df_chick_220_gen['gen']
pred_ibi_2 = df_chick_220_pred_2['pred']

initial_t = chick_220_t1.iloc[0]
end_t = chick_220_t1.iloc[-1]
initial_theta = df_chick_220_gen['theta'].iloc[0] - (df_chick_220_gen['theta'].iloc[1] - df_chick_220_gen['theta'].iloc[0])
end_theta =  initial_theta + (len(chick_220_t1) - 1) * (df_chick_220_gen['theta'].iloc[1] - df_chick_220_gen['theta'].iloc[0])

# period-2 bifurcation

t_pd = initial_t + (0.973317-initial_theta)/(end_theta-initial_theta)*(end_t-initial_t)
df_chick_220_gen['distance'] = (df_chick_220_gen['Time'] - t_pd).abs()
closest_row = df_chick_220_gen.loc[df_chick_220_gen['distance'].idxmin()]
x_pd = closest_row['gen']

ax4.plot(chick_220_t1,ibi,c='black',linewidth=1,zorder=2)
ax4.scatter(chick_220_t1,ibi,s=10,c='black',marker='o',zorder=2)
ax4.scatter(chick_220_t1[1:1+len(gen_ibi)][::2],gen_ibi[::2],s=50,marker='o',facecolors='none',edgecolors='darkorange',zorder=3)
ax4.scatter(chick_220_t2,pred_ibi_2,s=50,marker='o',facecolors='none',edgecolors='darkorange',zorder=3)

ax4.scatter(t_pd,x_pd,s=220, marker='h',facecolors='white', edgecolors='black',zorder=5)

ax4.set_xlabel('Beat number',font_x,labelpad=-5)
ax4.set_ylabel('IBI (s)',font_y,labelpad=-15)
ax4.set_xlim(-10,230)
ax4.set_xticks([0,220])
ax4.set_ylim(0.5,1.5)
ax4.set_yticks([0.6,1.4])
ax4.set_yticklabels(['0.6','1.4'])
ax4.tick_params(direction='in')

# ax4.yaxis.set_label_coords(-0.05, 0.48)

ax4.set_title('Beating chick-heart (I)',y=1.02,fontdict=font_title)
ax4.text(-0.125, 1,'c',ha='left', transform=ax4.transAxes,fontdict={'family':'DejaVu Sans','size':30,'weight':'bold'})

ax4.tick_params(axis='x', labelsize=18)
ax4.tick_params(axis='y', labelsize=18)

# ax5

chick_230_t1 = df_chick_230_tseries['Beat number']
chick_230_t2 = df_chick_230_pred_2['Time']

ibi = df_chick_230_tseries['IBI (s)']
gen_ibi = df_chick_230_gen['gen']
pred_ibi_2 = df_chick_230_pred_2['pred']

initial_t = chick_230_t1.iloc[0]
end_t = chick_230_t1.iloc[-1]
initial_theta = df_chick_230_gen['theta'].iloc[0] - (df_chick_230_gen['theta'].iloc[1] - df_chick_230_gen['theta'].iloc[0])
end_theta =  initial_theta + (len(chick_230_t1) - 1) * (df_chick_230_gen['theta'].iloc[1] - df_chick_230_gen['theta'].iloc[0])

# period-2 bifurcation

t_pd = initial_t + (-0.0218195-initial_theta)/(end_theta-initial_theta)*(end_t-initial_t)
df_chick_230_gen['distance'] = (df_chick_230_gen['Time'] - t_pd).abs()
closest_row = df_chick_230_gen.loc[df_chick_230_gen['distance'].idxmin()]
x_pd = closest_row['gen']

ax5.plot(chick_230_t1,ibi,c='black',linewidth=1,zorder=2)
ax5.scatter(chick_230_t1,ibi,s=10,c='black',marker='o',zorder=2)
ax5.scatter(chick_230_t1[1:1+len(gen_ibi)][::2],gen_ibi[::2],s=50,marker='o',facecolors='none',edgecolors='darkorange',zorder=3)
ax5.scatter(chick_230_t2,pred_ibi_2,s=50,marker='o',facecolors='none',edgecolors='darkorange',zorder=3)

ax5.scatter(t_pd,x_pd,s=220, marker='h',facecolors='white', edgecolors='black',zorder=5)

ax5.set_xlabel('Beat number',font_x,labelpad=-5)
ax5.set_ylabel('IBI (s)',font_y,labelpad=-15)
ax5.set_xlim(-11,241)
ax5.set_xticks([0,230])
ax5.set_ylim(0.3,1.4)
ax5.set_yticks([0.4,1.3])
ax5.set_yticklabels(['0.4','1.3'])
ax5.tick_params(direction='in')

# ax5.yaxis.set_label_coords(-0.05, 0.48)

ax5.set_title('Beating chick-heart (II)',y=1.02,fontdict=font_title)
ax5.text(-0.125, 1,'d',ha='left', transform=ax5.transAxes,fontdict={'family':'DejaVu Sans','size':30,'weight':'bold'})

ax5.tick_params(axis='x', labelsize=18)
ax5.tick_params(axis='y', labelsize=18)

# ax6

chick_335_t1 = df_chick_335_tseries['Beat number']
chick_335_t2 = df_chick_335_pred_2['Time']

ibi = df_chick_335_tseries['IBI (s)']
gen_ibi = df_chick_335_gen['gen']
pred_ibi_2 = df_chick_335_pred_2['pred']

initial_t = chick_335_t1.iloc[0]
end_t = chick_335_t1.iloc[-1]
initial_theta = df_chick_335_gen['theta'].iloc[0] - (df_chick_335_gen['theta'].iloc[1] - df_chick_335_gen['theta'].iloc[0])
end_theta =  initial_theta + (len(chick_335_t1) - 1) * (df_chick_335_gen['theta'].iloc[1] - df_chick_335_gen['theta'].iloc[0])

# period-2 bifurcation

t_pd = initial_t + (-0.704613-initial_theta)/(end_theta-initial_theta)*(end_t-initial_t)
df_chick_335_gen['distance'] = (df_chick_335_gen['Time'] - t_pd).abs()
closest_row = df_chick_335_gen.loc[df_chick_335_gen['distance'].idxmin()]
x_pd = closest_row['gen']

ax6.plot(chick_335_t1,ibi,c='black',linewidth=1,zorder=2)
ax6.scatter(chick_335_t1,ibi,s=10,c='black',marker='o',zorder=2)
ax6.scatter(chick_335_t1[1:1+len(gen_ibi)][::2],gen_ibi[::2],s=50,marker='o',facecolors='none',edgecolors='darkorange',zorder=3)
ax6.scatter(chick_335_t2,pred_ibi_2,s=50,marker='o',facecolors='none',edgecolors='darkorange',zorder=3)

ax6.scatter(t_pd,x_pd,s=220, marker='h',facecolors='white', edgecolors='black',zorder=5)

ax6.set_xlabel('Beat number',font_x,labelpad=-5)
ax6.set_ylabel('IBI (s)',font_y,labelpad=-15)
ax6.set_xlim(-15,350)
ax6.set_xticks([0,335])
ax6.set_ylim(0.5,1.6)
ax6.set_yticks([0.6,1.5])
ax6.set_yticklabels(['0.6','1.5'])
ax6.tick_params(direction='in')

# ax6.yaxis.set_label_coords(-0.05, 0.48)

ax6.set_title('Beating chick-heart (III)',y=1.02,fontdict=font_title)
ax6.text(-0.125, 1,'e',ha='left', transform=ax6.transAxes,fontdict={'family':'DejaVu Sans','size':30,'weight':'bold'})

ax6.tick_params(axis='x', labelsize=18)
ax6.tick_params(axis='y', labelsize=18)

plt.subplots_adjust(top=0.95, bottom=0.06, left=0.05, right=0.99, hspace=0.3, wspace=0.2)
plt.savefig('../figures/SFIG3.pdf',format='pdf')
plt.savefig('/Users/zhugchzo/Desktop/3paper_fig/SFIG3.png',format='png',dpi=600)







