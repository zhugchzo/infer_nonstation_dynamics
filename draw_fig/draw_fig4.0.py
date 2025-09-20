import pandas
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib import font_manager

font_x = {'family':'Arial','weight':'normal','size': 20}
font_y = {'family':'Arial','weight':'normal','size': 20}
font_title = {'family':'DejaVu Sans','weight':'normal','size': 20, 'style': 'italic'}

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.labelweight'] = 'bold'

df_mitochondria_tseries = pandas.read_csv('../mitochondria/mitochondria_data.csv')
df_mitochondria_pred = pandas.read_csv('../results/mitochondria/mitochondria_pred.csv')
df_mitochondria_gen = pandas.read_csv('../results/mitochondria/mitochondria_gen.csv')

df_UAV_tseries = pandas.read_csv('../UAV/UAV_data.csv')
df_UAV_pred = pandas.read_csv('../results/UAV/UAV_pred.csv')

df_chick_220_tseries = pandas.read_csv('../chick/chick_data_220.csv')
df_chick_220_pred_1 = pandas.read_csv('../results/chick/chick_220_pred_1.csv')
df_chick_220_pred_2 = pandas.read_csv('../results/chick/chick_220_pred_2.csv')
df_chick_220_gen = pandas.read_csv('../results/chick/chick_220_gen.csv')

df_chick_230_tseries = pandas.read_csv('../chick/chick_data_230.csv')
df_chick_230_pred_1 = pandas.read_csv('../results/chick/chick_230_pred_1.csv')
df_chick_230_pred_2 = pandas.read_csv('../results/chick/chick_230_pred_2.csv')
df_chick_230_gen = pandas.read_csv('../results/chick/chick_230_gen.csv')

df_chick_270_tseries = pandas.read_csv('../chick/chick_data_270.csv')
df_chick_270_pred_1 = pandas.read_csv('../results/chick/chick_270_pred_1.csv')
df_chick_270_pred_2 = pandas.read_csv('../results/chick/chick_270_pred_2.csv')
df_chick_270_gen = pandas.read_csv('../results/chick/chick_270_gen.csv')

mitochondria_train_length = len(df_mitochondria_tseries) - len(df_mitochondria_pred) - 1
UAV_train_length = len(df_UAV_tseries) - len(df_UAV_pred) - 1
chick_train_length = 150

fig, axs = plt.subplots(2, 4, figsize=(20,12))

ax1, ax2, ax3, ax4 = axs[0, 0], axs[0, 1], axs[0, 2], axs[0, 3]
ax5, ax6, ax7, ax8 = axs[1, 0], axs[1, 1], axs[1, 2], axs[1, 3]

# ax1

ax1.axis('off')
ax1.set_title('Experiment illustration',x=0.4,y=0.9,fontdict=font_title)
ax1.text(-0.125, 1.05,'a',ha='left', transform=ax1.transAxes,fontdict={'family':'DejaVu Sans','size':25,'weight':'bold'})

# ax2

mitochondria_t1 = df_mitochondria_tseries['Time']
mitochondria_t2 = df_mitochondria_pred['Time']

rfr = df_mitochondria_tseries['Relative fluorescence ratio']
pred_rfr = df_mitochondria_pred['pred']

initial_t = mitochondria_t1.iloc[0]
end_t = mitochondria_t1.iloc[-1]
initial_theta = df_mitochondria_gen['theta'].iloc[0] - (df_mitochondria_gen['theta'].iloc[1] - df_mitochondria_gen['theta'].iloc[0])
end_theta =  df_mitochondria_gen['theta'].iloc[-1]

# fold bifurcation

t_fold = initial_t + (1.24329-initial_theta)/(end_theta-initial_theta)*(end_t-initial_t)
print('fold:{}'.format(t_fold))
df_mitochondria_gen['distance'] = (df_mitochondria_gen['Time'] - t_fold).abs()
closest_row = df_mitochondria_gen.loc[df_mitochondria_gen['distance'].idxmin()]
x_fold = closest_row['gen']

ax2.plot(mitochondria_t1[:mitochondria_train_length],rfr[:mitochondria_train_length],c='slategrey',zorder=2)
ax2.scatter(mitochondria_t1[:mitochondria_train_length],rfr[:mitochondria_train_length],s=10,c='slategrey',marker='o',zorder=2)
ax2.plot(mitochondria_t1[mitochondria_train_length:],rfr[mitochondria_train_length:],c='black',zorder=2)
ax2.scatter(mitochondria_t1[mitochondria_train_length:],rfr[mitochondria_train_length:],s=10,c='black',marker='o',zorder=2)
ax2.scatter(mitochondria_t2[:62][::2],pred_rfr[:62][::2],s=50,marker='o',facecolors='none',edgecolors='crimson',zorder=3)
ax2.scatter(mitochondria_t2[62:],pred_rfr[62:],s=50,marker='o',facecolors='none',edgecolors='crimson',zorder=3)

ax2.scatter(t_fold,x_fold,s=180, marker='o',facecolors='white', edgecolors='black',zorder=5)

ax2.set_xlabel('Time (min)',font_x,labelpad=0)
ax2.set_ylabel('Relative fluorescence ratio (%)',font_y)
ax2.set_xlim(1.8,5.1)
ax2.set_xticks([2,4,5])
ax2.set_ylim(0.71,0.9)
ax2.set_yticks([0.72,0.88])
ax2.set_yticklabels(['72','88'])
ax2.tick_params(direction='in')

ax2.yaxis.set_label_coords(-0.1, 0.48)

ax2.set_title('Cellular energy depletion',y=1.02,fontdict=font_title)
ax2.text(-0.125, 1.05,'b',ha='left', transform=ax2.transAxes,fontdict={'family':'DejaVu Sans','size':25,'weight':'bold'})

ax2.tick_params(axis='x', labelsize=18)
ax2.tick_params(axis='y', labelsize=18)

# ax3

ax3.axis('off')
ax3.set_title('Experiment illustration',x=0.45,y=0.9,fontdict=font_title)
ax3.text(-0.125, 1.05,'c',ha='left', transform=ax3.transAxes,fontdict={'family':'DejaVu Sans','size':25,'weight':'bold'})

# ax4

UAV_x = df_UAV_tseries['x']
UAV_y = df_UAV_tseries['y']
pred_UAV_x = df_UAV_pred['pred_x']
pred_UAV_y = df_UAV_pred['pred_y']

ax4.plot(UAV_x[:UAV_train_length][::4],UAV_y[:UAV_train_length][::4],c='slategrey',zorder=2)
ax4.scatter(UAV_x[:UAV_train_length][::4],UAV_y[:UAV_train_length][::4],s=10,c='slategrey',marker='o',zorder=2)
ax4.plot(UAV_x[UAV_train_length:][::4],UAV_y[UAV_train_length:][::4],c='black',zorder=2)
ax4.scatter(UAV_x[UAV_train_length:][::4],UAV_y[UAV_train_length:][::4],s=10,c='black',marker='o',zorder=2)
ax4.scatter(UAV_x.iloc[0], UAV_y.iloc[0], color='royalblue',s=180,zorder=4,marker=(5,1))  # Mark the start point
ax4.scatter(pred_UAV_x[::4],pred_UAV_y[::4],s=50,marker='o',facecolors='none',edgecolors='crimson',zorder=3)

ax4.annotate(
    text='',
    xy=(2.8, -0.18),
    xytext=(1.5, -0.3),
    arrowprops=dict(arrowstyle='->', color='black', lw=3)
)

ax4.annotate(
    text='',
    xy=(6.3, 1.6),
    xytext=(5, 1.9),
    arrowprops=dict(arrowstyle='->', color='black', lw=3)
)

ax4.annotate(
    text='',
    xy=(9.3, 0.5),
    xytext=(8, 0.3),
    arrowprops=dict(arrowstyle='->', color='black', lw=3)
)

ax4.set_xlabel('X Position (m)',font_x,labelpad=-10)
ax4.set_ylabel('Y Position (m)',font_y,labelpad=-15)
ax4.set_xlim(-1,11.5)
ax4.set_xticks([0,10])
# ax4.set_ylim(0.1,1.5)
# ax4.set_yticks([0.2,1.4])
ax4.set_ylim(-6.25,6.25)
ax4.set_yticks([-5,5])

ax4.set_title('UAV obstacle avoidance',y=1.02,fontdict=font_title)
ax4.text(-0.125, 1.05,'d',ha='left', transform=ax4.transAxes,fontdict={'family':'DejaVu Sans','size':25,'weight':'bold'})

ax4.tick_params(axis='x', labelsize=18)
ax4.tick_params(axis='y', labelsize=18)
ax4.tick_params(direction='in')

# ax5

ax5.axis('off')
ax5.set_title('Experiment illustration',x=0.4,y=0.9,fontdict=font_title)
ax5.text(-0.125, 1.05,'e',ha='left', transform=ax5.transAxes,fontdict={'family':'DejaVu Sans','size':25,'weight':'bold'})

# ax6

chick_220_t1 = df_chick_220_tseries['Beat number']
chick_220_t2 = df_chick_220_pred_1['Time']
chick_220_t3 = df_chick_220_pred_2['Time']

ibi = df_chick_220_tseries['IBI (s)']
train_ibi = df_chick_220_gen['gen'][:chick_train_length]
pred_ibi_1 = df_chick_220_pred_1['pred']
pred_ibi_2 = df_chick_220_pred_2['pred']

initial_t = chick_220_t1.iloc[0]
end_t = chick_220_t1.iloc[-1]
initial_theta = df_chick_220_gen['theta'].iloc[0] - (df_chick_220_gen['theta'].iloc[1] - df_chick_220_gen['theta'].iloc[0])
end_theta =  initial_theta + (len(chick_220_t1) - 1) * (df_chick_220_gen['theta'].iloc[1] - df_chick_220_gen['theta'].iloc[0])

# period-2 bifurcation

t_pd = initial_t + (0.973317-initial_theta)/(end_theta-initial_theta)*(end_t-initial_t)
print('period-2 220:{}'.format(t_pd))
df_chick_220_gen['distance'] = (df_chick_220_gen['Time'] - t_pd).abs()
closest_row = df_chick_220_gen.loc[df_chick_220_gen['distance'].idxmin()]
x_pd = closest_row['gen']

ax6.plot(chick_220_t1[:chick_train_length],ibi[:chick_train_length],c='slategrey',zorder=2)
ax6.scatter(chick_220_t1[:chick_train_length],ibi[:chick_train_length],s=10,c='slategrey',marker='o',zorder=2)
ax6.plot(chick_220_t1[chick_train_length:],ibi[chick_train_length:],c='black',zorder=2)
ax6.scatter(chick_220_t1[chick_train_length:],ibi[chick_train_length:],s=10,c='black',marker='o',zorder=2)
ax6.scatter(chick_220_t2[::2],pred_ibi_1[::2],s=50,marker='o',facecolors='none',edgecolors='crimson',zorder=3)
ax6.scatter(chick_220_t3,pred_ibi_2,s=50,marker='o',facecolors='none',edgecolors='crimson',zorder=3)

ax6.scatter(t_pd,x_pd,s=220, marker='h',facecolors='white', edgecolors='black',zorder=5)

ax6.set_xlabel('Beat number',font_x,labelpad=0)
ax6.set_ylabel('IBI (s)',font_y,labelpad=-15)
ax6.set_xlim(-10,230)
ax6.set_xticks([0,150,220])
ax6.set_ylim(0.5,1.5)
ax6.set_yticks([0.6,1.4])
ax6.set_yticklabels(['0.6','1.4'])
ax6.tick_params(direction='in')

# ax6.yaxis.set_label_coords(-0.05, 0.48)

ax6.set_title('Beating chick-heart (I)',y=1.02,fontdict=font_title)
ax6.text(-0.125, 1.05,'f',ha='left', transform=ax6.transAxes,fontdict={'family':'DejaVu Sans','size':25,'weight':'bold'})

ax6.tick_params(axis='x', labelsize=18)
ax6.tick_params(axis='y', labelsize=18)

# ax7

chick_230_t1 = df_chick_230_tseries['Beat number']
chick_230_t2 = df_chick_230_pred_1['Time']
chick_230_t3 = df_chick_230_pred_2['Time']

ibi = df_chick_230_tseries['IBI (s)']
train_ibi = df_chick_230_gen['gen'][:chick_train_length]
pred_ibi_1 = df_chick_230_pred_1['pred']
pred_ibi_2 = df_chick_230_pred_2['pred']

initial_t = chick_230_t1.iloc[0]
end_t = chick_230_t1.iloc[-1]
initial_theta = df_chick_230_gen['theta'].iloc[0] - (df_chick_230_gen['theta'].iloc[1] - df_chick_230_gen['theta'].iloc[0])
end_theta =  initial_theta + (len(chick_230_t1) - 1) * (df_chick_230_gen['theta'].iloc[1] - df_chick_230_gen['theta'].iloc[0])

# period-2 bifurcation

t_pd = initial_t + (-0.0218195-initial_theta)/(end_theta-initial_theta)*(end_t-initial_t)
print('period-2 230:{}'.format(t_pd))
df_chick_230_gen['distance'] = (df_chick_230_gen['Time'] - t_pd).abs()
closest_row = df_chick_230_gen.loc[df_chick_230_gen['distance'].idxmin()]
x_pd = closest_row['gen']

ax7.plot(chick_230_t1[:chick_train_length],ibi[:chick_train_length],c='slategrey',zorder=2)
ax7.scatter(chick_230_t1[:chick_train_length],ibi[:chick_train_length],s=10,c='slategrey',marker='o',zorder=2)
ax7.plot(chick_230_t1[chick_train_length:],ibi[chick_train_length:],c='black',zorder=2)
ax7.scatter(chick_230_t1[chick_train_length:],ibi[chick_train_length:],s=10,c='black',marker='o',zorder=2)
ax7.scatter(chick_230_t2[::2],pred_ibi_1[::2],s=50,marker='o',facecolors='none',edgecolors='crimson',zorder=3)
ax7.scatter(chick_230_t3,pred_ibi_2,s=50,marker='o',facecolors='none',edgecolors='crimson',zorder=3)

ax7.scatter(t_pd,x_pd,s=220, marker='h',facecolors='white', edgecolors='black',zorder=5)

ax7.set_xlabel('Beat number',font_x,labelpad=0)
ax7.set_ylabel('IBI (s)',font_y,labelpad=-15)
ax7.set_xlim(-11,241)
ax7.set_xticks([0,150,230])
ax7.set_ylim(0.3,1.4)
ax7.set_yticks([0.4,1.3])
ax7.set_yticklabels(['0.4','1.3'])
ax7.tick_params(direction='in')

# ax7.yaxis.set_label_coords(-0.05, 0.48)

ax7.set_title('Beating chick-heart (II)',y=1.02,fontdict=font_title)
ax7.text(-0.125, 1.05,'g',ha='left', transform=ax7.transAxes,fontdict={'family':'DejaVu Sans','size':25,'weight':'bold'})

ax7.tick_params(axis='x', labelsize=18)
ax7.tick_params(axis='y', labelsize=18)

# ax6

chick_270_t1 = df_chick_270_tseries['Beat number']
chick_270_t2 = df_chick_270_pred_1['Time']
chick_270_t3 = df_chick_270_pred_2['Time']

ibi = df_chick_270_tseries['IBI (s)']
train_ibi = df_chick_270_gen['gen'][:chick_train_length]
pred_ibi_1 = df_chick_270_pred_1['pred']
pred_ibi_2 = df_chick_270_pred_2['pred']

initial_t = chick_270_t1.iloc[0]
end_t = chick_270_t1.iloc[-1]
initial_theta = df_chick_270_gen['theta'].iloc[0] - (df_chick_270_gen['theta'].iloc[1] - df_chick_270_gen['theta'].iloc[0])
end_theta =  initial_theta + (len(chick_270_t1) - 1) * (df_chick_270_gen['theta'].iloc[1] - df_chick_270_gen['theta'].iloc[0])

# period-2 bifurcation

t_pd = initial_t + (0.119827-initial_theta)/(end_theta-initial_theta)*(end_t-initial_t)
print('period-2 270:{}'.format(t_pd))
df_chick_270_gen['distance'] = (df_chick_270_gen['Time'] - t_pd).abs()
closest_row = df_chick_270_gen.loc[df_chick_270_gen['distance'].idxmin()]
x_pd = closest_row['gen']

ax8.plot(chick_270_t1[:chick_train_length],ibi[:chick_train_length],c='slategrey',zorder=2)
ax8.scatter(chick_270_t1[:chick_train_length],ibi[:chick_train_length],s=10,c='slategrey',marker='o',zorder=2)
ax8.plot(chick_270_t1[chick_train_length:],ibi[chick_train_length:],c='black',zorder=2)
ax8.scatter(chick_270_t1[chick_train_length:],ibi[chick_train_length:],s=10,c='black',marker='o',zorder=2)
ax8.scatter(chick_270_t2[::2],pred_ibi_1[::2],s=50,marker='o',facecolors='none',edgecolors='crimson',zorder=3)
ax8.scatter(chick_270_t3,pred_ibi_2,s=50,marker='o',facecolors='none',edgecolors='crimson',zorder=3)

ax8.scatter(t_pd,x_pd,s=220, marker='h',facecolors='white', edgecolors='black',zorder=5)

ax8.set_xlabel('Beat number',font_x,labelpad=0)
ax8.set_ylabel('IBI (s)',font_y,labelpad=-15)
ax8.set_xlim(-13,283)
ax8.set_xticks([0,150,270])
ax8.set_ylim(0.3,1.8)
ax8.set_yticks([0.4,1.7])
ax8.set_yticklabels(['0.4','1.7'])
ax8.tick_params(direction='in')

# ax8.yaxis.set_label_coords(-0.05, 0.48)

ax8.set_title('Beating chick-heart (III)',y=1.02,fontdict=font_title)
ax8.text(-0.125, 1.05,'h',ha='left', transform=ax8.transAxes,fontdict={'family':'DejaVu Sans','size':25,'weight':'bold'})

ax8.tick_params(axis='x', labelsize=18)
ax8.tick_params(axis='y', labelsize=18)

legend_state = mlines.Line2D([], [], color='black', marker='o', markersize=3, linestyle='-', markeredgewidth=1.5)
legend_state_null = mlines.Line2D([0], [0], color='none')
legend_train = mlines.Line2D([], [], color='slategrey', marker='o', markersize=3, linestyle='-', markeredgewidth=1.5)
legend_pstate = mlines.Line2D([], [], markerfacecolor='none',color='crimson', marker='o', markersize=5, linestyle='None', markeredgewidth=1.5)
legend_fold = mlines.Line2D([], [], markerfacecolor='white',color='black', marker='o', markersize=5, linestyle='None', markeredgewidth=1.5)
legend_pd = mlines.Line2D([], [], markerfacecolor='white',color='black', marker='h', markersize=5, linestyle='None', markeredgewidth=1.5)
legend_start = mlines.Line2D([], [], color='royalblue', marker=(5,1), markersize=6, linestyle='None', markeredgewidth=1.5)

legend_1 = fig.legend(
    handles=[legend_state,legend_state_null,legend_state_null,legend_state_null],
    labels=['System state','(b) ATP concentration','(d) Odometry path','(f-h) Inter-beat intervals'],
    loc='upper center',
    bbox_to_anchor=(0.25, 1.01),
    ncol=1,
    frameon=False,
    markerscale=2.5,
    prop=font_manager.FontProperties(family='Arial Unicode MS', size=18)
)

legend_1.get_texts()[1].set_fontsize(15)
legend_1.get_texts()[2].set_fontsize(15)
legend_1.get_texts()[3].set_fontsize(15)

fig.legend(
    handles=[legend_train,legend_pstate,legend_fold,legend_pd],
    labels=['Training data','Prediction','Predicted fold bifurcation','Predicted period-doubling bifurcation'],
    loc='upper center',
    bbox_to_anchor=(0.5, 1.01),
    ncol=1,
    frameon=False,
    markerscale=2.5,
    prop=font_manager.FontProperties(family='Arial Unicode MS', size=18)
)

fig.legend(
    handles=[legend_start],
    labels=['Initial UAV position'],
    loc='upper center',
    bbox_to_anchor=(0.75, 1.01),
    ncol=1,
    frameon=False,
    markerscale=2.5,
    prop=font_manager.FontProperties(family='Arial Unicode MS', size=18)
)

plt.subplots_adjust(top=0.8, bottom=0.05, left=0.05, right=0.99, hspace=0.3, wspace=0.2)
plt.savefig('../figures/FIG4.0.pdf',format='pdf')







