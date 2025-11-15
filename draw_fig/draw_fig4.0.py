import pandas
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib import font_manager
import matplotlib.patches as mpatches
from matplotlib.legend_handler import HandlerBase

class HandlerQuiver(HandlerBase):
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        qv = orig_handle
        color = qv.get_facecolor()[0] if hasattr(qv, "get_facecolor") else 'chocolate'

        arrow_length = width * 0.8    # 总长度（原 0.8）
        arrow_thickness = height * 0.2  # 箭杆厚度（原 0.1）

        patch = mpatches.FancyArrow(
            width * 0.05, height * 0.5,   # 起点位置（略微下移）
            arrow_length, 0,               # dx, dy
            width=arrow_thickness,
            color=color
        )
        return [patch]


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
df_UAV_acc = pandas.read_csv('../results/UAV/UAV_acc.csv')

df_chick_220_tseries = pandas.read_csv('../chick/chick_data_220.csv')
df_chick_220_pred_1 = pandas.read_csv('../results/chick/chick_220_pred_1.csv')
df_chick_220_pred_2 = pandas.read_csv('../results/chick/chick_220_pred_2.csv')
df_chick_220_gen = pandas.read_csv('../results/chick/chick_220_gen.csv')

df_chick_230_tseries = pandas.read_csv('../chick/chick_data_230.csv')
df_chick_230_pred_1 = pandas.read_csv('../results/chick/chick_230_pred_1.csv')
df_chick_230_pred_2 = pandas.read_csv('../results/chick/chick_230_pred_2.csv')
df_chick_230_gen = pandas.read_csv('../results/chick/chick_230_gen.csv')

df_chick_335_tseries = pandas.read_csv('../chick/chick_data_335.csv')
df_chick_335_pred_1 = pandas.read_csv('../results/chick/chick_335_pred_1.csv')
df_chick_335_pred_2 = pandas.read_csv('../results/chick/chick_335_pred_2.csv')
df_chick_335_gen = pandas.read_csv('../results/chick/chick_335_gen.csv')

mitochondria_train_length = len(df_mitochondria_tseries) - len(df_mitochondria_pred) - 1
UAV_train_length = len(df_UAV_tseries) - len(df_UAV_pred) - 1
acc_length = len(df_UAV_acc)

chick_train_length_1 = 150
chick_train_length_2 = 200

fig, axs = plt.subplots(2, 4, figsize=(20,12))

ax1, ax2, ax3, ax4 = axs[0, 0], axs[0, 1], axs[0, 2], axs[0, 3]
ax5, ax6, ax7, ax8 = axs[1, 0], axs[1, 1], axs[1, 2], axs[1, 3]

# ax1

ax1.axis('off')
ax1.set_title('Experiment illustration',x=0.4,y=0.9,fontdict=font_title)
ax1.text(-0.125, 1.05,'a',ha='left', transform=ax1.transAxes,fontdict={'family':'DejaVu Sans','size':30,'weight':'bold'})

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

ax2.plot(mitochondria_t1[:mitochondria_train_length],rfr[:mitochondria_train_length],c='slategrey',linewidth=1,zorder=2)
ax2.scatter(mitochondria_t1[:mitochondria_train_length],rfr[:mitochondria_train_length],s=10,c='slategrey',marker='o',zorder=2)
ax2.plot(mitochondria_t1[mitochondria_train_length:],rfr[mitochondria_train_length:],c='black',linewidth=1,zorder=2)
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
ax2.text(-0.125, 1.05,'b',ha='left', transform=ax2.transAxes,fontdict={'family':'DejaVu Sans','size':30,'weight':'bold'})

ax2.tick_params(axis='x', labelsize=18)
ax2.tick_params(axis='y', labelsize=18)

# ax3

ax3.axis('off')
ax3.set_title('Experiment illustration',x=0.45,y=0.9,fontdict=font_title)
ax3.text(-0.125, 1.05,'c',ha='left', transform=ax3.transAxes,fontdict={'family':'DejaVu Sans','size':30,'weight':'bold'})

# ax4

UAV_x = df_UAV_tseries['x']
UAV_y = df_UAV_tseries['y']
pred_UAV_x = df_UAV_pred['pred_x']
pred_UAV_y = df_UAV_pred['pred_y']
acc_x = df_UAV_acc['accx']
acc_y = df_UAV_acc['accy']

ax4.plot(UAV_x[:UAV_train_length][::4],UAV_y[:UAV_train_length][::4],c='slategrey',linewidth=1,zorder=2)
ax4.scatter(UAV_x[:UAV_train_length][::4],UAV_y[:UAV_train_length][::4],s=10,c='slategrey',marker='o',zorder=2)
ax4.plot(UAV_x[UAV_train_length:][::4],UAV_y[UAV_train_length:][::4],c='black',linewidth=1,zorder=2)
ax4.scatter(UAV_x[UAV_train_length:][::4],UAV_y[UAV_train_length:][::4],s=10,c='black',marker='o',zorder=2)
# ax4.scatter(UAV_x.iloc[0], UAV_y.iloc[0], color='royalblue',s=180,zorder=4,marker=(5,1))  # Mark the start point
ax4.scatter(pred_UAV_x[::4],pred_UAV_y[::4],s=50,marker='o',facecolors='none',edgecolors='crimson',zorder=3)

for i in range(24,56,8):
    ax4.quiver(UAV_x.iloc[i], UAV_y.iloc[i], acc_x.iloc[i], acc_y.iloc[i], color='chocolate', angles='xy', scale_units='xy', scale=1, headwidth=5, headlength=6, headaxislength=5, width=0.005, alpha=0.6)

for i in range(72,184,8):
    ax4.quiver(UAV_x.iloc[i], UAV_y.iloc[i], acc_x.iloc[i], acc_y.iloc[i], color='chocolate', angles='xy', scale_units='xy', scale=1, headwidth=5, headlength=6, headaxislength=5, width=0.005, alpha=0.6)

for i in range(184,acc_length-16,8):
    ax4.quiver(UAV_x.iloc[i], UAV_y.iloc[i], acc_x.iloc[i], acc_y.iloc[i], color='chocolate', angles='xy', scale_units='xy', scale=0.7, headwidth=5, headlength=6, headaxislength=5, width=0.005, alpha=0.6)

ax4.quiver(UAV_x.iloc[0], UAV_y.iloc[0], acc_x.iloc[0], acc_y.iloc[0], color='chocolate', angles='xy', scale_units='xy', scale=1, headwidth=5, headlength=6, headaxislength=5, width=0.005, alpha=0.6)
legend_arrow = ax4.quiver(UAV_x.iloc[-1], UAV_y.iloc[-1], acc_x.iloc[-1], acc_y.iloc[-1], color='chocolate', angles='xy', scale_units='xy', scale=0.7, headwidth=5, headlength=6, headaxislength=5, width=0.005, alpha=0.6)

ax4.annotate(
    text='',
    xy=(0.8, -1.38),
    xytext=(-0.5, -1.5),
    arrowprops=dict(
        arrowstyle='-|>',    
        color='black',
        lw=3,
        mutation_scale=18    
    )
)

ax4.text(
    x=1.2, y=-2,             
    s='Initial velocity',       
    fontsize=14,
    color='black',
    ha='center', va='top'      
)

ax4.text(
    x=0, y=-0.1,             
    s=r'$\text{P}_{\text{s}}$',       
    fontsize=18,
    color='black',
    ha='center', va='top'      
)

ax4.text(
    x=11, y=1,             
    s=r'$\text{P}_{\text{g}}$',       
    fontsize=18,
    color='black',
    ha='center', va='top'      
)

# ax4.annotate(
#     text='',
#     xy=(6.3, 1.6),
#     xytext=(5, 1.9),
#     arrowprops=dict(arrowstyle='->', color='black', lw=3)
# )

# ax4.annotate(
#     text='',
#     xy=(9.3, 0.5),
#     xytext=(8, 0.3),
#     arrowprops=dict(arrowstyle='->', color='black', lw=3)
# )

ax4.set_xlabel('X Position (m)',font_x,labelpad=-10)
ax4.set_ylabel('Y Position (m)',font_y,labelpad=-15)
ax4.set_xlim(-0.8,11.7)
ax4.set_xticks([0,10])
# ax4.set_ylim(0.1,1.5)
# ax4.set_yticks([0.2,1.4])
ax4.set_ylim(-5.25,7.25)
ax4.set_yticks([-4,6])

ax4.set_title('UAV autonomous flight',y=1.02,fontdict=font_title)
ax4.text(-0.125, 1.05,'d',ha='left', transform=ax4.transAxes,fontdict={'family':'DejaVu Sans','size':30,'weight':'bold'})

ax4.tick_params(axis='x', labelsize=18)
ax4.tick_params(axis='y', labelsize=18)
ax4.tick_params(direction='in')

# ax5

ax5.axis('off')
ax5.set_title('Experiment illustration',x=0.4,y=0.9,fontdict=font_title)
ax5.text(-0.125, 1.05,'e',ha='left', transform=ax5.transAxes,fontdict={'family':'DejaVu Sans','size':30,'weight':'bold'})

# ax6

chick_220_t1 = df_chick_220_tseries['Beat number']
chick_220_t2 = df_chick_220_pred_1['Time']
chick_220_t3 = df_chick_220_pred_2['Time']

ibi = df_chick_220_tseries['IBI (s)']
train_ibi = df_chick_220_gen['gen'][:chick_train_length_1]
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

ax6.plot(chick_220_t1[:chick_train_length_1],ibi[:chick_train_length_1],c='slategrey',linewidth=1,zorder=2)
ax6.scatter(chick_220_t1[:chick_train_length_1],ibi[:chick_train_length_1],s=10,c='slategrey',marker='o',zorder=2)
ax6.plot(chick_220_t1[chick_train_length_1:],ibi[chick_train_length_1:],c='black',linewidth=1,zorder=2)
ax6.scatter(chick_220_t1[chick_train_length_1:],ibi[chick_train_length_1:],s=10,c='black',marker='o',zorder=2)
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
ax6.text(-0.125, 1.05,'f',ha='left', transform=ax6.transAxes,fontdict={'family':'DejaVu Sans','size':30,'weight':'bold'})

ax6.tick_params(axis='x', labelsize=18)
ax6.tick_params(axis='y', labelsize=18)

# ax7

chick_230_t1 = df_chick_230_tseries['Beat number']
chick_230_t2 = df_chick_230_pred_1['Time']
chick_230_t3 = df_chick_230_pred_2['Time']

ibi = df_chick_230_tseries['IBI (s)']
train_ibi = df_chick_230_gen['gen'][:chick_train_length_1]
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

ax7.plot(chick_230_t1[:chick_train_length_1],ibi[:chick_train_length_1],c='slategrey',linewidth=1,zorder=2)
ax7.scatter(chick_230_t1[:chick_train_length_1],ibi[:chick_train_length_1],s=10,c='slategrey',marker='o',zorder=2)
ax7.plot(chick_230_t1[chick_train_length_1:],ibi[chick_train_length_1:],c='black',linewidth=1,zorder=2)
ax7.scatter(chick_230_t1[chick_train_length_1:],ibi[chick_train_length_1:],s=10,c='black',marker='o',zorder=2)
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
ax7.text(-0.125, 1.05,'g',ha='left', transform=ax7.transAxes,fontdict={'family':'DejaVu Sans','size':30,'weight':'bold'})

ax7.tick_params(axis='x', labelsize=18)
ax7.tick_params(axis='y', labelsize=18)

# ax8

chick_335_t1 = df_chick_335_tseries['Beat number']
chick_335_t2 = df_chick_335_pred_1['Time']
chick_335_t3 = df_chick_335_pred_2['Time']

ibi = df_chick_335_tseries['IBI (s)']
train_ibi = df_chick_335_gen['gen'][:chick_train_length_2]
pred_ibi_1 = df_chick_335_pred_1['pred']
pred_ibi_2 = df_chick_335_pred_2['pred']

initial_t = chick_335_t1.iloc[0]
end_t = chick_335_t1.iloc[-1]
initial_theta = df_chick_335_gen['theta'].iloc[0] - (df_chick_335_gen['theta'].iloc[1] - df_chick_335_gen['theta'].iloc[0])
end_theta =  initial_theta + (len(chick_335_t1) - 1) * (df_chick_335_gen['theta'].iloc[1] - df_chick_335_gen['theta'].iloc[0])

# period-2 bifurcation

t_pd = initial_t + (-0.704613-initial_theta)/(end_theta-initial_theta)*(end_t-initial_t)
print('period-2 335:{}'.format(t_pd))
df_chick_335_gen['distance'] = (df_chick_335_gen['Time'] - t_pd).abs()
closest_row = df_chick_335_gen.loc[df_chick_335_gen['distance'].idxmin()]
x_pd = closest_row['gen']

ax8.plot(chick_335_t1[:chick_train_length_2],ibi[:chick_train_length_2],c='slategrey',linewidth=1,zorder=2)
ax8.scatter(chick_335_t1[:chick_train_length_2],ibi[:chick_train_length_2],s=10,c='slategrey',marker='o',zorder=2)
ax8.plot(chick_335_t1[chick_train_length_2:],ibi[chick_train_length_2:],c='black',linewidth=1,zorder=2)
ax8.scatter(chick_335_t1[chick_train_length_2:],ibi[chick_train_length_2:],s=10,c='black',marker='o',zorder=2)
ax8.scatter(chick_335_t2[::2],pred_ibi_1[::2],s=50,marker='o',facecolors='none',edgecolors='crimson',zorder=3)
ax8.scatter(chick_335_t3,pred_ibi_2,s=50,marker='o',facecolors='none',edgecolors='crimson',zorder=3)

ax8.scatter(t_pd,x_pd,s=220, marker='h',facecolors='white', edgecolors='black',zorder=5)

ax8.set_xlabel('Beat number',font_x,labelpad=0)
ax8.set_ylabel('IBI (s)',font_y,labelpad=-15)
ax8.set_xlim(-15,350)
ax8.set_xticks([0,200,335])
ax8.set_ylim(0.5,1.6)
ax8.set_yticks([0.6,1.5])
ax8.set_yticklabels(['0.6','1.5'])
ax8.tick_params(direction='in')

# ax8.yaxis.set_label_coords(-0.05, 0.48)

ax8.set_title('Beating chick-heart (III)',y=1.02,fontdict=font_title)
ax8.text(-0.125, 1.05,'h',ha='left', transform=ax8.transAxes,fontdict={'family':'DejaVu Sans','size':30,'weight':'bold'})

ax8.tick_params(axis='x', labelsize=18)
ax8.tick_params(axis='y', labelsize=18)

legend_state = mlines.Line2D([], [], color='black', marker='o', markersize=4, linestyle='-', linewidth=2)
legend_train = mlines.Line2D([], [], color='slategrey', marker='o', markersize=4, linestyle='-', linewidth=2)
legend_pstate = mlines.Line2D([], [], markerfacecolor='none',color='crimson', marker='o', markersize=8, linestyle='None', markeredgewidth=1.5)
legend_fold = mlines.Line2D([], [], markerfacecolor='white',color='black', marker='o', markersize=8, linestyle='None', markeredgewidth=1.5)
legend_pd = mlines.Line2D([], [], markerfacecolor='white',color='black', marker='h', markersize=8, linestyle='None', markeredgewidth=1.5)
#legend_start = mlines.Line2D([], [], color='royalblue', marker=(5,1), markersize=8, linestyle='None', markeredgewidth=1.5)

fig.legend(
    handles=[legend_state,legend_pstate,legend_train,legend_fold,legend_arrow,legend_pd],
    labels=['System state','Prediction','Training data','Predicted fold bifurcation',
            'Calculated UAV horizontal thrust','Predicted period-doubling bifurcation'],
    handler_map={type(legend_arrow): HandlerQuiver()},
    loc='upper center',
    bbox_to_anchor=(0.55, 1),
    ncol=3,
    frameon=False,
    markerscale=2.5,
    prop=font_manager.FontProperties(family='Arial Unicode MS', size=22)
)

plt.subplots_adjust(top=0.8, bottom=0.05, left=0.05, right=0.99, hspace=0.3, wspace=0.2)
plt.savefig('../figures/FIG4.0.pdf',format='pdf')







