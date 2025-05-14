import pandas
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

font_x = {'family':'Arial','weight':'normal','size': 20}
font_y = {'family':'Arial','weight':'normal','size': 20}

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.labelweight'] = 'bold'

df_tseries = pandas.read_csv('../UAV/UAV_data.csv')
df_pred = pandas.read_csv('../results/UAV/UAV_pred.csv')
df_gen = pandas.read_csv('../results/UAV/UAV_gen.csv')

train_length = len(df_tseries) - len(df_pred) - 1

x = df_tseries['x']
y = df_tseries['y']
pred_x = df_pred['pred_x']
pred_y = df_pred['pred_y']
gen_x = df_gen['gen_x']
gen_y = df_gen['gen_y']
train_x = df_tseries['x'][:train_length]
train_y = df_tseries['y'][:train_length]

fig, axs = plt.subplots(1, 2, figsize=(12,6))
(ax1, ax2) = axs.flatten()

# ax1

ax1.plot(x,y,c='black',zorder=2)
ax1.scatter(x,y,s=10,c='black',marker='o',zorder=2)
ax1.scatter(x.iloc[0], y.iloc[0], color='royalblue',s=150,zorder=4,marker=(5,1))  # Mark the start point
ax1.scatter(pred_x,pred_y,s=50,marker='o',facecolors='none',edgecolors='crimson',zorder=3)

ax1.fill_between(train_x, train_y-0.05, train_y+0.05, color='silver', alpha=0.9, linewidth=0, zorder=1)

ax1.annotate(
    text='',
    xy=(2.3, 0.82),
    xytext=(2, 0.7),
    arrowprops=dict(arrowstyle='->', color='black', lw=3)
)

ax1.annotate(
    text='',
    xy=(6.3, 1.2),
    xytext=(6, 1.3),
    arrowprops=dict(arrowstyle='->', color='black', lw=3)
)

ax1.annotate(
    text='',
    xy=(8.7, 1.2),
    xytext=(8.1, 1.1),
    arrowprops=dict(arrowstyle='->', color='black', lw=3)
)

legend_state = mlines.Line2D([], [], color='black', marker='o', markersize=3, linestyle='-', markeredgewidth=1.5)
legend_pstate = mlines.Line2D([], [], markerfacecolor='none',color='crimson', marker='o', markersize=5, linestyle='None', markeredgewidth=1.5)
legend_start = mlines.Line2D([], [], color='royalblue', marker=(5,1), markersize=6, linestyle='None', markeredgewidth=1.5)
legend_fill = mpatches.Patch(color='silver', alpha=0.9, linewidth=0)

ax1.legend(handles=[legend_state,legend_pstate,legend_start,legend_fill],labels=['Odometry path','Prediction','Start','Training data'],loc='center', frameon=False, bbox_to_anchor=(0.72, 0.25), markerscale=2.5,prop={'size':18})

ax1.set_xlabel('X Position (m)',font_x,labelpad=-13)
ax1.set_ylabel('Y Position (m)',font_y,labelpad=-15)
ax1.set_xlim(-1,11.5)
ax1.set_xticks([0,10])
ax1.set_ylim(0.1,1.5)
ax1.set_yticks([0.4,1.2])

ax1.text(-0.075, 0.95,'a',ha='left', transform=ax1.transAxes,fontdict={'family':'DejaVu Sans','size':20,'weight':'bold'})

ax1.tick_params(axis='x', labelsize=18)
ax1.tick_params(axis='y', labelsize=18)
ax1.tick_params(direction='in')

# ax2

ax2.plot(x,y,c='black',zorder=2)
ax2.scatter(x,y,s=10,c='black',marker='o',zorder=2)
ax2.scatter(x.iloc[0], y.iloc[0], color='royalblue',s=150,zorder=4,marker=(5,1))  # Mark the start point
ax2.scatter(gen_x,gen_y,s=50,marker='o',facecolors='none',edgecolors='darkorange',zorder=3)

ax2.annotate(
    text='',
    xy=(4.1, 0.92),
    xytext=(3.8, 0.8),
    arrowprops=dict(arrowstyle='->', color='black', lw=3)
)

ax2.annotate(
    text='',
    xy=(6.3, 1.2),
    xytext=(6, 1.3),
    arrowprops=dict(arrowstyle='->', color='black', lw=3)
)

ax2.annotate(
    text='',
    xy=(8.7, 1.2),
    xytext=(8.1, 1.1),
    arrowprops=dict(arrowstyle='->', color='black', lw=3)
)

legend_state = mlines.Line2D([], [], color='black', marker='o', markersize=3, linestyle='-', markeredgewidth=1.5)
legend_gstate = mlines.Line2D([], [], markerfacecolor='none',color='darkorange', marker='o', markersize=5, linestyle='None', markeredgewidth=1.5)
legend_start = mlines.Line2D([], [], color='royalblue',marker=(5,1), markersize=6, linestyle='None', markeredgewidth=1.5)

ax2.legend(handles=[legend_state,legend_gstate,legend_start],labels=['Odometry path','Generation','Start'],loc='center', frameon=False, bbox_to_anchor=(0.72, 0.25), markerscale=2.5,prop={'size':18})

ax2.set_xlabel('X Position (m)',font_x,labelpad=-13)
ax2.set_xlim(-1,11.5)
ax2.set_xticks([0,10])
ax2.set_ylim(0.1,1.5)
ax2.set_yticks([0.4,1.2])

ax2.text(-0.075, 0.95,'b',ha='left', transform=ax2.transAxes,fontdict={'family':'DejaVu Sans','size':20,'weight':'bold'})

ax2.tick_params(axis='x', labelsize=18)
ax2.tick_params(axis='y', labelsize=18)
ax2.tick_params(direction='in')

plt.subplots_adjust(top=0.98, bottom=0.07, left=0.05, right=0.99, wspace=0.15)
plt.savefig('../figures/FIG3.pdf',format='pdf')
plt.savefig('/Users/zhugchzo/Desktop/3paper_fig/FIG3.png',format='png',dpi=600)



