import pandas
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib import font_manager

font_x = {'family':'Arial','weight':'normal','size': 20}
font_y = {'family':'Arial','weight':'normal','size': 20}
font_title = {'family':'DejaVu Sans','weight':'normal','size': 20, 'style': 'italic'}

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.labelweight'] = 'bold'

df_chick_150_tseries = pandas.read_csv('../chick/chick_data_150.csv')
df_chick_150_pred_1 = pandas.read_csv('../results/chick/chick_150_pred_1.csv')
df_chick_150_pred_2 = pandas.read_csv('../results/chick/chick_150_pred_2.csv')
df_chick_150_gen = pandas.read_csv('../results/chick/chick_150_gen.csv')

df_chick_335_tseries = pandas.read_csv('../chick/chick_data_335.csv')
df_chick_335_pred_1 = pandas.read_csv('../results/chick/chick_335_pred_1.csv')
df_chick_335_pred_2 = pandas.read_csv('../results/chick/chick_335_pred_2.csv')
df_chick_335_gen = pandas.read_csv('../results/chick/chick_335_gen.csv')

df_chick_600_tseries = pandas.read_csv('../chick/chick_data_600.csv')
df_chick_600_pred_1 = pandas.read_csv('../results/chick/chick_600_pred_1.csv')
df_chick_600_pred_2 = pandas.read_csv('../results/chick/chick_600_pred_2.csv')
df_chick_600_gen = pandas.read_csv('../results/chick/chick_600_gen.csv')

chick_150_train_length = 70
chick_335_train_length = 200
chick_600_train_length = 200

fig, axs = plt.subplots(1, 3, figsize=(15,5.5))

ax1, ax2, ax3 = axs[0], axs[1], axs[2]

# ax1

chick_150_t1 = df_chick_150_tseries['Beat number']
chick_150_t2 = df_chick_150_pred_1['Time']
chick_150_t3 = df_chick_150_pred_2['Time']
chick_150_t4 = df_chick_150_tseries[:chick_150_train_length]['Beat number']

ibi = df_chick_150_tseries['IBI (s)']
train_ibi = df_chick_150_gen['gen'][:chick_150_train_length]
pred_ibi_1 = df_chick_150_pred_1['pred']
pred_ibi_2 = df_chick_150_pred_2['pred']

initial_t = chick_150_t1.iloc[0]
end_t = chick_150_t1.iloc[-1]
initial_theta = df_chick_150_gen['theta'].iloc[0] - (df_chick_150_gen['theta'].iloc[1] - df_chick_150_gen['theta'].iloc[0])
end_theta =  initial_theta + (len(chick_150_t1) - 1) * (df_chick_150_gen['theta'].iloc[1] - df_chick_150_gen['theta'].iloc[0])

# period-2 bifurcation

t_pd = initial_t + (0.935438-initial_theta)/(end_theta-initial_theta)*(end_t-initial_t)
print('period-2 150:{}'.format(t_pd))
df_chick_150_gen['distance'] = (df_chick_150_gen['Time'] - t_pd).abs()
closest_row = df_chick_150_gen.loc[df_chick_150_gen['distance'].idxmin()]
x_pd = closest_row['gen']

ax1.plot(chick_150_t1,ibi,c='black',zorder=2)
ax1.scatter(chick_150_t1,ibi,s=10,c='black',marker='o',zorder=2)
ax1.scatter(chick_150_t2[::2],pred_ibi_1[::2],s=50,marker='o',facecolors='none',edgecolors='crimson',zorder=3)
ax1.scatter(chick_150_t3,pred_ibi_2,s=50,marker='o',facecolors='none',edgecolors='crimson',zorder=3)

ax1.scatter(t_pd,x_pd,s=150, marker='h',facecolors='white', edgecolors='black',zorder=5)

ax1.fill_between(chick_150_t4,train_ibi-0.05,train_ibi+0.05,color='silver',alpha=0.9,linewidth=0,zorder=1)

ax1.set_xlabel('Beat number',font_x,labelpad=0)
ax1.set_ylabel('IBI (s)',font_y,labelpad=-15)
ax1.set_xlim(-10,160)
ax1.set_xticks([0,70,150])
ax1.set_ylim(0.3,1.1)
ax1.set_yticks([0.4,1])
ax1.set_yticklabels(['0.4','1'])
ax1.tick_params(direction='in')

# ax1.yaxis.set_label_coords(-0.05, 0.48)

ax1.set_title('Beating chick-heart (IV)',y=1.02,fontdict=font_title)
ax1.text(-0.125, 1,'a',ha='left', transform=ax1.transAxes,fontdict={'family':'DejaVu Sans','size':30,'weight':'bold'})

ax1.tick_params(axis='x', labelsize=18)
ax1.tick_params(axis='y', labelsize=18)

# ax2

chick_335_t1 = df_chick_335_tseries['Beat number']
chick_335_t2 = df_chick_335_pred_1['Time']
chick_335_t3 = df_chick_335_pred_2['Time']
chick_335_t4 = df_chick_335_tseries[:chick_335_train_length]['Beat number']

ibi = df_chick_335_tseries['IBI (s)']
train_ibi = df_chick_335_gen['gen'][:chick_335_train_length]
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

ax2.plot(chick_335_t1,ibi,c='black',zorder=2)
ax2.scatter(chick_335_t1,ibi,s=10,c='black',marker='o',zorder=2)
ax2.scatter(chick_335_t2[::2],pred_ibi_1[::2],s=50,marker='o',facecolors='none',edgecolors='crimson',zorder=3)
ax2.scatter(chick_335_t3,pred_ibi_2,s=50,marker='o',facecolors='none',edgecolors='crimson',zorder=3)

ax2.scatter(t_pd,x_pd,s=150, marker='h',facecolors='white', edgecolors='black',zorder=5)

ax2.fill_between(chick_335_t4,train_ibi-0.05,train_ibi+0.05,color='silver',alpha=0.9,linewidth=0,zorder=1)

ax2.set_xlabel('Beat number',font_x,labelpad=0)
ax2.set_ylabel('IBI (s)',font_y,labelpad=-15)
ax2.set_xlim(-15,350)
ax2.set_xticks([0,200,335])
ax2.set_ylim(0.5,1.6)
ax2.set_yticks([0.6,1.5])
ax2.set_yticklabels(['0.6','1.5'])
ax2.tick_params(direction='in')

# ax2.yaxis.set_label_coords(-0.05, 0.48)

ax2.set_title('Beating chick-heart (V)',y=1.02,fontdict=font_title)
ax2.text(-0.125, 1,'b',ha='left', transform=ax2.transAxes,fontdict={'family':'DejaVu Sans','size':30,'weight':'bold'})

ax2.tick_params(axis='x', labelsize=18)
ax2.tick_params(axis='y', labelsize=18)

# ax3

chick_600_t1 = df_chick_600_tseries['Beat number']
chick_600_t2 = df_chick_600_pred_1['Time']
chick_600_t3 = df_chick_600_pred_2['Time']
chick_600_t4 = df_chick_600_tseries[:chick_600_train_length]['Beat number']

ibi = df_chick_600_tseries['IBI (s)']
train_ibi = df_chick_600_gen['gen'][:chick_600_train_length]
pred_ibi_1 = df_chick_600_pred_1['pred']
pred_ibi_2 = df_chick_600_pred_2['pred']

initial_t = chick_600_t1.iloc[0]
end_t = chick_600_t1.iloc[-1]
initial_theta = df_chick_600_gen['theta'].iloc[0] - (df_chick_600_gen['theta'].iloc[1] - df_chick_600_gen['theta'].iloc[0])
end_theta =  initial_theta + (len(chick_600_t1) - 1) * (df_chick_600_gen['theta'].iloc[1] - df_chick_600_gen['theta'].iloc[0])

# period-2 bifurcation

t_pd = initial_t + (0.460545-initial_theta)/(end_theta-initial_theta)*(end_t-initial_t)
print('period-2 600:{}'.format(t_pd))
df_chick_600_gen['distance'] = (df_chick_600_gen['Time'] - t_pd).abs()
closest_row = df_chick_600_gen.loc[df_chick_600_gen['distance'].idxmin()]
x_pd = closest_row['gen']

ax3.plot(chick_600_t1,ibi,c='black',zorder=2)
ax3.scatter(chick_600_t1,ibi,s=10,c='black',marker='o',zorder=2)
ax3.scatter(chick_600_t2[::2],pred_ibi_1[::2],s=50,marker='o',facecolors='none',edgecolors='crimson',zorder=3)
ax3.scatter(chick_600_t3,pred_ibi_2,s=50,marker='o',facecolors='none',edgecolors='crimson',zorder=3)

ax3.scatter(t_pd,x_pd,s=150, marker='h',facecolors='white', edgecolors='black',zorder=5)

ax3.fill_between(chick_600_t4,train_ibi-0.05,train_ibi+0.05,color='silver',alpha=0.9,linewidth=0,zorder=1)

ax3.set_xlabel('Beat number',font_x,labelpad=0)
ax3.set_ylabel('IBI (s)',font_y,labelpad=-15)
ax3.set_xlim(-40,640)
ax3.set_xticks([0,200,600])
ax3.set_ylim(0.4,1.4)
ax3.set_yticks([0.5,1.3])
ax3.set_yticklabels(['0.5','1.3'])
ax3.tick_params(direction='in')

# ax3.yaxis.set_label_coords(-0.05, 0.48)

ax3.set_title('Beating chick-heart (VI)',y=1.02,fontdict=font_title)
ax3.text(-0.125, 1,'c',ha='left', transform=ax3.transAxes,fontdict={'family':'DejaVu Sans','size':30,'weight':'bold'})

ax3.tick_params(axis='x', labelsize=18)
ax3.tick_params(axis='y', labelsize=18)

legend_state = mlines.Line2D([], [], color='black', marker='o', markersize=3, linestyle='-', markeredgewidth=1.5)
legend_pstate = mlines.Line2D([], [], markerfacecolor='none',color='crimson', marker='o', markersize=5, linestyle='None', markeredgewidth=1.5)
legend_pd = mlines.Line2D([], [], markerfacecolor='white',color='black', marker='h', markersize=5, linestyle='None', markeredgewidth=1.5)
legend_fill = mpatches.Patch(color='silver', alpha=0.9, linewidth=0)

fig.legend(handles=[legend_state,legend_pstate,legend_pd,legend_fill],
           labels=['Inter-beat intervals','Prediction','Predicted period-doubling bifurcation','Training data'],
           loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=4, frameon=False, markerscale=2.5,
           prop=font_manager.FontProperties(family='Arial Unicode MS', size=18))

plt.subplots_adjust(top=0.8, bottom=0.1, left=0.04, right=0.99, wspace=0.2)
plt.savefig('../figures/SFIG5.pdf',format='pdf')
plt.savefig('/Users/zhugchzo/Desktop/3paper_fig/SFIG5.png',format='png',dpi=600)







