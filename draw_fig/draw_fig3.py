import pandas
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

font_x = {'family':'Arial','weight':'normal','size': 20}
font_y = {'family':'Arial','weight':'normal','size': 20}

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.labelweight'] = 'bold'

df_tseries = pandas.read_csv('../mitochondria/mitochondria_data.csv')
df_pred = pandas.read_csv('../results/mitochondria/mitochondria_pred.csv')
df_gen = pandas.read_csv('../results/mitochondria/mitochondria_gen.csv')

train_length = len(df_tseries) - len(df_pred) - 1

t1 = df_tseries['Time']
t2 = df_pred['Time']
t3 = df_tseries[:train_length]['Time']

rfr = df_tseries['Relative fluorescence ratio']
# train_rfr = df_tseries[:train_length]['Relative fluorescence ratio']
train_rfr = df_gen['gen'][:train_length]
pred_rfr = df_pred['pred']
gen_rfr = df_gen['gen']

initial_t = t1.iloc[0]
end_t = t1.iloc[-1]
initial_theta = df_gen['theta'].iloc[0] - (df_gen['theta'].iloc[1] - df_gen['theta'].iloc[0])
end_theta =  df_gen['theta'].iloc[-1]

fig, axs = plt.subplots(1, 2, figsize=(12,6))
(ax1, ax2) = axs.flatten()

# fold bifurcation

t_fold = initial_t + (1.24329-initial_theta)/(end_theta-initial_theta)*(end_t-initial_t)
# print(t_fold)
df_gen['distance'] = (df_gen['Time'] - t_fold).abs()
closest_row = df_gen.loc[df_gen['distance'].idxmin()]
x_fold = closest_row['gen']

# ax1

ax1.plot(t1,rfr,c='black',zorder=2)
ax1.scatter(t1,rfr,s=10,c='black',marker='o',zorder=2)
ax1.scatter(t2,pred_rfr,s=50,marker='o',facecolors='none',edgecolors='crimson',zorder=3)

ax1.scatter(t_fold,x_fold,s=180, marker='o',facecolors='white', edgecolors='black',zorder=5)

ax1.fill_between(t3,train_rfr-0.005,train_rfr+0.005,color='silver',alpha=0.9,linewidth=0,zorder=1)

legend_state = mlines.Line2D([], [], color='black', marker='o', markersize=3, linestyle='-', markeredgewidth=1.5)
legend_pstate = mlines.Line2D([], [], markerfacecolor='none',color='crimson', marker='o', markersize=5, linestyle='None', markeredgewidth=1.5)
legend_fold = mlines.Line2D([], [], markerfacecolor='white',color='black', marker='o', markersize=5, linestyle='None', markeredgewidth=1.5)
legend_fill = mpatches.Patch(color='silver', alpha=0.9, linewidth=0)

ax1.legend(handles=[legend_state,legend_pstate,legend_fold,legend_fill],labels=['ATP concentration','Prediction','Predicted fold bifurcation','Training data'],loc='center', frameon=False, bbox_to_anchor=(0.35, 0.3), markerscale=2.5,prop={'size':18})

ax1.set_xlabel('Time (min)',font_x,labelpad=-13)
ax1.set_ylabel('Relative fluorescence ratio (%)',font_y)
ax1.set_xlim(1.8,5.1)
ax1.set_xticks([2,5])
ax1.set_ylim(0.71,0.9)
ax1.set_yticks([0.72,0.88])
ax1.set_yticklabels(['72','88'])
ax1.tick_params(direction='in')

ax1.yaxis.set_label_coords(-0.05, 0.48)

ax1.text(-0.075, 0.96,'a',ha='left', transform=ax1.transAxes,fontdict={'family':'DejaVu Sans','size':20,'weight':'bold'})

ax1.tick_params(axis='x', labelsize=18)
ax1.tick_params(axis='y', labelsize=18)

# ax2

ax2.plot(t1,rfr,c='black',zorder=2)
ax2.scatter(t1,rfr,s=10,c='black',marker='o',zorder=2)
ax2.scatter(t1[1:],gen_rfr,s=50,marker='o',facecolors='none',edgecolors='darkorange',zorder=3)

ax2.scatter(t_fold,x_fold,s=180, marker='o',facecolors='white', edgecolors='black',zorder=5)

legend_state = mlines.Line2D([], [], color='black', marker='o', markersize=3, linestyle='-', markeredgewidth=1.5)
legend_gstate = mlines.Line2D([], [], markerfacecolor='none',color='darkorange', marker='o', markersize=5, linestyle='None', markeredgewidth=1.5)
legend_fold = mlines.Line2D([], [], markerfacecolor='white',color='black', marker='o', markersize=5, linestyle='None', markeredgewidth=1.5)

ax2.legend(handles=[legend_state,legend_gstate,legend_fold],labels=['ATP concentration','Generation','Predicted fold bifurcation'],loc='center', frameon=False, bbox_to_anchor=(0.35, 0.3), markerscale=2.5,prop={'size':18})

ax2.set_xlabel('Time (min)',font_x,labelpad=-13)
ax2.set_xlim(1.8,5.1)
ax2.set_xticks([2,5])
ax2.set_ylim(0.71,0.9)
ax2.set_yticks([0.72,0.88])
ax2.set_yticklabels(['72','88'])
ax2.tick_params(direction='in')

ax2.text(-0.075, 0.96,'b',ha='left', transform=ax2.transAxes,fontdict={'family':'DejaVu Sans','size':20,'weight':'bold'})

ax2.tick_params(axis='x', labelsize=18)
ax2.tick_params(axis='y', labelsize=18)

plt.subplots_adjust(top=0.98, bottom=0.07, left=0.05, right=0.99, wspace=0.15)
plt.savefig('../figures/FIG3.pdf',format='pdf')
plt.savefig('/Users/zhugchzo/Desktop/3paper_fig/FIG3.png',format='png',dpi=600)







