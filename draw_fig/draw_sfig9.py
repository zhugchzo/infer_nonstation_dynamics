import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from scipy.stats import kendalltau

font_x = {'family':'Arial','weight':'normal','size': 28}
font_y1 = {'family':'Arial','weight':'normal','size': 20}
font_y2 = {'family':'Arial','weight':'normal','size': 24}

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.labelweight'] = 'bold'

df_dom_eval = pd.read_csv('../results/fish/dom_eval.csv')
df_var = pd.read_csv('../results/fish/fish_var.csv')

col = ['Aurelia.sp', 'Plotosus.japonicus', 'Sebastes.cheni', 'Trachurus.japonicus', 'Girella.punctata',
       'Pseudolabrus.sieboldi', 'Parajulis.poecilopterus', 'Halichoeres.tenuispinnis', 'Chaenogobius.gulosus',
       'Pterogobius.zonoleucus', 'Tridentiger.trigonocephalus', 'Siganus.fuscescens', 'Sphyraena.pinguis', 'Rudarius.ercodes']

N = len(col)

var_tseries = df_var.values.reshape(-1)

t = df_dom_eval['Time'][:275]
dom_eval = df_dom_eval['dom_eval'].values.reshape(-1)[:275]

tau, p_value = kendalltau(var_tseries, dom_eval)

print(f"Kendall's tau: {tau}")
print(f"P-value: {p_value}")

fig, ax1 = plt.subplots(figsize=(12,6))

ax1.plot(t, dom_eval, c='black', linewidth=2, zorder=2)

for i in [0,2,4,6,8]:
       ax1.axvspan(t[14+24*i],t[37+24*i], color='silver', alpha=0.3, linewidth=0, zorder=1)
ax1.axvspan(t[14+24*10],t[274], color='silver', alpha=0.3, linewidth=0, zorder=1)

positions = list()
for i in [0,2,4,6,8,10]:
       positions.append(t[25+24*i])

years = ['2003','2005', '2007', '2009', '2011', '2013']
ax1.set_xticks(positions)
ax1.set_xticklabels(years, fontsize=18) 
ax1.tick_params(axis='x', which='both', bottom=False)
ax1.tick_params(direction='in')

ax1.set_xlabel('Year',font_x,labelpad=10)

ax1.set_yticks([0,6000,12000])
ax1.set_yticklabels(['0','6','12'], fontsize=18) 

ax1.set_ylabel(r'Modulus of the dominant eigenvalue ($10^3$)',font_y1,labelpad=10)

ax2 = ax1.twinx()
ax2.plot(t, var_tseries, c='crimson', linestyle='--', linewidth=2, zorder=2)

ax2.set_yticks([0,0.75,1.5])
ax2.set_yticklabels(['0','0.75','1.5'], fontsize=18)
ax2.tick_params(direction='in')

ax2.set_ylabel('Variance',font_y2,labelpad=15)

legend_dom_eval = mlines.Line2D([], [], color='black', marker='none', linestyle='-', linewidth=2)
legend_var = mlines.Line2D([], [], color='crimson', marker='none', linestyle='--', linewidth=2)
ax1.legend(handles=[legend_dom_eval,legend_var],labels=['Modulus of the \ndominant eigenvalue','Variance'],loc='center',frameon=False,bbox_to_anchor=(0.85, 0.9), prop={'size':15})

plt.subplots_adjust(top=0.98, bottom=0.14, left=0.07, right=0.9)
plt.savefig('../figures/SFIG9.pdf',format='pdf')
plt.savefig('/Users/zhugchzo/Desktop/3paper_fig/SFIG9.png',format='png',dpi=600)

