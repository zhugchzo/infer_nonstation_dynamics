import pandas
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib import font_manager
from matplotlib.ticker import MaxNLocator
import networkx as nx

font_x = font_manager.FontProperties(family='Arial', size=24, weight='normal')
font_y = font_manager.FontProperties(family='Arial', size=24, weight='normal')
font_title = {'family':'DejaVu Sans','weight':'light','size': 18, 'style': 'italic'}

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.labelweight'] = 'bold'

df_tseries = pandas.read_csv('../fish/fish_data.csv')
df_network = pandas.read_csv('../fish/fish_network.csv')
df_pred = pandas.read_csv('../results/fish/fish_pred.csv')

col1 = ['Aurelia.sp', 'Plotosus.japonicus', 'Sebastes.cheni', 'Trachurus.japonicus', 'Girella.punctata',
       'Pseudolabrus.sieboldi', 'Parajulis.poecilopterus', 'Halichoeres.tenuispinnis', 'Chaenogobius.gulosus',
       'Pterogobius.zonoleucus', 'Tridentiger.trigonocephalus', 'Siganus.fuscescens', 'Sphyraena.pinguis', 'Rudarius.ercodes']

col2 = ['pred_Aurelia.sp', 'pred_Plotosus.japonicus', 'pred_Sebastes.cheni', 'pred_Trachurus.japonicus', 'pred_Girella.punctata',
       'pred_Pseudolabrus.sieboldi', 'pred_Parajulis.poecilopterus', 'pred_Halichoeres.tenuispinnis', 'pred_Chaenogobius.gulosus',
       'pred_Pterogobius.zonoleucus', 'pred_Tridentiger.trigonocephalus', 'pred_Siganus.fuscescens', 'pred_Sphyraena.pinguis', 'pred_Rudarius.ercodes']

name1 = ['Jellyfish (Aurelia sp.)', 'Plotosus japonicus', 'Sebastes cheni', 'Trachurus japonicus', 'Girella punctata',
       'Pseudolabrus sieboldi', 'Parajulis poecilopterus', 'Halichoeres tenuispinnis', 'Chaenogobius gulosus',
       'Pterogobius zonoleucus', 'Tridentiger trigonocephalus', 'Siganus fuscescens', 'Sphyraena pinguis', 'Rudarius ercodes']

name2 = ['J.', 'P.j.', 'S.c.', 'T.j.', 'G.p.', 'P.s.', 'P.p.', 'H.t.', 'C.g.', 'P.z.', 'T.t.', 'S.f.', 'S.p.', 'R.e.']

# the number of node
N = len(col1)

data_tseries = df_tseries[col1].values
pred_tseries = df_pred[['Time'] + col2].values
data_network = df_network[col1].values

length = len(data_tseries)
time = np.arange(1, 1 + length, 1)

show_length = 158
train_length = 254

G = nx.DiGraph()

nodes = [i for i in range(N)]

G.add_nodes_from(nodes)

edge = []
for i in range(N):
    for j in range(N):
        if data_network[j, i] == 1:
            edge.append((j, i))

G.add_edges_from(edge)

pos = {}
radius = 2
for i, node in enumerate(nodes):
    angle = -2 * np.pi * i / N + np.pi / 2
    x = radius * np.cos(angle)
    y = radius * np.sin(angle)
    pos[node] = (x, y)

fig, axs = plt.subplots(4, 4, figsize=(18,12))

cnt = 0

for i in range(4):
    if i == 0:
        for j in range(4):
            if j < 2:
                ax = axs[i, j]
                ax.plot(time[show_length:], data_tseries[show_length:,cnt],c='black',zorder=2)
                ax.scatter(pred_tseries[:,0], pred_tseries[:,cnt+1], s=15, marker='o', facecolors='none', edgecolors='crimson',zorder=3)
                ax.plot(pred_tseries[:,0], pred_tseries[:,cnt+1],c='crimson',zorder=3)
                ax.axvspan(time[show_length:][0],time[show_length:][23], color='silver', alpha=0.3, linewidth=0,zorder=1)
                ax.axvspan(time[show_length:][48],time[show_length:][71], color='silver', alpha=0.3, linewidth=0,zorder=1)
                ax.axvspan(time[show_length:][96],time[show_length:][119], color='silver', alpha=0.3, linewidth=0,zorder=1)
                # ax.axvspan(time[show_length:][24],time[show_length:][47], color='silver', alpha=0.3, linewidth=0,zorder=1)
                # ax.axvspan(time[show_length:][72],time[show_length:][95], color='silver', alpha=0.3, linewidth=0,zorder=1)
                # ax.axvspan(time[show_length:][120],time[show_length:][-1], color='silver', alpha=0.3, linewidth=0,zorder=1)
                ax.yaxis.set_major_locator(MaxNLocator(nbins=4)) 
                ax.set_title(name1[cnt],fontdict=font_title)
                ax.set_xticks([])
                ax.tick_params(axis='y', labelsize=14) 
                cnt += 1

            elif j == 2:
                ax = axs[i, j]
                ax.axis('off')

                legend_gt = mlines.Line2D([], [], color='black', marker='none', linestyle='-', linewidth=2)
                legend_pred = mlines.Line2D([], [], markerfacecolor='none',color='crimson', marker='o', markersize=6, linestyle='-', markeredgewidth=2, linewidth=2)
                ax.legend(handles=[legend_gt,legend_pred],labels=['Abundance','Prediction'],loc='center',frameon=False, markerscale=2, prop={'size':24})

            elif j == 3:
                ax = axs[i, j]
                ax.axis('off')
                ax.set_aspect('equal')

                nx.draw(G, pos, ax=ax,
                with_labels=False,
                node_size=70,
                node_color='darkslateblue',
                edgecolors='darkslateblue',
                edge_color='silver',
                arrowstyle='->',
                arrowsize=7,width=2)

                label_radius1 = radius * 1.2
                label_radius2 = radius * 1.25
                label_radius3 = radius * 1.3
                for i, node in enumerate(nodes):
                    if i in [4]:
                        angle = -2 * np.pi * i / N + np.pi / 2
                        label_x = label_radius3 * np.cos(angle)
                        label_y = label_radius3 * np.sin(angle)
                        ax.text(label_x, label_y, name2[i], fontsize=13, fontstyle='italic', fontfamily='DejaVu Sans', ha='center', va='center')
                    elif i in [2,3,5,11,12]:
                        angle = -2 * np.pi * i / N + np.pi / 2
                        label_x = label_radius2 * np.cos(angle)
                        label_y = label_radius2 * np.sin(angle)
                        ax.text(label_x, label_y, name2[i], fontsize=13, fontstyle='italic', fontfamily='DejaVu Sans', ha='center', va='center')
                    else:                       
                        angle = -2 * np.pi * i / N + np.pi / 2
                        label_x = label_radius1 * np.cos(angle)
                        label_y = label_radius1 * np.sin(angle)
                        ax.text(label_x, label_y, name2[i], fontsize=13, fontstyle='italic', fontfamily='DejaVu Sans', ha='center', va='center')

    elif i == 3:
        for j in range(4):
            ax = axs[i, j]
            ax.plot(time[show_length:], data_tseries[show_length:,cnt],c='black',zorder=2)
            ax.scatter(pred_tseries[:,0], pred_tseries[:,cnt+1], s=15, marker='o', facecolors='none', edgecolors='crimson',zorder=3)
            ax.plot(pred_tseries[:,0], pred_tseries[:,cnt+1],c='crimson',zorder=3)
            ax.axvspan(time[show_length:][0],time[show_length:][23], color='silver', alpha=0.3, linewidth=0,zorder=1)
            ax.axvspan(time[show_length:][48],time[show_length:][71], color='silver', alpha=0.3, linewidth=0,zorder=1)
            ax.axvspan(time[show_length:][96],time[show_length:][119], color='silver', alpha=0.3, linewidth=0,zorder=1)
            # ax.axvspan(time[show_length:][24],time[show_length:][47], color='silver', alpha=0.3, linewidth=0,zorder=1)
            # ax.axvspan(time[show_length:][72],time[show_length:][95], color='silver', alpha=0.3, linewidth=0,zorder=1)
            # ax.axvspan(time[show_length:][120],time[show_length:][-1], color='silver', alpha=0.3, linewidth=0,zorder=1)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.set_title(name1[cnt],fontdict=font_title)
            positions = [time[show_length:][11], time[show_length:][35], time[show_length:][59], time[show_length:][83], time[show_length:][107], time[show_length:][-1]]
            years = ['2009','2010', '2011', '2012', '2013', '  2014']
            ax.set_xticks(positions)
            ax.set_xticklabels(years, fontsize=14, fontfamily='Arial') 
            ax.tick_params(axis='x', which='both', bottom=False)
            ax.tick_params(axis='y', labelsize=14) 
            cnt += 1        

    else:
        for j in range(4):
            ax = axs[i, j]
            ax.plot(time[show_length:], data_tseries[show_length:,cnt],c='black',zorder=2)
            ax.scatter(pred_tseries[:,0], pred_tseries[:,cnt+1], s=15, marker='o', facecolors='none', edgecolors='crimson',zorder=3)
            ax.plot(pred_tseries[:,0], pred_tseries[:,cnt+1],c='crimson',zorder=3)
            ax.axvspan(time[show_length:][0],time[show_length:][23], color='silver', alpha=0.3, linewidth=0,zorder=1)
            ax.axvspan(time[show_length:][48],time[show_length:][71], color='silver', alpha=0.3, linewidth=0,zorder=1)
            ax.axvspan(time[show_length:][96],time[show_length:][119], color='silver', alpha=0.3, linewidth=0,zorder=1)
            # ax.axvspan(time[show_length:][24],time[show_length:][47], color='silver', alpha=0.3, linewidth=0,zorder=1)
            # ax.axvspan(time[show_length:][72],time[show_length:][95], color='silver', alpha=0.3, linewidth=0,zorder=1)
            # ax.axvspan(time[show_length:][120],time[show_length:][-1], color='silver', alpha=0.3, linewidth=0,zorder=1)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.set_title(name1[cnt],fontdict=font_title)
            ax.set_xticks([])
            ax.tick_params(axis='y', labelsize=14) 
            cnt += 1

fig.supxlabel('Year',x=0.535, y=0, fontproperties=font_x)
fig.supylabel('Abundance',x=0.005, y=0.55, fontproperties=font_y)

plt.subplots_adjust(top=0.96, bottom=0.06, left=0.055, right=0.99, hspace=0.25, wspace=0.16)
plt.savefig('../figures/FIG4.pdf',format='pdf')
plt.savefig('/Users/zhugchzo/Desktop/3paper_fig/FIG4.png',format='png',dpi=600)
