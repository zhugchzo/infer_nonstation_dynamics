import pandas
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

font_x = {'family':'Arial','weight':'medium','size': 24}
font_y = {'family':'Arial','weight':'medium','size': 24}

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18

df_network = pandas.read_csv('../fish/fish_network.csv')

col = ['Aurelia.sp', 'Plotosus.japonicus', 'Sebastes.cheni', 'Trachurus.japonicus', 'Girella.punctata',
       'Pseudolabrus.sieboldi', 'Parajulis.poecilopterus', 'Halichoeres.tenuispinnis', 'Chaenogobius.gulosus',
       'Pterogobius.zonoleucus', 'Tridentiger.trigonocephalus', 'Siganus.fuscescens', 'Sphyraena.pinguis', 'Rudarius.ercodes']

name = ['J.', 'P.j.', 'S.c.', 'T.j.', 'G.p.', 'P.s.', 'P.p.', 'H.t.', 'C.g.', 'P.z.', 'T.t.', 'S.f.', 'S.p.', 'R.e.']

# the number of node
N = len(col)

# draw
fig, axs = plt.subplots(1, 3, figsize=(15,4.5))

ax1, ax2, ax3 = axs[0], axs[1], axs[2]

# ax1
data_network = df_network[col].values

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

ax1.axis('off')
ax1.set_aspect('equal')

nx.draw(G, pos, ax=ax1,
with_labels=False,
node_size=100,
node_color='darkslateblue',
edgecolors='darkslateblue',
edge_color='silver',
arrowstyle='->',
arrowsize=12,width=2)

label_radius1 = radius * 1.15
label_radius2 = radius * 1.2
label_radius3 = radius * 1.25
for i, node in enumerate(nodes):
    if i in [4]:
        angle = -2 * np.pi * i / N + np.pi / 2
        label_x = label_radius3 * np.cos(angle)
        label_y = label_radius3 * np.sin(angle)
        ax1.text(label_x, label_y, name[i], fontsize=13, fontstyle='italic', fontfamily='DejaVu Sans', ha='center', va='center')
    elif i in [2,3,5,11,12]:
        angle = -2 * np.pi * i / N + np.pi / 2
        label_x = label_radius2 * np.cos(angle)
        label_y = label_radius2 * np.sin(angle)
        ax1.text(label_x, label_y, name[i], fontsize=13, fontstyle='italic', fontfamily='DejaVu Sans', ha='center', va='center')
    else:                       
        angle = -2 * np.pi * i / N + np.pi / 2
        label_x = label_radius1 * np.cos(angle)
        label_y = label_radius1 * np.sin(angle)
        ax1.text(label_x, label_y, name[i], fontsize=13, fontstyle='italic', fontfamily='DejaVu Sans', ha='center', va='center')

ax1.text(-0.15, 0.96,'b',ha='left', transform=ax1.transAxes,fontdict={'family':'Arial','size':30,'weight':'bold'})

# ax2
in_degree = np.sum(data_network == 1, axis=0)
in_degree_values, in_degree_counts = np.unique(in_degree, return_counts=True)

ax2.plot(in_degree_values,in_degree_counts,c='darkslateblue',linewidth=3,alpha=0.8)
ax2.scatter(in_degree_values,in_degree_counts,c='darkslateblue',s=100)

ax2.set_xticks([0,1,2])

ax2.set_ylim(3.8,6.2)
ax2.set_yticks([4,5,6])
ax2.tick_params(direction='in')

ax2.set_xlabel('In-degree',font_x)
ax2.set_ylabel('Frequency',font_y,labelpad=15)

ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)

ax2.text(-0.15, 0.96,'c',ha='left', transform=ax2.transAxes,fontdict={'family':'Arial','size':30,'weight':'bold'})

# ax3
in_degree = np.sum(data_network == 1, axis=1)
in_degree_values, in_degree_counts = np.unique(in_degree, return_counts=True)

ax3.plot(in_degree_values,in_degree_counts,c='darkslateblue',linewidth=3,alpha=0.8)
ax3.scatter(in_degree_values,in_degree_counts,c='darkslateblue',s=100)

ax3.set_xticks([0,1,2,3])

ax3.set_ylim(0.5,7.5)
ax3.set_yticks([1,2,3,4,5,6,7])
ax3.tick_params(direction='in')

ax3.set_xlabel('Out-degree',font_x)
ax3.set_ylabel('Frequency',font_y,labelpad=15)

ax3.spines['right'].set_visible(False)
ax3.spines['top'].set_visible(False)

ax3.text(-0.15, 0.96,'d',ha='left', transform=ax3.transAxes,fontdict={'family':'Arial','size':30,'weight':'bold'})

plt.subplots_adjust(top=0.9, bottom=0.15, left=0.04, right=0.98, hspace=0.25, wspace=0.3)
plt.savefig('../figures/SFIG8.2.pdf',format='pdf')