import pandas
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib import font_manager
from matplotlib.ticker import MaxNLocator

def encode_date_tag(tag,min,delta): # this function is used to transform date to theta

    month, part = tag.split('-')[-2], tag.split('-')[-1]
    month = int(month)  

    if 2 <= month < 8:
        if part == "early":
            return (month - 2) * 2*delta + min  # early
        elif part == "late":
            return (month - 2) * 2*delta + min + delta  # late
        
    elif 9 <= month <= 12:
        if part == "early":
            return -(month - 8) * 2*delta + min + 12*delta  # early
        elif part == "late":
            return -(month - 8) * 2*delta + min + 11*delta  # late

    elif month == 1:
        if part == "early":
            return min + 2*delta  # early
        elif part == "late":
            return min + 1*delta  # late 
            
    elif month == 8:
        if part == "early":
            return min + 12*delta  # early
        elif part == "late":
            return min + 11*delta  # late

font_x = font_manager.FontProperties(family='Arial', size=24, weight='normal')
font_y = font_manager.FontProperties(family='Arial', size=24, weight='normal')
font_y1 = {'family':'Arial','weight':'normal','size': 14}
font_title = {'family':'DejaVu Sans','weight':'light','size': 18, 'style': 'italic'}

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.labelweight'] = 'bold'

df_tseries = pandas.read_csv('../fish/fish_data.csv')

col = ['Aurelia.sp', 'Plotosus.japonicus', 'Sebastes.cheni', 'Trachurus.japonicus', 'Girella.punctata',
       'Pseudolabrus.sieboldi', 'Parajulis.poecilopterus', 'Halichoeres.tenuispinnis', 'Chaenogobius.gulosus',
       'Pterogobius.zonoleucus', 'Tridentiger.trigonocephalus', 'Siganus.fuscescens', 'Sphyraena.pinguis', 'Rudarius.ercodes']

name = ['Jellyfish (Aurelia sp.)', 'Plotosus japonicus', 'Sebastes cheni', 'Trachurus japonicus', 'Girella punctata',
       'Pseudolabrus sieboldi', 'Parajulis poecilopterus', 'Halichoeres tenuispinnis', 'Chaenogobius gulosus',
       'Pterogobius zonoleucus', 'Tridentiger trigonocephalus', 'Siganus fuscescens', 'Sphyraena pinguis', 'Rudarius ercodes']

# the number of node
N = len(col)

data_tseries = df_tseries[col].values
temperature_tseries = df_tseries['surf.t'].values
theta_tseries = (df_tseries['date_tag'].apply(lambda x: encode_date_tag(x, min=0, delta=1))).values

length = len(data_tseries)
time = np.arange(1, 1 + length, 1)

fig, axs = plt.subplots(4, 4, figsize=(18,12))

cnt = 0

for i in range(4):
    if i == 0:
        for j in range(4):
            if j < 2:
                ax = axs[i, j]
                ax.plot(time, data_tseries[:,cnt],c='black',zorder=2)
                for count_y in range(6):
                    ax.axvspan(time[14+48*count_y],time[14+48*count_y+23], color='silver', alpha=0.3, linewidth=0, zorder=1)
                ax.yaxis.set_major_locator(MaxNLocator(nbins=4)) 
                ax.set_title(name[cnt],fontdict=font_title)
                ax.set_xticks([])
                ax.tick_params(axis='y', labelsize=14)
                cnt += 1

            elif j == 2:
                ax1 = axs[i, j]
                ax1.plot(time, temperature_tseries,c='royalblue',zorder=2)
                for count_y in range(6):
                    ax1.axvspan(time[14+48*count_y],time[14+48*count_y+23], color='silver', alpha=0.3, linewidth=0, zorder=1)
                ax1.set_xticks([])
                ax1.set_ylabel('Water temperature',font_y1)

                ax2 = ax1.twinx()
                ax2.plot(time, theta_tseries,linestyle='--',c='crimson',alpha=0.9,zorder=3)
                ax2.set_ylabel('Value of the virtual variable',font_y1)

                ax1.yaxis.set_major_locator(MaxNLocator(nbins=4)) 
                ax2.yaxis.set_major_locator(MaxNLocator(nbins=4))
                
            elif j == 3:
                ax = axs[i, j]
                ax.axis('off')

                legend_abundance = mlines.Line2D([], [], color='black', marker='none', linestyle='-', linewidth=2)
                legend_temperature = mlines.Line2D([], [], color='royalblue', marker='none', linestyle='-', linewidth=2)
                legend_vv = mlines.Line2D([], [], color='crimson', marker='none', linestyle='--', linewidth=2)
                ax.legend(handles=[legend_abundance,legend_temperature,legend_vv],labels=['Abundance','Water temperature','Virtual variable'],loc='center',frameon=False, handlelength=1, prop={'size':24})


    elif i == 3:
        for j in range(4):
            ax = axs[i, j]
            ax.plot(time, data_tseries[:,cnt],c='black',zorder=2)
            for count_y in range(6):
                ax.axvspan(time[14+48*count_y],time[14+48*count_y+23], color='silver', alpha=0.3, linewidth=0, zorder=1)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.set_title(name[cnt],fontdict=font_title)
            positions = [time[27], time[75], time[123], time[171], time[219], time[267]]
            years = ['2003', '2005', '2007', '2009', '2011', '2013']
            ax.set_xticks(positions)
            ax.set_xticklabels(years, fontsize=14, fontfamily='Arial') 
            ax.tick_params(axis='x', which='both', bottom=False)
            ax.tick_params(axis='y', labelsize=14) 
            cnt += 1        

    else:
        for j in range(4):
            ax = axs[i, j]
            ax.plot(time, data_tseries[:,cnt],c='black',zorder=2)
            for count_y in range(6):
                ax.axvspan(time[14+48*count_y],time[14+48*count_y+23], color='silver', alpha=0.3, linewidth=0, zorder=1)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.set_title(name[cnt],fontdict=font_title)
            ax.set_xticks([])
            ax.tick_params(axis='y', labelsize=14) 
            cnt += 1

fig.supxlabel('Year',x=0.535, y=0, fontproperties=font_x)
fig.supylabel('Abundance',x=0.005, y=0.55, fontproperties=font_y)

plt.subplots_adjust(top=0.96, bottom=0.06, left=0.055, right=0.99, hspace=0.25, wspace=0.16)
plt.savefig('../figures/SFIG3.pdf',format='pdf')
plt.savefig('/Users/zhugchzo/Desktop/3paper_fig/SFIG3.png',format='png',dpi=600)
