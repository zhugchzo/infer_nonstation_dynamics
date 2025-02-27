import pandas as pd
from sklearn.preprocessing import MinMaxScaler

df_tseries = pd.read_csv('fish_data.csv')

col = ['Aurelia.sp', 'Plotosus.japonicus', 'Sebastes.cheni', 'Trachurus.japonicus', 'Girella.punctata',
       'Pseudolabrus.sieboldi', 'Parajulis.poecilopterus', 'Halichoeres.tenuispinnis', 'Chaenogobius.gulosus',
       'Pterogobius.zonoleucus', 'Tridentiger.trigonocephalus', 'Siganus.fuscescens', 'Sphyraena.pinguis', 'Rudarius.ercodes']

data_tseries = df_tseries[col].values

scaler = MinMaxScaler()

data_scaled = scaler.fit_transform(data_tseries)

node_sums = data_scaled.sum(axis=1)

df_node_sums = pd.DataFrame(node_sums)

df_var = df_node_sums.rolling(window=10, center=False).var().dropna()

df_var.to_csv('../results/fish/fish_var.csv',header = False,index=False)
