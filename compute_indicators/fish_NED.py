import numpy as np
import pandas as pd
import os

if not os.path.exists('../results/NED/fish'):
    os.makedirs('../results/NED/fish')

def normalize(arr):
    normalized_arr = np.zeros_like(arr)
    
    for i in range(arr.shape[1]):

        col = arr[:, i]
        col_min = col.min()
        col_max = col.max()
        
        if col_max != col_min:
            normalized_arr[:, i] = (col - col_min) / (col_max - col_min)
        else:
            normalized_arr[:, i] = 0

    return normalized_arr

df_pred = pd.read_csv('../results/fish/fish_pred.csv')
df_pred_AIC = pd.read_csv('../results/fish/fish_pred_AIC.csv')
df_pred_p = pd.read_csv('../results/fish/fish_pred_p.csv')
df_pred_t = pd.read_csv('../results/fish/fish_pred_t.csv')

pred_col = ['Time', 'traj_Aurelia.sp', 'pred_Aurelia.sp', 'traj_Plotosus.japonicus', 'pred_Plotosus.japonicus',
             'traj_Sebastes.cheni', 'pred_Sebastes.cheni', 'traj_Trachurus.japonicus', 'pred_Trachurus.japonicus',
               'traj_Girella.punctata', 'pred_Girella.punctata', 'traj_Pseudolabrus.sieboldi', 'pred_Pseudolabrus.sieboldi',
                 'traj_Parajulis.poecilopterus', 'pred_Parajulis.poecilopterus', 'traj_Halichoeres.tenuispinnis',
                   'pred_Halichoeres.tenuispinnis', 'traj_Chaenogobius.gulosus', 'pred_Chaenogobius.gulosus',
                     'traj_Pterogobius.zonoleucus', 'pred_Pterogobius.zonoleucus', 'traj_Tridentiger.trigonocephalus',
                       'pred_Tridentiger.trigonocephalus', 'traj_Siganus.fuscescens', 'pred_Siganus.fuscescens',
                       'traj_Sphyraena.pinguis', 'pred_Sphyraena.pinguis', 'traj_Rudarius.ercodes', 'pred_Rudarius.ercodes']

N = int((len(pred_col) - 1)/2)

pred = df_pred[pred_col].values
pred_AIC = df_pred_AIC[pred_col].values
pred_p = df_pred_p[pred_col].values
pred_t = df_pred_t[pred_col].values

pred_normalized = normalize(pred[:,1:])
pred_AIC_normalized = normalize(pred_AIC[:,1:])
pred_p_normalized = normalize(pred_p[:,1:])
pred_t_normalized = normalize(pred_t[:,1:])

############################################

pred_normal_Eucli_d = list()
pred_AIC_normal_Eucli_d = list()
pred_p_normal_Eucli_d = list()
pred_t_normal_Eucli_d = list()

for i in range(len(pred)):

    pred_i = pred_normalized[:i+1,:]
    error_i = np.zeros((i+1,N))
    scale_i = np.zeros((i+1,N))

    for node in range(N):

        error = pred_i[:,2*node+1] - pred_i[:,2*node]
        error_i[:,node] = error**2
        scale_i[:,node] = pred_i[:,2*node+1]**2 + pred_i[:,2*node]**2

    Eucli_d = np.sqrt(np.sum(error_i))
    normal_d = np.sqrt(np.sum(scale_i))

    normal_Eucli_d = Eucli_d / normal_d

    pred_normal_Eucli_d.append(normal_Eucli_d)

for i in range(len(pred_AIC)):

    pred_i = pred_AIC_normalized[:i+1,:]
    error_i = np.zeros((i+1,N))
    scale_i = np.zeros((i+1,N))

    for node in range(N):

        error = pred_i[:,2*node+1] - pred_i[:,2*node]
        error_i[:,node] = error**2
        scale_i[:,node] = pred_i[:,2*node+1]**2 + pred_i[:,2*node]**2

    Eucli_d = np.sqrt(np.sum(error_i))
    normal_d = np.sqrt(np.sum(scale_i))

    normal_Eucli_d = Eucli_d / normal_d

    pred_AIC_normal_Eucli_d.append(normal_Eucli_d)

for i in range(len(pred_p)):

    pred_i = pred_p_normalized[:i+1,:]
    error_i = np.zeros((i+1,N))
    scale_i = np.zeros((i+1,N))

    for node in range(N):

        error = pred_i[:,2*node+1] - pred_i[:,2*node]
        error_i[:,node] = error**2
        scale_i[:,node] = pred_i[:,2*node+1]**2 + pred_i[:,2*node]**2

    Eucli_d = np.sqrt(np.sum(error_i))
    normal_d = np.sqrt(np.sum(scale_i))

    normal_Eucli_d = Eucli_d / normal_d

    pred_p_normal_Eucli_d.append(normal_Eucli_d)

for i in range(len(pred_t)):

    pred_i = pred_t_normalized[:i+1,:]
    error_i = np.zeros((i+1,N))
    scale_i = np.zeros((i+1,N))

    for node in range(N):

        error = pred_i[:,2*node+1] - pred_i[:,2*node]
        error_i[:,node] = error**2
        scale_i[:,node] = pred_i[:,2*node+1]**2 + pred_i[:,2*node]**2

    Eucli_d = np.sqrt(np.sum(error_i))
    normal_d = np.sqrt(np.sum(scale_i))

    normal_Eucli_d = Eucli_d / normal_d

    pred_t_normal_Eucli_d.append(normal_Eucli_d)

dic_pred = {'Time':pred[:,0], 'normal_Eucli_d':pred_normal_Eucli_d}
dic_pred_AIC = {'Time':pred_AIC[:,0], 'normal_Eucli_d':pred_AIC_normal_Eucli_d}
dic_pred_p = {'Time':pred_p[:,0], 'normal_Eucli_d':pred_p_normal_Eucli_d}
dic_pred_t = {'Time':pred_t[:,0], 'normal_Eucli_d':pred_t_normal_Eucli_d}

dic_pred = pd.DataFrame(dic_pred)
dic_pred.to_csv('../results/NED/fish/fish_pred.csv',header = True)

dic_pred_AIC = pd.DataFrame(dic_pred_AIC)
dic_pred_AIC.to_csv('../results/NED/fish/fish_pred_AIC.csv',header = True)

dic_pred_p = pd.DataFrame(dic_pred_p)
dic_pred_p.to_csv('../results/NED/fish/fish_pred_p.csv',header = True)

dic_pred_t = pd.DataFrame(dic_pred_t)
dic_pred_t.to_csv('../results/NED/fish/fish_pred_t.csv',header = True)

