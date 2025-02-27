import numpy as np
import pandas as pd
import os

if not os.path.exists('../results/NED/UAV'):
    os.makedirs('../results/NED/UAV')

def find_tipping(arr1, arr2):

    arr = np.sqrt(arr1**2 + arr2**2)  
    diffs = np.abs(np.diff(arr))
    indices = np.where(diffs > 5)[0]

    return indices[0] if len(indices) > 0 else len(arr)

df_pred = pd.read_csv('../results/UAV/UAV_pred.csv')
df_pred_AIC = pd.read_csv('../results/UAV/UAV_pred_AIC.csv')
df_pred_t = pd.read_csv('../results/UAV/UAV_pred_t.csv')

df_gen = pd.read_csv('../results/UAV/UAV_gen.csv')
df_gen_AIC = pd.read_csv('../results/UAV/UAV_gen_AIC.csv')
df_gen_t = pd.read_csv('../results/UAV/UAV_gen_t.csv')

pred_col = ['Time', 'traj_x', 'pred_x', 'traj_y', 'pred_y']
gen_col = ['Time', 'traj_x', 'gen_x', 'traj_y', 'gen_y']

pred = df_pred[pred_col].values
pred_AIC = df_pred_AIC[pred_col].values
pred_t = df_pred_t[pred_col].values

gen = df_gen[gen_col].values
gen_AIC = df_gen_AIC[gen_col].values
gen_t = df_gen_t[gen_col].values

############################################

pred_indices = find_tipping(pred[:, 2], pred[:, 4])
pred_AIC_indices = find_tipping(pred_AIC[:, 2], pred_AIC[:, 4])
pred_t_indices = find_tipping(pred_t[:, 2], pred_t[:, 4])

gen_indices = find_tipping(gen[:, 2], gen[:, 4])
gen_AIC_indices = find_tipping(gen_AIC[:, 2], gen_AIC[:, 4])
gen_t_indices = find_tipping(gen_t[:, 2], gen_t[:, 4])

############################################

pred = pred[:pred_indices]
pred_AIC = pred_AIC[:pred_AIC_indices]
pred_t = pred_t[:pred_t_indices]

gen = gen[:gen_indices]
gen_AIC = gen_AIC[:gen_AIC_indices]
gen_t = gen_t[:gen_t_indices]

############################################

pred_normal_Eucli_d = list()
pred_AIC_normal_Eucli_d = list()
pred_t_normal_Eucli_d = list()

for i in range(len(pred)):

    pred_i = pred[:i+1,1:5]

    error_x = pred_i[:,1] - pred_i[:,0]
    error_y = pred_i[:,3] - pred_i[:,2]

    Eucli_d = np.sqrt(np.sum(error_x**2 + error_y**2))
    normal_d = np.sqrt(np.sum(pred_i[:,0]**2) + np.sum(pred_i[:,1]**2) + np.sum(pred_i[:,2]**2) + np.sum(pred_i[:,3]**2))

    normal_Eucli_d = Eucli_d / normal_d

    pred_normal_Eucli_d.append(normal_Eucli_d)

for i in range(len(pred_AIC)):

    pred_i = pred_AIC[:i+1,1:5]

    error_x = pred_i[:,1] - pred_i[:,0]
    error_y = pred_i[:,3] - pred_i[:,2]

    Eucli_d = np.sqrt(np.sum(error_x**2 + error_y**2))
    normal_d = np.sqrt(np.sum(pred_i[:,0]**2) + np.sum(pred_i[:,1]**2) + np.sum(pred_i[:,2]**2) + np.sum(pred_i[:,3]**2))
    
    normal_Eucli_d = Eucli_d / normal_d

    pred_AIC_normal_Eucli_d.append(normal_Eucli_d)

for i in range(len(pred_t)):

    pred_i = pred_t[:i+1,1:5]
    
    error_x = pred_i[:,1] - pred_i[:,0]
    error_y = pred_i[:,3] - pred_i[:,2]

    Eucli_d = np.sqrt(np.sum(error_x**2 + error_y**2))
    normal_d = np.sqrt(np.sum(pred_i[:,0]**2) + np.sum(pred_i[:,1]**2) + np.sum(pred_i[:,2]**2) + np.sum(pred_i[:,3]**2))
    
    normal_Eucli_d = Eucli_d / normal_d

    pred_t_normal_Eucli_d.append(normal_Eucli_d)


dic_pred = {'Time':pred[:,0], 'normal_Eucli_d':pred_normal_Eucli_d}
dic_pred_AIC = {'Time':pred_AIC[:,0], 'normal_Eucli_d':pred_AIC_normal_Eucli_d}
dic_pred_t = {'Time':pred_t[:,0], 'normal_Eucli_d':pred_t_normal_Eucli_d}

dic_pred = pd.DataFrame(dic_pred)
dic_pred.to_csv('../results/NED/UAV/UAV_pred.csv',header = True)

dic_pred_AIC = pd.DataFrame(dic_pred_AIC)
dic_pred_AIC.to_csv('../results/NED/UAV/UAV_pred_AIC.csv',header = True)

dic_pred_t = pd.DataFrame(dic_pred_t)
dic_pred_t.to_csv('../results/NED/UAV/UAV_pred_t.csv',header = True)

############################################

gen_normal_Eucli_d = list()
gen_AIC_normal_Eucli_d = list()
gen_t_normal_Eucli_d = list()

for i in range(len(gen)):

    gen_i = gen[:i+1,1:5]

    error_x = gen_i[:,1] - gen_i[:,0]
    error_y = gen_i[:,3] - gen_i[:,2]

    Eucli_d = np.sqrt(np.sum(error_x**2 + error_y**2))
    normal_d = np.sqrt(np.sum(gen_i[:,0]**2) + np.sum(gen_i[:,1]**2) + np.sum(gen_i[:,2]**2) + np.sum(gen_i[:,3]**2))
    
    normal_Eucli_d = Eucli_d / normal_d

    gen_normal_Eucli_d.append(normal_Eucli_d)

for i in range(len(gen_AIC)):

    gen_i = gen_AIC[:i+1,1:5]

    error_x = gen_i[:,1] - gen_i[:,0]
    error_y = gen_i[:,3] - gen_i[:,2]

    Eucli_d = np.sqrt(np.sum(error_x**2 + error_y**2))
    normal_d = np.sqrt(np.sum(gen_i[:,0]**2) + np.sum(gen_i[:,1]**2) + np.sum(gen_i[:,2]**2) + np.sum(gen_i[:,3]**2))
    
    normal_Eucli_d = Eucli_d / normal_d

    gen_AIC_normal_Eucli_d.append(normal_Eucli_d)

for i in range(len(gen_t)):

    gen_i = gen_t[:i+1,1:5]
    
    error_x = gen_i[:,1] - gen_i[:,0]
    error_y = gen_i[:,3] - gen_i[:,2]

    Eucli_d = np.sqrt(np.sum(error_x**2 + error_y**2))
    normal_d = np.sqrt(np.sum(gen_i[:,0]**2) + np.sum(gen_i[:,1]**2) + np.sum(gen_i[:,2]**2) + np.sum(gen_i[:,3]**2))
    
    normal_Eucli_d = Eucli_d / normal_d

    gen_t_normal_Eucli_d.append(normal_Eucli_d)


dic_gen = {'Time':gen[:,0], 'normal_Eucli_d':gen_normal_Eucli_d}
dic_gen_AIC = {'Time':gen_AIC[:,0], 'normal_Eucli_d':gen_AIC_normal_Eucli_d}
dic_gen_t = {'Time':gen_t[:,0], 'normal_Eucli_d':gen_t_normal_Eucli_d}

dic_gen = pd.DataFrame(dic_gen)
dic_gen.to_csv('../results/NED/UAV/UAV_gen.csv',header = True)

dic_gen_AIC = pd.DataFrame(dic_gen_AIC)
dic_gen_AIC.to_csv('../results/NED/UAV/UAV_gen_AIC.csv',header = True)

dic_gen_t = pd.DataFrame(dic_gen_t)
dic_gen_t.to_csv('../results/NED/UAV/UAV_gen_t.csv',header = True)

