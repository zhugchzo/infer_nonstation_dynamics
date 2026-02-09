import numpy as np
import pandas as pd
import os
import scipy.stats as stats

if not os.path.exists('../results/NED'):
    os.makedirs('../results/NED')

def comp_NED(pred_x,traj_x):

    NED_results = np.zeros(len(pred_x))

    for i in range(len(pred_x)):

        pred_i = pred_x[:i+1]
        traj_i = traj_x[:i+1]

        error = pred_i - traj_i
        Eucli_d = np.sqrt(np.sum(error**2))
        normal_d = np.sqrt(np.sum(pred_i**2) + np.sum(traj_i**2))
        normal_Eucli_d = Eucli_d / normal_d

        NED_results[i] = normal_Eucli_d

    return NED_results

length = 499

NED_matrix = np.zeros((100,length))
NED_matrix_ab = np.zeros((100,length))
NED_matrix_t = np.zeros((100,length))

row_count = 0

for al in [0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2]:
    for bl in [4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5]:

        df_pred = pd.read_csv('../results/cusp/robust/cusp_pred_{}_{}.csv'.format(al,bl))
        df_pred_RCab = pd.read_csv('../results/cusp/competing/cusp_pred_RCab_{}_{}.csv'.format(al,bl))
        df_pred_RCt = pd.read_csv('../results/cusp/competing/cusp_pred_RCt_{}_{}.csv'.format(al,bl))

        t = np.linspace(0, 0 + (length - 1) * 0.01, length)

        #####################################################################################

        predx = df_pred['pred'].to_numpy()
        trajx = df_pred['traj'].to_numpy()

        NED = comp_NED(predx,trajx)

        #####################################################################################

        predx_ab = df_pred_RCab['pred'].to_numpy()
        trajx_ab = df_pred_RCab['traj'].to_numpy()

        NED_ab = comp_NED(predx_ab,trajx_ab)

        #####################################################################################

        predx_t = df_pred_RCt['pred'].to_numpy()
        trajx_t = df_pred_RCt['traj'].to_numpy()

        NED_t = comp_NED(predx_t,trajx_t)

        #####################################################################################

        NED_matrix[row_count,:] = NED
        NED_matrix_ab[row_count,:] = NED_ab
        NED_matrix_t[row_count,:] = NED_t

        row_count += 1

for i in range(NED_matrix_t.shape[0]):
    row = NED_matrix_t[i, :]
    
    non_nan_index = np.where(~np.isnan(row))[0]
    
    if len(non_nan_index) > 0:
        start_value = row[non_nan_index[0]]
        
        for j in range(non_nan_index[0] + 1, len(row)):
            if np.isnan(row[j]):
                row[j] = start_value + (j - non_nan_index[0]) * (1 - start_value) / (len(row) - non_nan_index[0] - 1)

z = stats.norm.ppf(0.95)

means = np.mean(NED_matrix, axis=0)
std_errors = stats.sem(NED_matrix, axis=0)
ci_lower = means - z * std_errors
ci_upper = means + z * std_errors

means_ab = np.mean(NED_matrix_ab, axis=0)
std_errors_ab = stats.sem(NED_matrix_ab, axis=0)
ci_lower_ab = means_ab - z * std_errors_ab
ci_upper_ab = means_ab + z * std_errors_ab

means_t = np.mean(NED_matrix_t, axis=0)
std_errors_t = stats.sem(NED_matrix_t, axis=0)
ci_lower_t = means_t - z * std_errors_t
ci_upper_t = means_t + z * std_errors_t

dic_NED = {'Time':t, 'mean_NED':means, 'lower_NED':ci_lower, 'upper_NED':ci_upper,
             'mean_NED_ab':means_ab, 'lower_NED_ab':ci_lower_ab, 'upper_NED_ab':ci_upper_ab,
             'mean_NED_t':means_t, 'lower_NED_t':ci_lower_t, 'upper_NED_t':ci_upper_t}

NED_out = pd.DataFrame(dic_NED)
NED_out.to_csv('../results/NED/cusp_NED.csv',header = True, index=False)