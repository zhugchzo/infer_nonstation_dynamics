import numpy as np
import pandas as pd
import os
import scipy.stats as stats

if not os.path.exists('../results/NED'):
    os.makedirs('../results/NED')

def comp_NED(pred_x,traj_x):

    NED_results = np.zeros(len(pred_x))

    for i in range(len(pred_x)):

        pred_i = pred_x[:i+1,:]
        traj_i = traj_x[:i+1,:]

        error = pred_i - traj_i
        Eucli_d = np.sqrt(np.sum(error**2))
        normal_d = np.sqrt(np.sum(pred_i**2) + np.sum(traj_i**2))
        normal_Eucli_d = Eucli_d / normal_d

        NED_results[i] = normal_Eucli_d

    return NED_results

length = 699

NED_matrix = np.zeros((100,length))
NED_matrix_p = np.zeros((100,length))
NED_matrix_t = np.zeros((100,length))

row_count = 0

for rand_seed in range(100):

    df_pred = pd.read_csv('../results/Koscillators/robust/Koscillators_pred_{}.csv'.format(rand_seed))
    df_pred_p = pd.read_csv('../results/Koscillators/competing/Koscillators_pred_RCp_{}.csv'.format(rand_seed))
    df_pred_t = pd.read_csv('../results/Koscillators/competing/Koscillators_pred_RCt_{}.csv'.format(rand_seed))

    df_network = pd.read_csv('../Koscillators/Koscillators_data/Koscillators_network_{}.csv'.format(rand_seed),header=None)

    data_network = df_network.values

    # the number of node
    N = len(data_network)

    pred_col = []
    traj_col = []

    for node in range(N):
        pred_col.append('predx_{}'.format(node))
        traj_col.append('trajx_{}'.format(node))

    t = np.linspace(0, 0 + (length - 1) * 0.01, length)

    #####################################################################################

    predx = df_pred[pred_col].to_numpy()
    trajx = df_pred[traj_col].to_numpy()

    NED = comp_NED(predx,trajx)

    #####################################################################################

    predx_p = df_pred_p[pred_col].to_numpy()
    trajx_p = df_pred_p[traj_col].to_numpy()

    NED_p = comp_NED(predx_p,trajx_p)

    #####################################################################################

    predx_t = df_pred_t[pred_col].to_numpy()
    trajx_t = df_pred_t[traj_col].to_numpy()

    NED_t = comp_NED(predx_t,trajx_t)

    #####################################################################################

    NED_matrix[row_count,:] = NED
    NED_matrix_p[row_count,:] = NED_p
    NED_matrix_t[row_count,:] = NED_t

    row_count += 1

# NED_matrix_p = NED_matrix_p - NED_matrix
# NED_matrix_t = NED_matrix_t - NED_matrix
# NED_matrix_AIC = NED_matrix_AIC - NED_matrix
# NED_matrix = NED_matrix - NED_matrix

z = stats.norm.ppf(0.95)

means = np.mean(NED_matrix, axis=0)
std_errors = stats.sem(NED_matrix, axis=0)
ci_lower = means - z * std_errors
ci_upper = means + z * std_errors

means_p = np.mean(NED_matrix_p, axis=0)
std_errors_p = stats.sem(NED_matrix_p, axis=0)
ci_lower_p = means_p - z * std_errors_p
ci_upper_p = means_p + z * std_errors_p

means_t = np.mean(NED_matrix_t, axis=0)
std_errors_t = stats.sem(NED_matrix_t, axis=0)
ci_lower_t = means_t - z * std_errors_t
ci_upper_t = means_t + z * std_errors_t

dic_NED = {'Time':t, 'mean_NED':means, 'lower_NED':ci_lower, 'upper_NED':ci_upper,
             'mean_NED_p':means_p, 'lower_NED_p':ci_lower_p, 'upper_NED_p':ci_upper_p,
             'mean_NED_t':means_t, 'lower_NED_t':ci_lower_t, 'upper_NED_t':ci_upper_t}

NED_out = pd.DataFrame(dic_NED)
NED_out.to_csv('../results/NED/Koscillators_NED.csv',header = True, index=False)