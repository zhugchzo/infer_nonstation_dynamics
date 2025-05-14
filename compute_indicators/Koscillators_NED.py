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
NED_matrix_AIC = np.zeros((100,length))

row_count = 0

pred_col = ['predx_0','predx_1','predx_2','predx_3','predx_4','predx_5','predx_6','predx_7','predx_8','predx_9','predx_10']
traj_col = ['trajx_0','trajx_1','trajx_2','trajx_3','trajx_4','trajx_5','trajx_6','trajx_7','trajx_8','trajx_9','trajx_10']

for rand_seed in range(100):

    df_pred = pd.read_csv('../results/Koscillators/robust/Koscillators_pred_{}.csv'.format(rand_seed))
    df_pred_p = pd.read_csv('../results/Koscillators/robust/Koscillators_pred_p_{}.csv'.format(rand_seed))
    df_pred_t = pd.read_csv('../results/Koscillators/robust/Koscillators_pred_t_{}.csv'.format(rand_seed))
    df_pred_AIC = pd.read_csv('../results/Koscillators/robust/Koscillators_pred_AIC_{}.csv'.format(rand_seed))

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

    predx_AIC = df_pred_AIC[pred_col].to_numpy()
    trajx_AIC = df_pred_AIC[traj_col].to_numpy()

    NED_AIC = comp_NED(predx_AIC,trajx_AIC)

    #####################################################################################

    NED_matrix[row_count,:] = NED
    NED_matrix_p[row_count,:] = NED_p
    NED_matrix_t[row_count,:] = NED_t
    NED_matrix_AIC[row_count,:] = NED_AIC

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

means_AIC = np.mean(NED_matrix_AIC, axis=0)
std_errors_AIC = stats.sem(NED_matrix_AIC, axis=0)
ci_lower_AIC = means_AIC - z * std_errors_AIC
ci_upper_AIC = means_AIC + z * std_errors_AIC

means_p = means_p - means
means_t = means_t - means
means_AIC = means_AIC - means
means = means - means

ci_lower_p = ci_lower_p - ci_lower
ci_lower_t = ci_lower_t - ci_lower
ci_lower_AIC = ci_lower_AIC - ci_lower
ci_lower = ci_lower - ci_lower

ci_upper_p = ci_upper_p - ci_upper
ci_upper_t = ci_upper_t - ci_upper
ci_upper_AIC = ci_upper_AIC - ci_upper
ci_upper = ci_upper - ci_upper


dic_NED = {'Time':t, 'mean_NED':means, 'lower_NED':ci_lower, 'upper_NED':ci_upper,
             'mean_NED_p':means_p, 'lower_NED_p':ci_lower_p, 'upper_NED_p':ci_upper_p,
             'mean_NED_t':means_t, 'lower_NED_t':ci_lower_t, 'upper_NED_t':ci_upper_t,
             'mean_NED_AIC':means_AIC, 'lower_NED_AIC':ci_lower_AIC, 'upper_NED_AIC':ci_upper_AIC}

NED_out = pd.DataFrame(dic_NED)
NED_out.to_csv('../results/NED/Koscillators_NED.csv',header = True, index=False)

