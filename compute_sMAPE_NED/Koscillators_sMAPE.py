import numpy as np
import pandas as pd
import os
import scipy.stats as stats

if not os.path.exists('../results/sMAPE'):
    os.makedirs('../results/sMAPE')

def sMAPE(matrix1, matrix2, length=1000):

    N = len(matrix1)

    sMAPE = np.zeros(length)

    M = N

    for i in range(N):
        if np.sum(np.abs(matrix1[i,:])) + np.sum(np.abs(matrix2[i,:])) == 0:
            sMAPE += 0
            M -= 1
        else:
            sMAPE += np.abs(matrix1[i,:]-matrix2[i,:]) / (np.abs(matrix1[i,:])+np.abs(matrix2[i,:]))

    return sMAPE/M

length = 1000

smape_matrix = np.zeros((100,length))
smape_matrix_p = np.zeros((100,length))
smape_matrix_t = np.zeros((100,length))
smape_matrix_AIC = np.zeros((100,length))

row_count = 0

for rand_seed in range(100):

    df_W_out = pd.read_csv('../results/Koscillators/robust/Koscillators_W_out_{}.csv'.format(rand_seed))
    df_W_out_p = pd.read_csv('../results/Koscillators/robust/Koscillators_W_out_p_{}.csv'.format(rand_seed))
    df_W_out_t = pd.read_csv('../results/Koscillators/robust/Koscillators_W_out_t_{}.csv'.format(rand_seed))
    df_W_out_AIC = pd.read_csv('../results/Koscillators/robust/Koscillators_W_out_AIC_{}.csv'.format(rand_seed))

    one = np.ones(length)

    initial_theta = df_W_out['initial_theta'].to_numpy()[0]
    delta_theta = df_W_out['delta_theta'].to_numpy()[0]
    initial_theta_AIC = df_W_out_AIC['initial_theta'].to_numpy()[0]
    delta_theta_AIC = df_W_out_AIC['delta_theta'].to_numpy()[0]

    theta = np.linspace(initial_theta, initial_theta + (length - 1) * delta_theta, length)
    theta_AIC = np.linspace(initial_theta_AIC, initial_theta_AIC + (length - 1) * delta_theta_AIC, length)
    p = np.linspace(0, 0 + (length - 1) * 0.001, length)
    t = np.linspace(0, 0 + (length - 1) * 0.01, length)

    gt_coef_matrix = np.zeros((6,length))

    coef_matrix = np.zeros((6,length))
    coef_matrix_p = np.zeros((6,length))
    coef_matrix_t = np.zeros((6,length))
    coef_matrix_AIC = np.zeros((6,length))

    gt_coef_matrix[3,:] = p

    #####################################################################################

    W_out_cte = df_W_out[['cte','p','p^2']].to_numpy()[0]
    W_out_x = df_W_out[['x','px']].to_numpy()[0]
    W_out_x2 = df_W_out[['x^2']].to_numpy()[0]
    W_out_sinx = df_W_out[['sum_sin(xj-xi)','psum_sin(xj-xi)']].to_numpy()[0]
    W_out_sinx2 = df_W_out[['sum_sin^2(xj-xi)']].to_numpy()[0]
    W_out_xsinx = df_W_out[['xsum_sin(xj-xi)']].to_numpy()[0]

    coef_cte = W_out_cte[0]*one + W_out_cte[1]*theta + W_out_cte[2]*theta**2
    coef_x = W_out_x[0]*one + W_out_x[1]*theta
    coef_x2 = W_out_x2[0]*one
    coef_sinx = W_out_sinx[0]*one + W_out_sinx[1]*theta
    coef_sinx2 = W_out_sinx2[0]*one
    coef_xsinx = W_out_xsinx[0]*one

    coef_matrix[0,:] = coef_cte
    coef_matrix[1,:] = coef_x
    coef_matrix[2,:] = coef_x2
    coef_matrix[3,:] = coef_sinx
    coef_matrix[4,:] = coef_sinx2
    coef_matrix[5,:] = coef_xsinx

    smape = sMAPE(coef_matrix,gt_coef_matrix)

    #####################################################################################

    W_out_cte_p = df_W_out_p[['cte','p','p^2']].to_numpy()[0]
    W_out_x_p = df_W_out_p[['x','px']].to_numpy()[0]
    W_out_x2_p = df_W_out_p[['x^2']].to_numpy()[0]
    W_out_sinx_p = df_W_out_p[['sum_sin(xj-xi)','psum_sin(xj-xi)']].to_numpy()[0]
    W_out_sinx2_p = df_W_out_p[['sum_sin^2(xj-xi)']].to_numpy()[0]
    W_out_xsinx_p = df_W_out_p[['xsum_sin(xj-xi)']].to_numpy()[0]

    coef_cte_p = W_out_cte_p[0]*one + W_out_cte_p[1]*p + W_out_cte_p[2]*p**2
    coef_x_p = W_out_x_p[0]*one + W_out_x_p[1]*p
    coef_x2_p = W_out_x2_p[0]*one
    coef_sinx_p = W_out_sinx_p[0]*one + W_out_sinx_p[1]*p
    coef_sinx2_p = W_out_sinx2_p[0]*one
    coef_xsinx_p = W_out_xsinx_p[0]*one

    coef_matrix_p[0,:] = coef_cte_p
    coef_matrix_p[1,:] = coef_x_p
    coef_matrix_p[2,:] = coef_x2_p
    coef_matrix_p[3,:] = coef_sinx_p
    coef_matrix_p[4,:] = coef_sinx2_p
    coef_matrix_p[5,:] = coef_xsinx_p

    smape_p = sMAPE(coef_matrix_p,gt_coef_matrix)

    #####################################################################################

    W_out_cte_t = df_W_out_t[['cte','p','p^2']].to_numpy()[0]
    W_out_x_t = df_W_out_t[['x','px']].to_numpy()[0]
    W_out_x2_t = df_W_out_t[['x^2']].to_numpy()[0]
    W_out_sinx_t = df_W_out_t[['sum_sin(xj-xi)','psum_sin(xj-xi)']].to_numpy()[0]
    W_out_sinx2_t = df_W_out_t[['sum_sin^2(xj-xi)']].to_numpy()[0]
    W_out_xsinx_t = df_W_out_t[['xsum_sin(xj-xi)']].to_numpy()[0]

    coef_cte_t = W_out_cte_t[0]*one + W_out_cte_t[1]*t + W_out_cte_t[2]*t**2
    coef_x_t = W_out_x_t[0]*one + W_out_x_t[1]*t
    coef_x2_t = W_out_x2_t[0]*one
    coef_sinx_t = W_out_sinx_t[0]*one + W_out_sinx_t[1]*t
    coef_sinx2_t = W_out_sinx2_t[0]*one
    coef_xsinx_t = W_out_xsinx_t[0]*one

    coef_matrix_t[0,:] = coef_cte_t
    coef_matrix_t[1,:] = coef_x_t
    coef_matrix_t[2,:] = coef_x2_t
    coef_matrix_t[3,:] = coef_sinx_t
    coef_matrix_t[4,:] = coef_sinx2_t
    coef_matrix_t[5,:] = coef_xsinx_t

    smape_t = sMAPE(coef_matrix_t,gt_coef_matrix)

    #####################################################################################

    W_out_cte_AIC = df_W_out_AIC[['cte','p','p^2']].to_numpy()[0]
    W_out_x_AIC = df_W_out_AIC[['x','px']].to_numpy()[0]
    W_out_x2_AIC = df_W_out_AIC[['x^2']].to_numpy()[0]
    W_out_sinx_AIC = df_W_out_AIC[['sum_sin(xj-xi)','psum_sin(xj-xi)']].to_numpy()[0]
    W_out_sinx2_AIC = df_W_out_AIC[['sum_sin^2(xj-xi)']].to_numpy()[0]
    W_out_xsinx_AIC = df_W_out_AIC[['xsum_sin(xj-xi)']].to_numpy()[0]

    coef_cte_AIC = W_out_cte_AIC[0]*one + W_out_cte_AIC[1]*theta_AIC + W_out_cte_AIC[2]*theta_AIC**2
    coef_x_AIC = W_out_x_AIC[0]*one + W_out_x_AIC[1]*theta_AIC
    coef_x2_AIC = W_out_x2_AIC[0]*one
    coef_sinx_AIC = W_out_sinx_AIC[0]*one + W_out_sinx_AIC[1]*theta_AIC
    coef_sinx2_AIC = W_out_sinx2_AIC[0]*one
    coef_xsinx_AIC = W_out_xsinx_AIC[0]*one

    coef_matrix_AIC[0,:] = coef_cte_AIC
    coef_matrix_AIC[1,:] = coef_x_AIC
    coef_matrix_AIC[2,:] = coef_x2_AIC
    coef_matrix_AIC[3,:] = coef_sinx_AIC
    coef_matrix_AIC[4,:] = coef_sinx2_AIC
    coef_matrix_AIC[5,:] = coef_xsinx_AIC

    smape_AIC = sMAPE(coef_matrix_AIC,gt_coef_matrix)

    #####################################################################################

    smape_matrix[row_count,:] = smape
    smape_matrix_p[row_count,:] = smape_p
    smape_matrix_t[row_count,:] = smape_t
    smape_matrix_AIC[row_count,:] = smape_AIC

    row_count += 1    

z = stats.norm.ppf(0.95)

means = np.mean(smape_matrix, axis=0)
std_errors = stats.sem(smape_matrix, axis=0)
ci_lower = means - z * std_errors
ci_upper = means + z * std_errors

means_p = np.mean(smape_matrix_p, axis=0)
std_errors_p = stats.sem(smape_matrix_p, axis=0)
ci_lower_p = means_p - z * std_errors_p
ci_upper_p = means_p + z * std_errors_p

means_t = np.mean(smape_matrix_t, axis=0)
std_errors_t = stats.sem(smape_matrix_t, axis=0)
ci_lower_t = means_t - z * std_errors_t
ci_upper_t = means_t + z * std_errors_t

means_AIC = np.mean(smape_matrix_AIC, axis=0)
std_errors_AIC = stats.sem(smape_matrix_AIC, axis=0)
ci_lower_AIC = means_AIC - z * std_errors_AIC
ci_upper_AIC = means_AIC + z * std_errors_AIC

dic_smape = {'Time':t[100:], 'mean_smape':means[100:], 'lower_smape':ci_lower[100:], 'upper_smape':ci_upper[100:],
             'mean_smape_p':means_p[100:], 'lower_smape_p':ci_lower_p[100:], 'upper_smape_p':ci_upper_p[100:],
             'mean_smape_t':means_t[100:], 'lower_smape_t':ci_lower_t[100:], 'upper_smape_t':ci_upper_t[100:],
             'mean_smape_AIC':means_AIC[100:], 'lower_smape_AIC':ci_lower_AIC[100:], 'upper_smape_AIC':ci_upper_AIC[100:]}

smape_out = pd.DataFrame(dic_smape)
smape_out.to_csv('../results/sMAPE/Koscillators_smape.csv',header = True, index=False)

