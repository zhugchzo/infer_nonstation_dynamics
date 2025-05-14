import numpy as np
import pandas as pd
import os
import scipy.stats as stats

if not os.path.exists('../results/sMAPE'):
    os.makedirs('../results/sMAPE')

def sMAPE(matrix1, matrix2):

    N = len(matrix1)

    sMAPE = np.zeros(length)

    M = N

    for i in range(N):
        if np.sum(matrix1[i,:]) + np.sum(matrix2[i,:]) == 0:
            sMAPE += 0
            M -= 1
        else:
            sMAPE += abs(matrix1[i,:]-matrix2[i,:]) / (abs(matrix1[i,:])+abs(matrix2[i,:]))
    
    return sMAPE/M

length = 1000

smape_matrix = np.zeros((100,length))
smape_matrix_ab = np.zeros((100,length))
smape_matrix_t = np.zeros((100,length))
smape_matrix_AIC = np.zeros((100,length))

row_count = 0

for al in [0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2]:
    for bl in [4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5]:

        df_W_out = pd.read_csv('../results/cusp/robust/cusp_W_out_{}_{}.csv'.format(al,bl))
        df_W_out_ab = pd.read_csv('../results/cusp/robust/cusp_W_out_ab_{}_{}.csv'.format(al,bl))
        df_W_out_t = pd.read_csv('../results/cusp/robust/cusp_W_out_t_{}_{}.csv'.format(al,bl))
        df_W_out_AIC = pd.read_csv('../results/cusp/robust/cusp_W_out_AIC_{}_{}.csv'.format(al,bl))

        df_ab = pd.read_csv('../cusp/cusp_data/cusp_data_{}_{}.csv'.format(al,bl))

        one = np.ones(length)

        initial_theta = df_W_out['initial_theta'].to_numpy()[0]
        delta_theta = df_W_out['delta_theta'].to_numpy()[0]
        initial_theta_AIC = df_W_out_AIC['initial_theta'].to_numpy()[0]
        delta_theta_AIC = df_W_out_AIC['delta_theta'].to_numpy()[0]

        a = df_ab['a'].to_numpy()
        b = df_ab['b'].to_numpy()

        theta = np.linspace(initial_theta, initial_theta + (length - 1) * delta_theta, length)
        theta_AIC = np.linspace(initial_theta_AIC, initial_theta_AIC + (length - 1) * delta_theta_AIC, length)
        t = np.linspace(0, 0 + (length - 1) * 0.01, length)

        gt_coef_matrix = np.zeros((4,length))

        coef_matrix = np.zeros((4,length))
        coef_matrix_ab = np.zeros((4,length))
        coef_matrix_t = np.zeros((4,length))
        coef_matrix_AIC = np.zeros((4,length))

        gt_coef_matrix[0,:] = a
        gt_coef_matrix[1,:] = b
        gt_coef_matrix[3,:] = -1*one

        #####################################################################################

        W_out_cte = df_W_out[['cte','p','p^2','p^3']].to_numpy()[0]
        W_out_x = df_W_out[['x','px','p^2x']].to_numpy()[0]
        W_out_x2 = df_W_out[['x^2','px^2']].to_numpy()[0]
        W_out_x3 = df_W_out[['x^3']].to_numpy()[0]

        coef_cte = W_out_cte[0]*one + W_out_cte[1]*theta + W_out_cte[2]*theta**2 + W_out_cte[3]*theta**3
        coef_x = W_out_x[0]*one + W_out_x[1]*theta + W_out_x[2]*theta**2
        coef_x2 = W_out_x2[0]*one + W_out_x2[1]*theta
        coef_x3 = W_out_x3[0]*one

        coef_matrix[0,:] = coef_cte
        coef_matrix[1,:] = coef_x
        coef_matrix[2,:] = coef_x2
        coef_matrix[3,:] = coef_x3

        smape = sMAPE(coef_matrix,gt_coef_matrix)

        #####################################################################################

        W_out_cte_ab = df_W_out_ab[['cte','a','b','a^2','ab','b^2','a^3','a^2b','ab^2','b^3']].to_numpy()[0]
        W_out_x_ab = df_W_out_ab[['x','ax','bx','a^2x','abx','b^2x']].to_numpy()[0]
        W_out_x2_ab = df_W_out_ab[['x^2','ax^2','bx^2']].to_numpy()[0]
        W_out_x3_ab = df_W_out_ab[['x^3']].to_numpy()[0]

        coef_cte_ab = W_out_cte_ab[0]*one + W_out_cte_ab[1]*a + W_out_cte_ab[2]*b + W_out_cte_ab[3]*a**2 + W_out_cte_ab[4]*a*b + W_out_cte_ab[5]*b**2 + W_out_cte_ab[6]*a**3 + W_out_cte_ab[7]*a**2*b + W_out_cte_ab[8]*a*b**2 + W_out_cte_ab[9]*b**3
        coef_x_ab = W_out_x_ab[0]*one + W_out_x_ab[1]*a + W_out_x_ab[2]*b + W_out_x_ab[3]*a**2 + W_out_x_ab[4]*a*b + W_out_x_ab[5]*b**2
        coef_x2_ab = W_out_x2_ab[0]*one + W_out_x2_ab[1]*a + W_out_x2_ab[2]*b
        coef_x3_ab = W_out_x3_ab[0]*one

        coef_matrix_ab[0,:] = coef_cte_ab
        coef_matrix_ab[1,:] = coef_x_ab
        coef_matrix_ab[2,:] = coef_x2_ab
        coef_matrix_ab[3,:] = coef_x3_ab

        smape_ab = sMAPE(coef_matrix_ab,gt_coef_matrix)

        #####################################################################################

        W_out_cte_t = df_W_out_t[['cte','p','p^2','p^3']].to_numpy()[0]
        W_out_x_t = df_W_out_t[['x','px','p^2x']].to_numpy()[0]
        W_out_x2_t = df_W_out_t[['x^2','px^2']].to_numpy()[0]
        W_out_x3_t = df_W_out_t[['x^3']].to_numpy()[0]

        coef_cte_t = W_out_cte_t[0]*one + W_out_cte_t[1]*t + W_out_cte_t[2]*t**2 + W_out_cte_t[3]*t**3
        coef_x_t = W_out_x_t[0]*one + W_out_x_t[1]*t + W_out_x_t[2]*t**2
        coef_x2_t = W_out_x2_t[0]*one + W_out_x2_t[1]*t
        coef_x3_t = W_out_x3_t[0]*one

        coef_matrix_t[0,:] = coef_cte_t
        coef_matrix_t[1,:] = coef_x_t
        coef_matrix_t[2,:] = coef_x2_t
        coef_matrix_t[3,:] = coef_x3_t

        smape_t = sMAPE(coef_matrix_t,gt_coef_matrix)

        #####################################################################################

        W_out_cte_AIC = df_W_out_AIC[['cte','p','p^2','p^3']].to_numpy()[0]
        W_out_x_AIC = df_W_out_AIC[['x','px','p^2x']].to_numpy()[0]
        W_out_x2_AIC = df_W_out_AIC[['x^2','px^2']].to_numpy()[0]
        W_out_x3_AIC = df_W_out_AIC[['x^3']].to_numpy()[0]

        coef_cte_AIC = W_out_cte_AIC[0]*one + W_out_cte_AIC[1]*theta_AIC + W_out_cte_AIC[2]*theta_AIC **2 + W_out_cte_AIC[3]*theta_AIC **3
        coef_x_AIC = W_out_x_AIC[0]*one + W_out_x_AIC[1]*theta_AIC  + W_out_x_AIC[2]*theta_AIC **2
        coef_x2_AIC = W_out_x2_AIC[0]*one + W_out_x2_AIC[1]*theta_AIC 
        coef_x3_AIC = W_out_x3_AIC[0]*one

        coef_matrix_AIC[0,:] = coef_cte_AIC
        coef_matrix_AIC[1,:] = coef_x_AIC
        coef_matrix_AIC[2,:] = coef_x2_AIC
        coef_matrix_AIC[3,:] = coef_x3_AIC

        smape_AIC = sMAPE(coef_matrix_AIC,gt_coef_matrix)

        #####################################################################################

        smape_matrix[row_count,:] = smape
        smape_matrix_ab[row_count,:] = smape_ab
        smape_matrix_t[row_count,:] = smape_t
        smape_matrix_AIC[row_count,:] = smape_AIC

        row_count += 1    

z = stats.norm.ppf(0.95)

means = np.mean(smape_matrix, axis=0)
std_errors = stats.sem(smape_matrix, axis=0)
ci_lower = means - z * std_errors
ci_upper = means + z * std_errors

means_ab = np.mean(smape_matrix_ab, axis=0)
std_errors_ab = stats.sem(smape_matrix_ab, axis=0)
ci_lower_ab = means_ab - z * std_errors_ab
ci_upper_ab = means_ab + z * std_errors_ab

means_t = np.mean(smape_matrix_t, axis=0)
std_errors_t = stats.sem(smape_matrix_t, axis=0)
ci_lower_t = means_t - z * std_errors_t
ci_upper_t = means_t + z * std_errors_t

means_AIC = np.mean(smape_matrix_AIC, axis=0)
std_errors_AIC = stats.sem(smape_matrix_AIC, axis=0)
ci_lower_AIC = means_AIC - z * std_errors_AIC
ci_upper_AIC = means_AIC + z * std_errors_AIC

dic_smape = {'Time':t, 'mean_smape':means, 'lower_smape':ci_lower, 'upper_smape':ci_upper,
             'mean_smape_ab':means_ab, 'lower_smape_ab':ci_lower_ab, 'upper_smape_ab':ci_upper_ab,
             'mean_smape_t':means_t, 'lower_smape_t':ci_lower_t, 'upper_smape_t':ci_upper_t,
             'mean_smape_AIC':means_AIC, 'lower_smape_AIC':ci_lower_AIC, 'upper_smape_AIC':ci_upper_AIC}

smape_out = pd.DataFrame(dic_smape)
smape_out.to_csv('../results/sMAPE/cusp_smape.csv',header = True, index=False)

