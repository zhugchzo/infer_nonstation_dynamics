import numpy as np
import pandas as pd
# import os

# if not os.path.exists('../results/sMAPE/cusp_sin'):
#     os.makedirs('../results/sMAPE/cusp_sin')

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

albl_list = []
smape_list = []

for al in [0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2]:
    for bl in [4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5]:

       df_W_out = pd.read_csv('../results/cusp/cusp_sin/cusp_sin_W_out_{}_{}.csv'.format(al,bl))
       df_ab = pd.read_csv('../cusp/cusp_sin_data/cusp_sin_data_{}_{}.csv'.format(al,bl))

       one = np.ones(length)

       initial_theta = df_W_out['initial_theta'].to_numpy()[0]
       delta_theta = df_W_out['delta_theta'].to_numpy()[0]

       a = df_ab['a'].to_numpy()
       b = df_ab['b'].to_numpy()

       theta = np.linspace(initial_theta, initial_theta + (length - 1) * delta_theta, length)

       t = np.linspace(0, 0 + (length - 1) * 0.01, length)

       gt_coef_matrix = np.zeros((6,length))
       coef_matrix = np.zeros((6,length))

       gt_coef_matrix[0,:] = a
       gt_coef_matrix[1,:] = b
       gt_coef_matrix[3,:] = -1*one

       #####################################################################################

       W_out_cte = df_W_out[['cte','p','p^2','p^3','p^4','p^5']].to_numpy()[0]
       W_out_x = df_W_out[['x','px','p^2x','p^3x','p^4x']].to_numpy()[0]
       W_out_x2 = df_W_out[['x^2','px^2','p^2x^2','p^3x^2']].to_numpy()[0]
       W_out_x3 = df_W_out[['x^3','px^3','p^2x^3']].to_numpy()[0]
       W_out_x4 = df_W_out[['x^4','px^4']].to_numpy()[0]
       W_out_x5 = df_W_out[['x^5']].to_numpy()[0]

       coef_cte = W_out_cte[0]*one + W_out_cte[1]*theta + W_out_cte[2]*theta**2 + W_out_cte[3]*theta**3 + W_out_cte[4]*theta**4 + W_out_cte[5]*theta**5
       coef_x = W_out_x[0]*one + W_out_x[1]*theta + W_out_x[2]*theta**2 + W_out_x[3]*theta**3 + W_out_x[4]*theta**4
       coef_x2 = W_out_x2[0]*one + W_out_x2[1]*theta + W_out_x2[2]*theta**2 + W_out_x2[3]*theta**3
       coef_x3 = W_out_x3[0]*one + W_out_x3[1]*theta + W_out_x3[2]*theta**2
       coef_x4 = W_out_x4[0]*one + W_out_x4[1]*theta
       coef_x5 = W_out_x5[0]*one

       coef_matrix[0,:] = coef_cte
       coef_matrix[1,:] = coef_x
       coef_matrix[2,:] = coef_x2
       coef_matrix[3,:] = coef_x3
       coef_matrix[4,:] = coef_x4
       coef_matrix[5,:] = coef_x5

       delta = gt_coef_matrix - coef_matrix

       # smape = np.mean(sMAPE(coef_matrix,gt_coef_matrix))

       # albl_list.append('({},{})'.format(al,bl))
       # smape_list.append(smape)

# # Store series data in a temporary DataFrame
# dic_smape = {'initial value': albl_list, 'smape': smape_list}
# df_smape = pd.DataFrame(dic_smape)

# # coef= {'cte_true':a, 'cte_infer': coef_cte, 'x_true':b, 'x_infer':coef_x, 'x2_true':np.zeros(length), 'x2_infer':coef_x2,
# #        'x3_true':-1*np.ones(length), 'x3_infer':coef_x3, 'x4_true':np.zeros(length), 'x4_infer':coef_x4, 'x5_true':np.zeros(length), 'x5_infer':coef_x5}
# # df_coef = pd.DataFrame(coef)

# df_smape.to_csv('../results/sMAPE/cusp_sin_smape.csv',index=False)