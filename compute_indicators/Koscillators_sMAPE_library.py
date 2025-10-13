import numpy as np
import pandas as pd
import os
import scipy.stats as stats

if not os.path.exists('../results/sMAPE/library'):
    os.makedirs('../results/sMAPE/library')

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

for tl in [300, 350, 400, 450, 500]:

    smape_matrix_2 = np.zeros((100,length))
    smape_matrix_3 = np.zeros((100,length))
    smape_matrix_4 = np.zeros((100,length))
    smape_matrix_5 = np.zeros((100,length))

    smape_matrix_p_2 = np.zeros((100,length))
    smape_matrix_p_3 = np.zeros((100,length))
    smape_matrix_p_4 = np.zeros((100,length))
    smape_matrix_p_5 = np.zeros((100,length))

    smape_matrix_t_2 = np.zeros((100,length))
    smape_matrix_t_3 = np.zeros((100,length))
    smape_matrix_t_4 = np.zeros((100,length))
    smape_matrix_t_5 = np.zeros((100,length))

    row_count = 0

    for rand_seed in range(100):

        df_W_out_2 = pd.read_csv('../results/Koscillators/library2/{}/Koscillators_W_out_{}.csv'.format(tl,rand_seed))
        df_W_out_3 = pd.read_csv('../results/Koscillators/library3/{}/Koscillators_W_out_{}.csv'.format(tl,rand_seed))
        df_W_out_4 = pd.read_csv('../results/Koscillators/library4/{}/Koscillators_W_out_{}.csv'.format(tl,rand_seed))
        df_W_out_5 = pd.read_csv('../results/Koscillators/library5/{}/Koscillators_W_out_{}.csv'.format(tl,rand_seed))

        df_W_out_p_2 = pd.read_csv('../results/Koscillators/library2/{}/Koscillators_W_out_p_{}.csv'.format(tl,rand_seed))
        df_W_out_p_3 = pd.read_csv('../results/Koscillators/library3/{}/Koscillators_W_out_p_{}.csv'.format(tl,rand_seed))
        df_W_out_p_4 = pd.read_csv('../results/Koscillators/library4/{}/Koscillators_W_out_p_{}.csv'.format(tl,rand_seed))
        df_W_out_p_5 = pd.read_csv('../results/Koscillators/library5/{}/Koscillators_W_out_p_{}.csv'.format(tl,rand_seed))

        df_W_out_t_2 = pd.read_csv('../results/Koscillators/library2/{}/Koscillators_W_out_t_{}.csv'.format(tl,rand_seed))
        df_W_out_t_3 = pd.read_csv('../results/Koscillators/library3/{}/Koscillators_W_out_t_{}.csv'.format(tl,rand_seed))
        df_W_out_t_4 = pd.read_csv('../results/Koscillators/library4/{}/Koscillators_W_out_t_{}.csv'.format(tl,rand_seed))
        df_W_out_t_5 = pd.read_csv('../results/Koscillators/library5/{}/Koscillators_W_out_t_{}.csv'.format(tl,rand_seed))

        one = np.ones(length)

        initial_theta_2 = df_W_out_2['initial_theta'].to_numpy()[0]
        delta_theta_2 = df_W_out_2['delta_theta'].to_numpy()[0]
        initial_theta_3 = df_W_out_3['initial_theta'].to_numpy()[0]
        delta_theta_3 = df_W_out_3['delta_theta'].to_numpy()[0]
        initial_theta_4 = df_W_out_4['initial_theta'].to_numpy()[0]
        delta_theta_4 = df_W_out_4['delta_theta'].to_numpy()[0]
        initial_theta_5 = df_W_out_5['initial_theta'].to_numpy()[0]
        delta_theta_5 = df_W_out_5['delta_theta'].to_numpy()[0]

        theta_2 = np.linspace(initial_theta_2, initial_theta_2 + (length - 1) * delta_theta_2, length)
        theta_3 = np.linspace(initial_theta_3, initial_theta_3 + (length - 1) * delta_theta_3, length)
        theta_4 = np.linspace(initial_theta_4, initial_theta_4 + (length - 1) * delta_theta_4, length)
        theta_5 = np.linspace(initial_theta_5, initial_theta_5 + (length - 1) * delta_theta_5, length)

        p = np.linspace(0, 0 + (length - 1) * 0.001, length)
        t = np.linspace(0, 0 + (length - 1) * 0.01, length)

        gt_coef_matrix = np.zeros((21,length))
        
        coef_matrix_2 = np.zeros((21,length))
        coef_matrix_3 = np.zeros((21,length))
        coef_matrix_4 = np.zeros((21,length))
        coef_matrix_5 = np.zeros((21,length))

        coef_matrix_p_2 = np.zeros((21,length))
        coef_matrix_p_3 = np.zeros((21,length))
        coef_matrix_p_4 = np.zeros((21,length))
        coef_matrix_p_5 = np.zeros((21,length))
        
        coef_matrix_t_2 = np.zeros((21,length))
        coef_matrix_t_3 = np.zeros((21,length))
        coef_matrix_t_4 = np.zeros((21,length))
        coef_matrix_t_5 = np.zeros((21,length))

        gt_coef_matrix[2,:] = p

        #####################################################################################

        W_out_cte_2 = df_W_out_2[['cte','p','p^2']].to_numpy()[0]
        W_out_x_2 = df_W_out_2[['x','px']].to_numpy()[0]
        W_out_sin_2 = df_W_out_2[['sin','psin']].to_numpy()[0]
        W_out_x2_2 = df_W_out_2[['x^2']].to_numpy()[0]
        W_out_sin2_2 = df_W_out_2[['sin^2']].to_numpy()[0]
        W_out_xsin_2 = df_W_out_2[['xsin']].to_numpy()[0]

        coef_cte_2 = W_out_cte_2[0]*one + W_out_cte_2[1]*theta_2 + W_out_cte_2[2]*theta_2**2
        coef_x_2 = W_out_x_2[0]*one + W_out_x_2[1]*theta_2
        coef_sin_2 = W_out_sin_2[0]*one + W_out_sin_2[1]*theta_2
        coef_x2_2 = W_out_x2_2[0]*one
        coef_sin2_2 = W_out_sin2_2[0]*one
        coef_xsin_2 = W_out_xsin_2[0]*one

        coef_matrix_2[0,:] = coef_cte_2
        coef_matrix_2[1,:] = coef_x_2
        coef_matrix_2[2,:] = coef_sin_2
        coef_matrix_2[3,:] = coef_x2_2
        coef_matrix_2[4,:] = coef_sin2_2
        coef_matrix_2[5,:] = coef_xsin_2

        smape_2 = sMAPE(coef_matrix_2,gt_coef_matrix)

        #####################################################################################

        W_out_cte_3 = df_W_out_3[['cte','p','p^2','p^3']].to_numpy()[0]
        W_out_x_3 = df_W_out_3[['x','px','p^2x']].to_numpy()[0]
        W_out_sin_3 = df_W_out_3[['sin','psin','p^2sin']].to_numpy()[0]
        W_out_x2_3 = df_W_out_3[['x^2','px^2']].to_numpy()[0]
        W_out_sin2_3 = df_W_out_3[['sin^2','psin^2']].to_numpy()[0]
        W_out_xsin_3 = df_W_out_3[['xsin','pxsin']].to_numpy()[0]
        W_out_x3_3 = df_W_out_3[['x^3']].to_numpy()[0]
        W_out_sin3_3 = df_W_out_3[['sin^3']].to_numpy()[0]
        W_out_x2sin_3 = df_W_out_3[['x^2sin']].to_numpy()[0]
        W_out_xsin2_3 = df_W_out_3[['xsin^2']].to_numpy()[0]

        coef_cte_3 = W_out_cte_3[0]*one + W_out_cte_3[1]*theta_3 + W_out_cte_3[2]*theta_3**2 + W_out_cte_3[3]*theta_3**3
        coef_x_3 = W_out_x_3[0]*one + W_out_x_3[1]*theta_3 + W_out_x_3[2]*theta_3**2
        coef_sin_3 = W_out_sin_3[0]*one + W_out_sin_3[1]*theta_3 + W_out_sin_3[2]*theta_3**2
        coef_x2_3 = W_out_x2_3[0]*one + W_out_x2_3[1]*theta_3
        coef_sin2_3 = W_out_sin2_3[0]*one + W_out_sin2_3[1]*theta_3
        coef_xsin_3 = W_out_xsin_3[0]*one + W_out_xsin_3[1]*theta_3
        coef_x3_3 = W_out_x3_3[0]*one
        coef_sin3_3 = W_out_sin3_3[0]*one     
        coef_x2sin_3 = W_out_x2sin_3[0]*one
        coef_xsin2_3 = W_out_xsin2_3[0]*one

        coef_matrix_3[0,:] = coef_cte_3
        coef_matrix_3[1,:] = coef_x_3
        coef_matrix_3[2,:] = coef_sin_3
        coef_matrix_3[3,:] = coef_x2_3
        coef_matrix_3[4,:] = coef_sin2_3
        coef_matrix_3[5,:] = coef_xsin_3
        coef_matrix_3[6,:] = coef_x3_3
        coef_matrix_3[7,:] = coef_sin3_3
        coef_matrix_3[8,:] = coef_x2sin_3
        coef_matrix_3[9,:] = coef_xsin2_3

        smape_3 = sMAPE(coef_matrix_3,gt_coef_matrix)

        #####################################################################################

        W_out_cte_4 = df_W_out_4[['cte','p','p^2','p^3','p^4']].to_numpy()[0]
        W_out_x_4 = df_W_out_4[['x','px','p^2x','p^3x']].to_numpy()[0]
        W_out_sin_4 = df_W_out_4[['sin','psin','p^2sin','p^3sin']].to_numpy()[0]
        W_out_x2_4 = df_W_out_4[['x^2','px^2','p^2x^2']].to_numpy()[0]
        W_out_sin2_4 = df_W_out_4[['sin^2','psin^2','p^2sin^2']].to_numpy()[0]
        W_out_xsin_4 = df_W_out_4[['xsin','pxsin','p^2xsin']].to_numpy()[0]
        W_out_x3_4 = df_W_out_4[['x^3','px^3']].to_numpy()[0]
        W_out_sin3_4 = df_W_out_4[['sin^3','psin^3']].to_numpy()[0]
        W_out_x2sin_4 = df_W_out_4[['x^2sin','px^2sin']].to_numpy()[0]
        W_out_xsin2_4 = df_W_out_4[['xsin^2','pxsin^2']].to_numpy()[0]
        W_out_x4_4 = df_W_out_4[['x^4']].to_numpy()[0]
        W_out_sin4_4 = df_W_out_4[['sin^4']].to_numpy()[0]
        W_out_x3sin_4 = df_W_out_4[['x^3sin']].to_numpy()[0]
        W_out_x2sin2_4 = df_W_out_4[['x^2sin^2']].to_numpy()[0]
        W_out_xsin3_4 = df_W_out_4[['xsin^3']].to_numpy()[0]

        coef_cte_4 = W_out_cte_4[0]*one + W_out_cte_4[1]*theta_4 + W_out_cte_4[2]*theta_4**2 + W_out_cte_4[3]*theta_4**3 + W_out_cte_4[4]*theta_4**4
        coef_x_4 = W_out_x_4[0]*one + W_out_x_4[1]*theta_4 + W_out_x_4[2]*theta_4**2 + W_out_x_4[3]*theta_4**3
        coef_sin_4 = W_out_sin_4[0]*one + W_out_sin_4[1]*theta_4 + W_out_sin_4[2]*theta_4**2 + W_out_sin_4[3]*theta_4**3
        coef_x2_4 = W_out_x2_4[0]*one + W_out_x2_4[1]*theta_4 + W_out_x2_4[2]*theta_4**2
        coef_sin2_4 = W_out_sin2_4[0]*one + W_out_sin2_4[1]*theta_4 + W_out_sin2_4[2]*theta_4**2
        coef_xsin_4 = W_out_xsin_4[0]*one + W_out_xsin_4[1]*theta_4 + W_out_xsin_4[2]*theta_4**2
        coef_x3_4 = W_out_x3_4[0]*one + W_out_x3_4[1]*theta_4
        coef_sin3_4 = W_out_sin3_4[0]*one + W_out_sin3_4[1]*theta_4
        coef_x2sin_4 = W_out_x2sin_4[0]*one + W_out_x2sin_4[1]*theta_4
        coef_xsin2_4 = W_out_xsin2_4[0]*one + W_out_xsin2_4[1]*theta_4
        coef_x4_4 = W_out_x4_4[0]*one
        coef_sin4_4 = W_out_sin4_4[0]*one
        coef_x3sin_4 = W_out_x3sin_4[0]*one
        coef_x2sin2_4 = W_out_x2sin2_4[0]*one
        coef_xsin3_4 = W_out_xsin3_4[0]*one

        coef_matrix_4[0,:] = coef_cte_4
        coef_matrix_4[1,:] = coef_x_4
        coef_matrix_4[2,:] = coef_sin_4
        coef_matrix_4[3,:] = coef_x2_4
        coef_matrix_4[4,:] = coef_sin2_4
        coef_matrix_4[5,:] = coef_xsin_4
        coef_matrix_4[6,:] = coef_x3_4
        coef_matrix_4[7,:] = coef_sin3_4
        coef_matrix_4[8,:] = coef_x2sin_4
        coef_matrix_4[9,:] = coef_xsin2_4
        coef_matrix_4[10,:] = coef_x4_4
        coef_matrix_4[11,:] = coef_sin4_4
        coef_matrix_4[12,:] = coef_x3sin_4
        coef_matrix_4[13,:] = coef_x2sin2_4
        coef_matrix_4[14,:] = coef_xsin3_4

        smape_4 = sMAPE(coef_matrix_4,gt_coef_matrix)

        #####################################################################################

        W_out_cte_5 = df_W_out_5[['cte','p','p^2','p^3','p^4','p^5']].to_numpy()[0]
        W_out_x_5 = df_W_out_5[['x','px','p^2x','p^3x','p^4x']].to_numpy()[0]
        W_out_sin_5 = df_W_out_5[['sin','psin','p^2sin','p^3sin','p^4sin']].to_numpy()[0]
        W_out_x2_5 = df_W_out_5[['x^2','px^2','p^2x^2','p^3x^2']].to_numpy()[0]
        W_out_sin2_5 = df_W_out_5[['sin^2','psin^2','p^2sin^2','p^3sin^2']].to_numpy()[0]
        W_out_xsin_5 = df_W_out_5[['xsin','pxsin','p^2xsin','p^3xsin']].to_numpy()[0]
        W_out_x3_5 = df_W_out_5[['x^3','px^3','p^2x^3']].to_numpy()[0]
        W_out_sin3_5 = df_W_out_5[['sin^3','psin^3','p^2sin^3']].to_numpy()[0]
        W_out_x2sin_5 = df_W_out_5[['x^2sin','px^2sin','p^2x^2sin']].to_numpy()[0]
        W_out_xsin2_5 = df_W_out_5[['xsin^2','pxsin^2','p^2xsin^2']].to_numpy()[0]
        W_out_x4_5 = df_W_out_5[['x^4','px^4']].to_numpy()[0]
        W_out_sin4_5 = df_W_out_5[['sin^4','psin^4']].to_numpy()[0]
        W_out_x3sin_5 = df_W_out_5[['x^3sin','px^3sin']].to_numpy()[0]
        W_out_x2sin2_5 = df_W_out_5[['x^2sin^2','px^2sin^2']].to_numpy()[0]
        W_out_xsin3_5 = df_W_out_5[['xsin^3','pxsin^3']].to_numpy()[0]
        W_out_x5_5 = df_W_out_5[['x^5']].to_numpy()[0]
        W_out_sin5_5 = df_W_out_5[['sin^5']].to_numpy()[0]
        W_out_x4sin_5 = df_W_out_5[['x^4sin']].to_numpy()[0]
        W_out_x3sin2_5 = df_W_out_5[['x^3sin^2']].to_numpy()[0]
        W_out_x2sin3_5 = df_W_out_5[['x^2sin^3']].to_numpy()[0]
        W_out_xsin4_5 = df_W_out_5[['xsin^4']].to_numpy()[0]

        coef_cte_5 = W_out_cte_5[0]*one + W_out_cte_5[1]*theta_5 + W_out_cte_5[2]*theta_5**2 + W_out_cte_5[3]*theta_5**3 + W_out_cte_5[4]*theta_5**4 + W_out_cte_5[5]*theta_5**5
        coef_x_5 = W_out_x_5[0]*one + W_out_x_5[1]*theta_5 + W_out_x_5[2]*theta_5**2 + W_out_x_5[3]*theta_5**3 + W_out_x_5[4]*theta_5**4
        coef_sin_5 = W_out_sin_5[0]*one + W_out_sin_5[1]*theta_5 + W_out_sin_5[2]*theta_5**2 + W_out_sin_5[3]*theta_5**3 + W_out_sin_5[4]*theta_5**4
        coef_x2_5 = W_out_x2_5[0]*one + W_out_x2_5[1]*theta_5 + W_out_x2_5[2]*theta_5**2 + W_out_x2_5[3]*theta_5**3
        coef_sin2_5 = W_out_sin2_5[0]*one + W_out_sin2_5[1]*theta_5 + W_out_sin2_5[2]*theta_5**2 + W_out_sin2_5[3]*theta_5**3
        coef_xsin_5 = W_out_xsin_5[0]*one + W_out_xsin_5[1]*theta_5 + W_out_xsin_5[2]*theta_5**2 + W_out_xsin_5[3]*theta_5**3
        coef_x3_5 = W_out_x3_5[0]*one + W_out_x3_5[1]*theta_5 + W_out_x3_5[2]*theta_5**2
        coef_sin3_5 = W_out_sin3_5[0]*one + W_out_sin3_5[1]*theta_5 + W_out_sin3_5[2]*theta_5**2
        coef_x2sin_5 = W_out_x2sin_5[0]*one + W_out_x2sin_5[1]*theta_5 + W_out_x2sin_5[2]*theta_5**2
        coef_xsin2_5 = W_out_xsin2_5[0]*one + W_out_xsin2_5[1]*theta_5 + W_out_xsin2_5[2]*theta_5**2
        coef_x4_5 = W_out_x4_5[0]*one + W_out_x4_5[1]*theta_5
        coef_sin4_5 = W_out_sin4_5[0]*one + W_out_sin4_5[1]*theta_5
        coef_x3sin_5 = W_out_x3sin_5[0]*one + W_out_x3sin_5[1]*theta_5
        coef_x2sin2_5 = W_out_x2sin2_5[0]*one + W_out_x2sin2_5[1]*theta_5
        coef_xsin3_5 = W_out_xsin3_5[0]*one + W_out_xsin3_5[1]*theta_5
        coef_x5_5 = W_out_x5_5[0]*one
        coef_sin5_5 = W_out_sin5_5[0]*one
        coef_x4sin_5 = W_out_x4sin_5[0]*one
        coef_x3sin2_5 = W_out_x3sin2_5[0]*one
        coef_x2sin3_5 = W_out_x2sin3_5[0]*one
        coef_xsin4_5 = W_out_xsin4_5[0]*one

        coef_matrix_5[0,:] = coef_cte_5
        coef_matrix_5[1,:] = coef_x_5
        coef_matrix_5[2,:] = coef_sin_5
        coef_matrix_5[3,:] = coef_x2_5
        coef_matrix_5[4,:] = coef_sin2_5
        coef_matrix_5[5,:] = coef_xsin_5
        coef_matrix_5[6,:] = coef_x3_5
        coef_matrix_5[7,:] = coef_sin3_5
        coef_matrix_5[8,:] = coef_x2sin_5
        coef_matrix_5[9,:] = coef_xsin2_5
        coef_matrix_5[10,:] = coef_x4_5
        coef_matrix_5[11,:] = coef_sin4_5
        coef_matrix_5[12,:] = coef_x3sin_5
        coef_matrix_5[13,:] = coef_x2sin2_5
        coef_matrix_5[14,:] = coef_xsin3_5
        coef_matrix_5[15,:] = coef_x5_5
        coef_matrix_5[16,:] = coef_sin5_5
        coef_matrix_5[17,:] = coef_x4sin_5
        coef_matrix_5[18,:] = coef_x3sin2_5
        coef_matrix_5[19,:] = coef_x2sin3_5
        coef_matrix_5[20,:] = coef_xsin4_5

        smape_5 = sMAPE(coef_matrix_5,gt_coef_matrix)

        #####################################################################################

        W_out_cte_p_2 = df_W_out_p_2[['cte','p','p^2']].to_numpy()[0]
        W_out_x_p_2 = df_W_out_p_2[['x','px']].to_numpy()[0]
        W_out_sin_p_2 = df_W_out_p_2[['sin','psin']].to_numpy()[0]
        W_out_x2_p_2 = df_W_out_p_2[['x^2']].to_numpy()[0]
        W_out_sin2_p_2 = df_W_out_p_2[['sin^2']].to_numpy()[0]
        W_out_xsin_p_2 = df_W_out_p_2[['xsin']].to_numpy()[0]

        coef_cte_p_2 = W_out_cte_p_2[0]*one + W_out_cte_p_2[1]*p + W_out_cte_p_2[2]*p**2
        coef_x_p_2 = W_out_x_p_2[0]*one + W_out_x_p_2[1]*p
        coef_sin_p_2 = W_out_sin_p_2[0]*one + W_out_sin_p_2[1]*p
        coef_x2_p_2 = W_out_x2_p_2[0]*one
        coef_sin2_p_2 = W_out_sin2_p_2[0]*one
        coef_xsin_p_2 = W_out_xsin_p_2[0]*one

        coef_matrix_p_2[0,:] = coef_cte_p_2
        coef_matrix_p_2[1,:] = coef_x_p_2
        coef_matrix_p_2[2,:] = coef_sin_p_2
        coef_matrix_p_2[3,:] = coef_x2_p_2
        coef_matrix_p_2[4,:] = coef_sin2_p_2
        coef_matrix_p_2[5,:] = coef_xsin_p_2

        smape_p_2 = sMAPE(coef_matrix_p_2,gt_coef_matrix)

        #####################################################################################

        W_out_cte_p_3 = df_W_out_p_3[['cte','p','p^2','p^3']].to_numpy()[0]
        W_out_x_p_3 = df_W_out_p_3[['x','px','p^2x']].to_numpy()[0]
        W_out_sin_p_3 = df_W_out_p_3[['sin','psin','p^2sin']].to_numpy()[0]
        W_out_x2_p_3 = df_W_out_p_3[['x^2','px^2']].to_numpy()[0]
        W_out_sin2_p_3 = df_W_out_p_3[['sin^2','psin^2']].to_numpy()[0]
        W_out_xsin_p_3 = df_W_out_p_3[['xsin','pxsin']].to_numpy()[0]
        W_out_x3_p_3 = df_W_out_p_3[['x^3']].to_numpy()[0]
        W_out_sin3_p_3 = df_W_out_p_3[['sin^3']].to_numpy()[0]
        W_out_x2sin_p_3 = df_W_out_p_3[['x^2sin']].to_numpy()[0]
        W_out_xsin2_p_3 = df_W_out_p_3[['xsin^2']].to_numpy()[0]

        coef_cte_p_3 = W_out_cte_p_3[0]*one + W_out_cte_p_3[1]*p + W_out_cte_p_3[2]*p**2 + W_out_cte_p_3[3]*p**3
        coef_x_p_3 = W_out_x_p_3[0]*one + W_out_x_p_3[1]*p + W_out_x_p_3[2]*p**2
        coef_sin_p_3 = W_out_sin_p_3[0]*one + W_out_sin_p_3[1]*p + W_out_sin_p_3[2]*p**2
        coef_x2_p_3 = W_out_x2_p_3[0]*one + W_out_x2_p_3[1]*p
        coef_sin2_p_3 = W_out_sin2_p_3[0]*one + W_out_sin2_p_3[1]*p
        coef_xsin_p_3 = W_out_xsin_p_3[0]*one + W_out_xsin_p_3[1]*p
        coef_x3_p_3 = W_out_x3_p_3[0]*one
        coef_sin3_p_3 = W_out_sin3_p_3[0]*one     
        coef_x2sin_p_3 = W_out_x2sin_p_3[0]*one
        coef_xsin2_p_3 = W_out_xsin2_p_3[0]*one

        coef_matrix_p_3[0,:] = coef_cte_p_3
        coef_matrix_p_3[1,:] = coef_x_p_3
        coef_matrix_p_3[2,:] = coef_sin_p_3
        coef_matrix_p_3[3,:] = coef_x2_p_3
        coef_matrix_p_3[4,:] = coef_sin2_p_3
        coef_matrix_p_3[5,:] = coef_xsin_p_3
        coef_matrix_p_3[6,:] = coef_x3_p_3
        coef_matrix_p_3[7,:] = coef_sin3_p_3
        coef_matrix_p_3[8,:] = coef_x2sin_p_3
        coef_matrix_p_3[9,:] = coef_xsin2_p_3

        smape_p_3 = sMAPE(coef_matrix_p_3,gt_coef_matrix)

        #####################################################################################

        W_out_cte_p_4 = df_W_out_p_4[['cte','p','p^2','p^3','p^4']].to_numpy()[0]
        W_out_x_p_4 = df_W_out_p_4[['x','px','p^2x','p^3x']].to_numpy()[0]
        W_out_sin_p_4 = df_W_out_p_4[['sin','psin','p^2sin','p^3sin']].to_numpy()[0]
        W_out_x2_p_4 = df_W_out_p_4[['x^2','px^2','p^2x^2']].to_numpy()[0]
        W_out_sin2_p_4 = df_W_out_p_4[['sin^2','psin^2','p^2sin^2']].to_numpy()[0]
        W_out_xsin_p_4 = df_W_out_p_4[['xsin','pxsin','p^2xsin']].to_numpy()[0]
        W_out_x3_p_4 = df_W_out_p_4[['x^3','px^3']].to_numpy()[0]
        W_out_sin3_p_4 = df_W_out_p_4[['sin^3','psin^3']].to_numpy()[0]
        W_out_x2sin_p_4 = df_W_out_p_4[['x^2sin','px^2sin']].to_numpy()[0]
        W_out_xsin2_p_4 = df_W_out_p_4[['xsin^2','pxsin^2']].to_numpy()[0]
        W_out_x4_p_4 = df_W_out_p_4[['x^4']].to_numpy()[0]
        W_out_sin4_p_4 = df_W_out_p_4[['sin^4']].to_numpy()[0]
        W_out_x3sin_p_4 = df_W_out_p_4[['x^3sin']].to_numpy()[0]
        W_out_x2sin2_p_4 = df_W_out_p_4[['x^2sin^2']].to_numpy()[0]
        W_out_xsin3_p_4 = df_W_out_p_4[['xsin^3']].to_numpy()[0]

        coef_cte_p_4 = W_out_cte_p_4[0]*one + W_out_cte_p_4[1]*p + W_out_cte_p_4[2]*p**2 + W_out_cte_p_4[3]*p**3 + W_out_cte_p_4[4]*p**4
        coef_x_p_4 = W_out_x_p_4[0]*one + W_out_x_p_4[1]*p + W_out_x_p_4[2]*p**2 + W_out_x_p_4[3]*p**3
        coef_sin_p_4 = W_out_sin_p_4[0]*one + W_out_sin_p_4[1]*p + W_out_sin_p_4[2]*p**2 + W_out_sin_p_4[3]*p**3
        coef_x2_p_4 = W_out_x2_p_4[0]*one + W_out_x2_p_4[1]*p + W_out_x2_p_4[2]*p**2
        coef_sin2_p_4 = W_out_sin2_p_4[0]*one + W_out_sin2_p_4[1]*p + W_out_sin2_p_4[2]*p**2
        coef_xsin_p_4 = W_out_xsin_p_4[0]*one + W_out_xsin_p_4[1]*p + W_out_xsin_p_4[2]*p**2
        coef_x3_p_4 = W_out_x3_p_4[0]*one + W_out_x3_p_4[1]*p
        coef_sin3_p_4 = W_out_sin3_p_4[0]*one + W_out_sin3_p_4[1]*p
        coef_x2sin_p_4 = W_out_x2sin_p_4[0]*one + W_out_x2sin_p_4[1]*p
        coef_xsin2_p_4 = W_out_xsin2_p_4[0]*one + W_out_xsin2_p_4[1]*p
        coef_x4_p_4 = W_out_x4_p_4[0]*one
        coef_sin4_p_4 = W_out_sin4_p_4[0]*one
        coef_x3sin_p_4 = W_out_x3sin_p_4[0]*one
        coef_x2sin2_p_4 = W_out_x2sin2_p_4[0]*one
        coef_xsin3_p_4 = W_out_xsin3_p_4[0]*one

        coef_matrix_p_4[0,:] = coef_cte_p_4
        coef_matrix_p_4[1,:] = coef_x_p_4
        coef_matrix_p_4[2,:] = coef_sin_p_4
        coef_matrix_p_4[3,:] = coef_x2_p_4
        coef_matrix_p_4[4,:] = coef_sin2_p_4
        coef_matrix_p_4[5,:] = coef_xsin_p_4
        coef_matrix_p_4[6,:] = coef_x3_p_4
        coef_matrix_p_4[7,:] = coef_sin3_p_4
        coef_matrix_p_4[8,:] = coef_x2sin_p_4
        coef_matrix_p_4[9,:] = coef_xsin2_p_4
        coef_matrix_p_4[10,:] = coef_x4_p_4
        coef_matrix_p_4[11,:] = coef_sin4_p_4
        coef_matrix_p_4[12,:] = coef_x3sin_p_4
        coef_matrix_p_4[13,:] = coef_x2sin2_p_4
        coef_matrix_p_4[14,:] = coef_xsin3_p_4

        smape_p_4 = sMAPE(coef_matrix_p_4,gt_coef_matrix)

        #####################################################################################

        W_out_cte_p_5 = df_W_out_p_5[['cte','p','p^2','p^3','p^4','p^5']].to_numpy()[0]
        W_out_x_p_5 = df_W_out_p_5[['x','px','p^2x','p^3x','p^4x']].to_numpy()[0]
        W_out_sin_p_5 = df_W_out_p_5[['sin','psin','p^2sin','p^3sin','p^4sin']].to_numpy()[0]
        W_out_x2_p_5 = df_W_out_p_5[['x^2','px^2','p^2x^2','p^3x^2']].to_numpy()[0]
        W_out_sin2_p_5 = df_W_out_p_5[['sin^2','psin^2','p^2sin^2','p^3sin^2']].to_numpy()[0]
        W_out_xsin_p_5 = df_W_out_p_5[['xsin','pxsin','p^2xsin','p^3xsin']].to_numpy()[0]
        W_out_x3_p_5 = df_W_out_p_5[['x^3','px^3','p^2x^3']].to_numpy()[0]
        W_out_sin3_p_5 = df_W_out_p_5[['sin^3','psin^3','p^2sin^3']].to_numpy()[0]
        W_out_x2sin_p_5 = df_W_out_p_5[['x^2sin','px^2sin','p^2x^2sin']].to_numpy()[0]
        W_out_xsin2_p_5 = df_W_out_p_5[['xsin^2','pxsin^2','p^2xsin^2']].to_numpy()[0]
        W_out_x4_p_5 = df_W_out_p_5[['x^4','px^4']].to_numpy()[0]
        W_out_sin4_p_5 = df_W_out_p_5[['sin^4','psin^4']].to_numpy()[0]
        W_out_x3sin_p_5 = df_W_out_p_5[['x^3sin','px^3sin']].to_numpy()[0]
        W_out_x2sin2_p_5 = df_W_out_p_5[['x^2sin^2','px^2sin^2']].to_numpy()[0]
        W_out_xsin3_p_5 = df_W_out_p_5[['xsin^3','pxsin^3']].to_numpy()[0]
        W_out_x5_p_5 = df_W_out_p_5[['x^5']].to_numpy()[0]
        W_out_sin5_p_5 = df_W_out_p_5[['sin^5']].to_numpy()[0]
        W_out_x4sin_p_5 = df_W_out_p_5[['x^4sin']].to_numpy()[0]
        W_out_x3sin2_p_5 = df_W_out_p_5[['x^3sin^2']].to_numpy()[0]
        W_out_x2sin3_p_5 = df_W_out_p_5[['x^2sin^3']].to_numpy()[0]
        W_out_xsin4_p_5 = df_W_out_p_5[['xsin^4']].to_numpy()[0]

        coef_cte_p_5 = W_out_cte_p_5[0]*one + W_out_cte_p_5[1]*p + W_out_cte_p_5[2]*p**2 + W_out_cte_p_5[3]*p**3 + W_out_cte_p_5[4]*p**4 + W_out_cte_p_5[5]*p**5
        coef_x_p_5 = W_out_x_p_5[0]*one + W_out_x_p_5[1]*p + W_out_x_p_5[2]*p**2 + W_out_x_p_5[3]*p**3 + W_out_x_p_5[4]*p**4
        coef_sin_p_5 = W_out_sin_p_5[0]*one + W_out_sin_p_5[1]*p + W_out_sin_p_5[2]*p**2 + W_out_sin_p_5[3]*p**3 + W_out_sin_p_5[4]*p**4
        coef_x2_p_5 = W_out_x2_p_5[0]*one + W_out_x2_p_5[1]*p + W_out_x2_p_5[2]*p**2 + W_out_x2_p_5[3]*p**3
        coef_sin2_p_5 = W_out_sin2_p_5[0]*one + W_out_sin2_p_5[1]*p + W_out_sin2_p_5[2]*p**2 + W_out_sin2_p_5[3]*p**3
        coef_xsin_p_5 = W_out_xsin_p_5[0]*one + W_out_xsin_p_5[1]*p + W_out_xsin_p_5[2]*p**2 + W_out_xsin_p_5[3]*p**3
        coef_x3_p_5 = W_out_x3_p_5[0]*one + W_out_x3_p_5[1]*p + W_out_x3_p_5[2]*p**2
        coef_sin3_p_5 = W_out_sin3_p_5[0]*one + W_out_sin3_p_5[1]*p + W_out_sin3_p_5[2]*p**2
        coef_x2sin_p_5 = W_out_x2sin_p_5[0]*one + W_out_x2sin_p_5[1]*p + W_out_x2sin_p_5[2]*p**2
        coef_xsin2_p_5 = W_out_xsin2_p_5[0]*one + W_out_xsin2_p_5[1]*p + W_out_xsin2_p_5[2]*p**2
        coef_x4_p_5 = W_out_x4_p_5[0]*one + W_out_x4_p_5[1]*p
        coef_sin4_p_5 = W_out_sin4_p_5[0]*one + W_out_sin4_p_5[1]*p
        coef_x3sin_p_5 = W_out_x3sin_p_5[0]*one + W_out_x3sin_p_5[1]*p
        coef_x2sin2_p_5 = W_out_x2sin2_p_5[0]*one + W_out_x2sin2_p_5[1]*p
        coef_xsin3_p_5 = W_out_xsin3_p_5[0]*one + W_out_xsin3_p_5[1]*p
        coef_x5_p_5 = W_out_x5_p_5[0]*one
        coef_sin5_p_5 = W_out_sin5_p_5[0]*one
        coef_x4sin_p_5 = W_out_x4sin_p_5[0]*one
        coef_x3sin2_p_5 = W_out_x3sin2_p_5[0]*one
        coef_x2sin3_p_5 = W_out_x2sin3_p_5[0]*one
        coef_xsin4_p_5 = W_out_xsin4_p_5[0]*one

        coef_matrix_p_5[0,:] = coef_cte_p_5
        coef_matrix_p_5[1,:] = coef_x_p_5
        coef_matrix_p_5[2,:] = coef_sin_p_5
        coef_matrix_p_5[3,:] = coef_x2_p_5
        coef_matrix_p_5[4,:] = coef_sin2_p_5
        coef_matrix_p_5[5,:] = coef_xsin_p_5
        coef_matrix_p_5[6,:] = coef_x3_p_5
        coef_matrix_p_5[7,:] = coef_sin3_p_5
        coef_matrix_p_5[8,:] = coef_x2sin_p_5
        coef_matrix_p_5[9,:] = coef_xsin2_p_5
        coef_matrix_p_5[10,:] = coef_x4_p_5
        coef_matrix_p_5[11,:] = coef_sin4_p_5
        coef_matrix_p_5[12,:] = coef_x3sin_p_5
        coef_matrix_p_5[13,:] = coef_x2sin2_p_5
        coef_matrix_p_5[14,:] = coef_xsin3_p_5
        coef_matrix_p_5[15,:] = coef_x5_p_5
        coef_matrix_p_5[16,:] = coef_sin5_p_5
        coef_matrix_p_5[17,:] = coef_x4sin_p_5
        coef_matrix_p_5[18,:] = coef_x3sin2_p_5
        coef_matrix_p_5[19,:] = coef_x2sin3_p_5
        coef_matrix_p_5[20,:] = coef_xsin4_p_5

        smape_p_5 = sMAPE(coef_matrix_p_5,gt_coef_matrix)

        #####################################################################################

        W_out_cte_t_2 = df_W_out_t_2[['cte','p','p^2']].to_numpy()[0]
        W_out_x_t_2 = df_W_out_t_2[['x','px']].to_numpy()[0]
        W_out_sin_t_2 = df_W_out_t_2[['sin','psin']].to_numpy()[0]
        W_out_x2_t_2 = df_W_out_t_2[['x^2']].to_numpy()[0]
        W_out_sin2_t_2 = df_W_out_t_2[['sin^2']].to_numpy()[0]
        W_out_xsin_t_2 = df_W_out_t_2[['xsin']].to_numpy()[0]

        coef_cte_t_2 = W_out_cte_t_2[0]*one + W_out_cte_t_2[1]*t + W_out_cte_t_2[2]*t**2
        coef_x_t_2 = W_out_x_t_2[0]*one + W_out_x_t_2[1]*t
        coef_sin_t_2 = W_out_sin_t_2[0]*one + W_out_sin_t_2[1]*t
        coef_x2_t_2 = W_out_x2_t_2[0]*one
        coef_sin2_t_2 = W_out_sin2_t_2[0]*one
        coef_xsin_t_2 = W_out_xsin_t_2[0]*one

        coef_matrix_t_2[0,:] = coef_cte_t_2
        coef_matrix_t_2[1,:] = coef_x_t_2
        coef_matrix_t_2[2,:] = coef_sin_t_2
        coef_matrix_t_2[3,:] = coef_x2_t_2
        coef_matrix_t_2[4,:] = coef_sin2_t_2
        coef_matrix_t_2[5,:] = coef_xsin_t_2

        smape_t_2 = sMAPE(coef_matrix_t_2,gt_coef_matrix)

        #####################################################################################

        W_out_cte_t_3 = df_W_out_t_3[['cte','p','p^2','p^3']].to_numpy()[0]
        W_out_x_t_3 = df_W_out_t_3[['x','px','p^2x']].to_numpy()[0]
        W_out_sin_t_3 = df_W_out_t_3[['sin','psin','p^2sin']].to_numpy()[0]
        W_out_x2_t_3 = df_W_out_t_3[['x^2','px^2']].to_numpy()[0]
        W_out_sin2_t_3 = df_W_out_t_3[['sin^2','psin^2']].to_numpy()[0]
        W_out_xsin_t_3 = df_W_out_t_3[['xsin','pxsin']].to_numpy()[0]
        W_out_x3_t_3 = df_W_out_t_3[['x^3']].to_numpy()[0]
        W_out_sin3_t_3 = df_W_out_t_3[['sin^3']].to_numpy()[0]
        W_out_x2sin_t_3 = df_W_out_t_3[['x^2sin']].to_numpy()[0]
        W_out_xsin2_t_3 = df_W_out_t_3[['xsin^2']].to_numpy()[0]

        coef_cte_t_3 = W_out_cte_t_3[0]*one + W_out_cte_t_3[1]*t + W_out_cte_t_3[2]*t**2 + W_out_cte_t_3[3]*t**3
        coef_x_t_3 = W_out_x_t_3[0]*one + W_out_x_t_3[1]*t + W_out_x_t_3[2]*t**2
        coef_sin_t_3 = W_out_sin_t_3[0]*one + W_out_sin_t_3[1]*t + W_out_sin_t_3[2]*t**2
        coef_x2_t_3 = W_out_x2_t_3[0]*one + W_out_x2_t_3[1]*t
        coef_sin2_t_3 = W_out_sin2_t_3[0]*one + W_out_sin2_t_3[1]*t
        coef_xsin_t_3 = W_out_xsin_t_3[0]*one + W_out_xsin_t_3[1]*t
        coef_x3_t_3 = W_out_x3_t_3[0]*one
        coef_sin3_t_3 = W_out_sin3_t_3[0]*one     
        coef_x2sin_t_3 = W_out_x2sin_t_3[0]*one
        coef_xsin2_t_3 = W_out_xsin2_t_3[0]*one

        coef_matrix_t_3[0,:] = coef_cte_t_3
        coef_matrix_t_3[1,:] = coef_x_t_3
        coef_matrix_t_3[2,:] = coef_sin_t_3
        coef_matrix_t_3[3,:] = coef_x2_t_3
        coef_matrix_t_3[4,:] = coef_sin2_t_3
        coef_matrix_t_3[5,:] = coef_xsin_t_3
        coef_matrix_t_3[6,:] = coef_x3_t_3
        coef_matrix_t_3[7,:] = coef_sin3_t_3
        coef_matrix_t_3[8,:] = coef_x2sin_t_3
        coef_matrix_t_3[9,:] = coef_xsin2_t_3

        smape_t_3 = sMAPE(coef_matrix_t_3,gt_coef_matrix)

        #####################################################################################

        W_out_cte_t_4 = df_W_out_t_4[['cte','p','p^2','p^3','p^4']].to_numpy()[0]
        W_out_x_t_4 = df_W_out_t_4[['x','px','p^2x','p^3x']].to_numpy()[0]
        W_out_sin_t_4 = df_W_out_t_4[['sin','psin','p^2sin','p^3sin']].to_numpy()[0]
        W_out_x2_t_4 = df_W_out_t_4[['x^2','px^2','p^2x^2']].to_numpy()[0]
        W_out_sin2_t_4 = df_W_out_t_4[['sin^2','psin^2','p^2sin^2']].to_numpy()[0]
        W_out_xsin_t_4 = df_W_out_t_4[['xsin','pxsin','p^2xsin']].to_numpy()[0]
        W_out_x3_t_4 = df_W_out_t_4[['x^3','px^3']].to_numpy()[0]
        W_out_sin3_t_4 = df_W_out_t_4[['sin^3','psin^3']].to_numpy()[0]
        W_out_x2sin_t_4 = df_W_out_t_4[['x^2sin','px^2sin']].to_numpy()[0]
        W_out_xsin2_t_4 = df_W_out_t_4[['xsin^2','pxsin^2']].to_numpy()[0]
        W_out_x4_t_4 = df_W_out_t_4[['x^4']].to_numpy()[0]
        W_out_sin4_t_4 = df_W_out_t_4[['sin^4']].to_numpy()[0]
        W_out_x3sin_t_4 = df_W_out_t_4[['x^3sin']].to_numpy()[0]
        W_out_x2sin2_t_4 = df_W_out_t_4[['x^2sin^2']].to_numpy()[0]
        W_out_xsin3_t_4 = df_W_out_t_4[['xsin^3']].to_numpy()[0]

        coef_cte_t_4 = W_out_cte_t_4[0]*one + W_out_cte_t_4[1]*t + W_out_cte_t_4[2]*t**2 + W_out_cte_t_4[3]*t**3 + W_out_cte_t_4[4]*t**4
        coef_x_t_4 = W_out_x_t_4[0]*one + W_out_x_t_4[1]*t + W_out_x_t_4[2]*t**2 + W_out_x_t_4[3]*t**3
        coef_sin_t_4 = W_out_sin_t_4[0]*one + W_out_sin_t_4[1]*t + W_out_sin_t_4[2]*t**2 + W_out_sin_t_4[3]*t**3
        coef_x2_t_4 = W_out_x2_t_4[0]*one + W_out_x2_t_4[1]*t + W_out_x2_t_4[2]*t**2
        coef_sin2_t_4 = W_out_sin2_t_4[0]*one + W_out_sin2_t_4[1]*t + W_out_sin2_t_4[2]*t**2
        coef_xsin_t_4 = W_out_xsin_t_4[0]*one + W_out_xsin_t_4[1]*t + W_out_xsin_t_4[2]*t**2
        coef_x3_t_4 = W_out_x3_t_4[0]*one + W_out_x3_t_4[1]*t
        coef_sin3_t_4 = W_out_sin3_t_4[0]*one + W_out_sin3_t_4[1]*t
        coef_x2sin_t_4 = W_out_x2sin_t_4[0]*one + W_out_x2sin_t_4[1]*t
        coef_xsin2_t_4 = W_out_xsin2_t_4[0]*one + W_out_xsin2_t_4[1]*t
        coef_x4_t_4 = W_out_x4_t_4[0]*one
        coef_sin4_t_4 = W_out_sin4_t_4[0]*one
        coef_x3sin_t_4 = W_out_x3sin_t_4[0]*one
        coef_x2sin2_t_4 = W_out_x2sin2_t_4[0]*one
        coef_xsin3_t_4 = W_out_xsin3_t_4[0]*one

        coef_matrix_t_4[0,:] = coef_cte_t_4
        coef_matrix_t_4[1,:] = coef_x_t_4
        coef_matrix_t_4[2,:] = coef_sin_t_4
        coef_matrix_t_4[3,:] = coef_x2_t_4
        coef_matrix_t_4[4,:] = coef_sin2_t_4
        coef_matrix_t_4[5,:] = coef_xsin_t_4
        coef_matrix_t_4[6,:] = coef_x3_t_4
        coef_matrix_t_4[7,:] = coef_sin3_t_4
        coef_matrix_t_4[8,:] = coef_x2sin_t_4
        coef_matrix_t_4[9,:] = coef_xsin2_t_4
        coef_matrix_t_4[10,:] = coef_x4_t_4
        coef_matrix_t_4[11,:] = coef_sin4_t_4
        coef_matrix_t_4[12,:] = coef_x3sin_t_4
        coef_matrix_t_4[13,:] = coef_x2sin2_t_4
        coef_matrix_t_4[14,:] = coef_xsin3_t_4

        smape_t_4 = sMAPE(coef_matrix_t_4,gt_coef_matrix)

        #####################################################################################

        W_out_cte_t_5 = df_W_out_t_5[['cte','p','p^2','p^3','p^4','p^5']].to_numpy()[0]
        W_out_x_t_5 = df_W_out_t_5[['x','px','p^2x','p^3x','p^4x']].to_numpy()[0]
        W_out_sin_t_5 = df_W_out_t_5[['sin','psin','p^2sin','p^3sin','p^4sin']].to_numpy()[0]
        W_out_x2_t_5 = df_W_out_t_5[['x^2','px^2','p^2x^2','p^3x^2']].to_numpy()[0]
        W_out_sin2_t_5 = df_W_out_t_5[['sin^2','psin^2','p^2sin^2','p^3sin^2']].to_numpy()[0]
        W_out_xsin_t_5 = df_W_out_t_5[['xsin','pxsin','p^2xsin','p^3xsin']].to_numpy()[0]
        W_out_x3_t_5 = df_W_out_t_5[['x^3','px^3','p^2x^3']].to_numpy()[0]
        W_out_sin3_t_5 = df_W_out_t_5[['sin^3','psin^3','p^2sin^3']].to_numpy()[0]
        W_out_x2sin_t_5 = df_W_out_t_5[['x^2sin','px^2sin','p^2x^2sin']].to_numpy()[0]
        W_out_xsin2_t_5 = df_W_out_t_5[['xsin^2','pxsin^2','p^2xsin^2']].to_numpy()[0]
        W_out_x4_t_5 = df_W_out_t_5[['x^4','px^4']].to_numpy()[0]
        W_out_sin4_t_5 = df_W_out_t_5[['sin^4','psin^4']].to_numpy()[0]
        W_out_x3sin_t_5 = df_W_out_t_5[['x^3sin','px^3sin']].to_numpy()[0]
        W_out_x2sin2_t_5 = df_W_out_t_5[['x^2sin^2','px^2sin^2']].to_numpy()[0]
        W_out_xsin3_t_5 = df_W_out_t_5[['xsin^3','pxsin^3']].to_numpy()[0]
        W_out_x5_t_5 = df_W_out_t_5[['x^5']].to_numpy()[0]
        W_out_sin5_t_5 = df_W_out_t_5[['sin^5']].to_numpy()[0]
        W_out_x4sin_t_5 = df_W_out_t_5[['x^4sin']].to_numpy()[0]
        W_out_x3sin2_t_5 = df_W_out_t_5[['x^3sin^2']].to_numpy()[0]
        W_out_x2sin3_t_5 = df_W_out_t_5[['x^2sin^3']].to_numpy()[0]
        W_out_xsin4_t_5 = df_W_out_t_5[['xsin^4']].to_numpy()[0]

        coef_cte_t_5 = W_out_cte_t_5[0]*one + W_out_cte_t_5[1]*t + W_out_cte_t_5[2]*t**2 + W_out_cte_t_5[3]*t**3 + W_out_cte_t_5[4]*t**4 + W_out_cte_t_5[5]*t**5
        coef_x_t_5 = W_out_x_t_5[0]*one + W_out_x_t_5[1]*t + W_out_x_t_5[2]*t**2 + W_out_x_t_5[3]*t**3 + W_out_x_t_5[4]*t**4
        coef_sin_t_5 = W_out_sin_t_5[0]*one + W_out_sin_t_5[1]*t + W_out_sin_t_5[2]*t**2 + W_out_sin_t_5[3]*t**3 + W_out_sin_t_5[4]*t**4
        coef_x2_t_5 = W_out_x2_t_5[0]*one + W_out_x2_t_5[1]*t + W_out_x2_t_5[2]*t**2 + W_out_x2_t_5[3]*t**3
        coef_sin2_t_5 = W_out_sin2_t_5[0]*one + W_out_sin2_t_5[1]*t + W_out_sin2_t_5[2]*t**2 + W_out_sin2_t_5[3]*t**3
        coef_xsin_t_5 = W_out_xsin_t_5[0]*one + W_out_xsin_t_5[1]*t + W_out_xsin_t_5[2]*t**2 + W_out_xsin_t_5[3]*t**3
        coef_x3_t_5 = W_out_x3_t_5[0]*one + W_out_x3_t_5[1]*t + W_out_x3_t_5[2]*t**2
        coef_sin3_t_5 = W_out_sin3_t_5[0]*one + W_out_sin3_t_5[1]*t + W_out_sin3_t_5[2]*t**2
        coef_x2sin_t_5 = W_out_x2sin_t_5[0]*one + W_out_x2sin_t_5[1]*t + W_out_x2sin_t_5[2]*t**2
        coef_xsin2_t_5 = W_out_xsin2_t_5[0]*one + W_out_xsin2_t_5[1]*t + W_out_xsin2_t_5[2]*t**2
        coef_x4_t_5 = W_out_x4_t_5[0]*one + W_out_x4_t_5[1]*t
        coef_sin4_t_5 = W_out_sin4_t_5[0]*one + W_out_sin4_t_5[1]*t
        coef_x3sin_t_5 = W_out_x3sin_t_5[0]*one + W_out_x3sin_t_5[1]*t
        coef_x2sin2_t_5 = W_out_x2sin2_t_5[0]*one + W_out_x2sin2_t_5[1]*t
        coef_xsin3_t_5 = W_out_xsin3_t_5[0]*one + W_out_xsin3_t_5[1]*t
        coef_x5_t_5 = W_out_x5_t_5[0]*one
        coef_sin5_t_5 = W_out_sin5_t_5[0]*one
        coef_x4sin_t_5 = W_out_x4sin_t_5[0]*one
        coef_x3sin2_t_5 = W_out_x3sin2_t_5[0]*one
        coef_x2sin3_t_5 = W_out_x2sin3_t_5[0]*one
        coef_xsin4_t_5 = W_out_xsin4_t_5[0]*one

        coef_matrix_t_5[0,:] = coef_cte_t_5
        coef_matrix_t_5[1,:] = coef_x_t_5
        coef_matrix_t_5[2,:] = coef_sin_t_5
        coef_matrix_t_5[3,:] = coef_x2_t_5
        coef_matrix_t_5[4,:] = coef_sin2_t_5
        coef_matrix_t_5[5,:] = coef_xsin_t_5
        coef_matrix_t_5[6,:] = coef_x3_t_5
        coef_matrix_t_5[7,:] = coef_sin3_t_5
        coef_matrix_t_5[8,:] = coef_x2sin_t_5
        coef_matrix_t_5[9,:] = coef_xsin2_t_5
        coef_matrix_t_5[10,:] = coef_x4_t_5
        coef_matrix_t_5[11,:] = coef_sin4_t_5
        coef_matrix_t_5[12,:] = coef_x3sin_t_5
        coef_matrix_t_5[13,:] = coef_x2sin2_t_5
        coef_matrix_t_5[14,:] = coef_xsin3_t_5
        coef_matrix_t_5[15,:] = coef_x5_t_5
        coef_matrix_t_5[16,:] = coef_sin5_t_5
        coef_matrix_t_5[17,:] = coef_x4sin_t_5
        coef_matrix_t_5[18,:] = coef_x3sin2_t_5
        coef_matrix_t_5[19,:] = coef_x2sin3_t_5
        coef_matrix_t_5[20,:] = coef_xsin4_t_5

        smape_t_5 = sMAPE(coef_matrix_t_5,gt_coef_matrix)

        #####################################################################################

        smape_matrix_2[row_count,:] = smape_2
        smape_matrix_3[row_count,:] = smape_3
        smape_matrix_4[row_count,:] = smape_4
        smape_matrix_5[row_count,:] = smape_5

        smape_matrix_p_2[row_count,:] = smape_p_2
        smape_matrix_p_3[row_count,:] = smape_p_3
        smape_matrix_p_4[row_count,:] = smape_p_4
        smape_matrix_p_5[row_count,:] = smape_p_5

        smape_matrix_t_2[row_count,:] = smape_t_2
        smape_matrix_t_3[row_count,:] = smape_t_3
        smape_matrix_t_4[row_count,:] = smape_t_4
        smape_matrix_t_5[row_count,:] = smape_t_5

        row_count += 1    

    smape_matrix_2 = np.mean(smape_matrix_2[:,10:], axis=1)
    smape_matrix_3 = np.mean(smape_matrix_3[:,10:], axis=1)
    smape_matrix_4 = np.mean(smape_matrix_4[:,10:], axis=1)
    smape_matrix_5 = np.mean(smape_matrix_5[:,10:], axis=1)
    
    smape_matrix_2 = smape_matrix_2[~np.isnan(smape_matrix_2)]
    smape_matrix_3 = smape_matrix_3[~np.isnan(smape_matrix_3)]
    smape_matrix_4 = smape_matrix_4[~np.isnan(smape_matrix_4)]
    smape_matrix_5 = smape_matrix_5[~np.isnan(smape_matrix_5)]

    smape_matrix_p_2 = np.mean(smape_matrix_p_2[:,10:], axis=1)
    smape_matrix_p_3 = np.mean(smape_matrix_p_3[:,10:], axis=1)
    smape_matrix_p_4 = np.mean(smape_matrix_p_4[:,10:], axis=1)
    smape_matrix_p_5 = np.mean(smape_matrix_p_5[:,10:], axis=1)

    smape_matrix_p_2 = smape_matrix_p_2[~np.isnan(smape_matrix_p_2)]
    smape_matrix_p_3 = smape_matrix_p_3[~np.isnan(smape_matrix_p_3)]
    smape_matrix_p_4 = smape_matrix_p_4[~np.isnan(smape_matrix_p_4)]
    smape_matrix_p_5 = smape_matrix_p_5[~np.isnan(smape_matrix_p_5)]
    
    smape_matrix_t_2 = np.mean(smape_matrix_t_2[:,10:], axis=1)
    smape_matrix_t_3 = np.mean(smape_matrix_t_3[:,10:], axis=1)
    smape_matrix_t_4 = np.mean(smape_matrix_t_4[:,10:], axis=1)
    smape_matrix_t_5 = np.mean(smape_matrix_t_5[:,10:], axis=1)
    
    smape_matrix_t_2 = smape_matrix_t_2[~np.isnan(smape_matrix_t_2)]
    smape_matrix_t_3 = smape_matrix_t_3[~np.isnan(smape_matrix_t_3)]
    smape_matrix_t_4 = smape_matrix_t_4[~np.isnan(smape_matrix_t_4)]
    smape_matrix_t_5 = smape_matrix_t_5[~np.isnan(smape_matrix_t_5)]

    z = stats.norm.ppf(0.95)

    means_2 = np.mean(smape_matrix_2, axis=0)
    std_errors_2 = stats.sem(smape_matrix_2, axis=0)
    ci_lower_2 = means_2 - z * std_errors_2
    ci_upper_2 = means_2 + z * std_errors_2

    means_3 = np.mean(smape_matrix_3, axis=0)
    std_errors_3 = stats.sem(smape_matrix_3, axis=0)
    ci_lower_3 = means_3 - z * std_errors_3
    ci_upper_3 = means_3 + z * std_errors_3

    means_4 = np.mean(smape_matrix_4, axis=0)
    std_errors_4 = stats.sem(smape_matrix_4, axis=0)
    ci_lower_4 = means_4 - z * std_errors_4
    ci_upper_4 = means_4 + z * std_errors_4

    means_5 = np.mean(smape_matrix_5, axis=0)
    std_errors_5 = stats.sem(smape_matrix_5, axis=0)
    ci_lower_5 = means_5 - z * std_errors_5
    ci_upper_5 = means_5 + z * std_errors_5

    means_list = list([means_2, means_3, means_4, means_5])
    ci_lower_list = list([ci_lower_2, ci_lower_3, ci_lower_4, ci_lower_5])
    ci_upper_list = list([ci_upper_2, ci_upper_3, ci_upper_4, ci_upper_5])

    #####################################################################################

    means_p_2 = np.mean(smape_matrix_p_2, axis=0)
    std_errors_p_2 = stats.sem(smape_matrix_p_2, axis=0)
    ci_lower_p_2 = means_p_2 - z * std_errors_p_2
    ci_upper_p_2 = means_p_2 + z * std_errors_p_2

    means_p_3 = np.mean(smape_matrix_p_3, axis=0)
    std_errors_p_3 = stats.sem(smape_matrix_p_3, axis=0)
    ci_lower_p_3 = means_p_3 - z * std_errors_p_3
    ci_upper_p_3 = means_p_3 + z * std_errors_p_3

    means_p_4 = np.mean(smape_matrix_p_4, axis=0)
    std_errors_p_4 = stats.sem(smape_matrix_p_4, axis=0)
    ci_lower_p_4 = means_p_4 - z * std_errors_p_4
    ci_upper_p_4 = means_p_4 + z * std_errors_p_4

    means_p_5 = np.mean(smape_matrix_p_5, axis=0)
    std_errors_p_5 = stats.sem(smape_matrix_p_5, axis=0)
    ci_lower_p_5 = means_p_5 - z * std_errors_p_5
    ci_upper_p_5 = means_p_5 + z * std_errors_p_5

    means_list_p = list([means_p_2, means_p_3, means_p_4, means_p_5])
    ci_lower_list_p = list([ci_lower_p_2, ci_lower_p_3, ci_lower_p_4, ci_lower_p_5])
    ci_upper_list_p = list([ci_upper_p_2, ci_upper_p_3, ci_upper_p_4, ci_upper_p_5])

    #####################################################################################

    means_t_2 = np.mean(smape_matrix_t_2, axis=0)
    std_errors_t_2 = stats.sem(smape_matrix_t_2, axis=0)
    ci_lower_t_2 = means_t_2 - z * std_errors_t_2
    ci_upper_t_2 = means_t_2 + z * std_errors_t_2

    means_t_3 = np.mean(smape_matrix_t_3, axis=0)
    std_errors_t_3 = stats.sem(smape_matrix_t_3, axis=0)
    ci_lower_t_3 = means_t_3 - z * std_errors_t_3
    ci_upper_t_3 = means_t_3 + z * std_errors_t_3

    means_t_4 = np.mean(smape_matrix_t_4, axis=0)
    std_errors_t_4 = stats.sem(smape_matrix_t_4, axis=0)
    ci_lower_t_4 = means_t_4 - z * std_errors_t_4
    ci_upper_t_4 = means_t_4 + z * std_errors_t_4

    means_t_5 = np.mean(smape_matrix_t_5, axis=0)
    std_errors_t_5 = stats.sem(smape_matrix_t_5, axis=0)
    ci_lower_t_5 = means_t_5 - z * std_errors_t_5
    ci_upper_t_5 = means_t_5 + z * std_errors_t_5

    means_list_t = list([means_t_2, means_t_3, means_t_4, means_t_5])
    ci_lower_list_t = list([ci_lower_t_2, ci_lower_t_3, ci_lower_t_4, ci_lower_t_5])
    ci_upper_list_t = list([ci_upper_t_2, ci_upper_t_3, ci_upper_t_4, ci_upper_t_5])

    dic_smape = {'library':[2,3,4,5], 
                'mean_smape':means_list, 'lower_smape':ci_lower_list, 'upper_smape':ci_upper_list,
                'mean_smape_p':means_list_p, 'lower_smape_p':ci_lower_list_p, 'upper_smape_p':ci_upper_list_p,
                'mean_smape_t':means_list_t, 'lower_smape_t':ci_lower_list_t, 'upper_smape_t':ci_upper_list_t}

    smape_out = pd.DataFrame(dic_smape)
    smape_out.to_csv('../results/sMAPE/library/Koscillators_smape_{}.csv'.format(tl),header = True, index=False)

