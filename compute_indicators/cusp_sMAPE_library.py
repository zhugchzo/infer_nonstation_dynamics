import numpy as np
import pandas as pd
import os
import scipy.stats as stats

if not os.path.exists('../results/sMAPE/library'):
    os.makedirs('../results/sMAPE/library')

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

for tl in [500, 550, 600, 650, 700]:

    smape_matrix_3 = np.zeros((100,length))
    smape_matrix_4 = np.zeros((100,length))
    smape_matrix_5 = np.zeros((100,length))
    smape_matrix_6 = np.zeros((100,length))

    smape_matrix_ab_3 = np.zeros((100,length))
    smape_matrix_ab_4 = np.zeros((100,length))
    smape_matrix_ab_5 = np.zeros((100,length))
    smape_matrix_ab_6 = np.zeros((100,length))

    smape_matrix_t_3 = np.zeros((100,length))
    smape_matrix_t_4 = np.zeros((100,length))
    smape_matrix_t_5 = np.zeros((100,length))
    smape_matrix_t_6 = np.zeros((100,length))

    row_count = 0

    for al in [0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2]:
        for bl in [4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5]:

            df_W_out_3 = pd.read_csv('../results/cusp/library3/{}/cusp_W_out_{}_{}.csv'.format(tl,al,bl))
            df_W_out_4 = pd.read_csv('../results/cusp/library4/{}/cusp_W_out_{}_{}.csv'.format(tl,al,bl))
            df_W_out_5 = pd.read_csv('../results/cusp/library5/{}/cusp_W_out_{}_{}.csv'.format(tl,al,bl))
            df_W_out_6 = pd.read_csv('../results/cusp/library6/{}/cusp_W_out_{}_{}.csv'.format(tl,al,bl))

            df_W_out_ab_3 = pd.read_csv('../results/cusp/library3/{}/cusp_W_out_ab_{}_{}.csv'.format(tl,al,bl))
            df_W_out_ab_4 = pd.read_csv('../results/cusp/library4/{}/cusp_W_out_ab_{}_{}.csv'.format(tl,al,bl))
            df_W_out_ab_5 = pd.read_csv('../results/cusp/library5/{}/cusp_W_out_ab_{}_{}.csv'.format(tl,al,bl))
            df_W_out_ab_6 = pd.read_csv('../results/cusp/library6/{}/cusp_W_out_ab_{}_{}.csv'.format(tl,al,bl))

            df_W_out_t_3 = pd.read_csv('../results/cusp/library3/{}/cusp_W_out_t_{}_{}.csv'.format(tl,al,bl))
            df_W_out_t_4 = pd.read_csv('../results/cusp/library4/{}/cusp_W_out_t_{}_{}.csv'.format(tl,al,bl))
            df_W_out_t_5 = pd.read_csv('../results/cusp/library5/{}/cusp_W_out_t_{}_{}.csv'.format(tl,al,bl))
            df_W_out_t_6 = pd.read_csv('../results/cusp/library6/{}/cusp_W_out_t_{}_{}.csv'.format(tl,al,bl))

            df_ab = pd.read_csv('../cusp/cusp_data/cusp_data_{}_{}.csv'.format(al,bl))

            one = np.ones(length)

            initial_theta_3 = df_W_out_3['initial_theta'].to_numpy()[0]
            delta_theta_3 = df_W_out_3['delta_theta'].to_numpy()[0]
            initial_theta_4 = df_W_out_4['initial_theta'].to_numpy()[0]
            delta_theta_4 = df_W_out_4['delta_theta'].to_numpy()[0]
            initial_theta_5 = df_W_out_5['initial_theta'].to_numpy()[0]
            delta_theta_5 = df_W_out_5['delta_theta'].to_numpy()[0]
            initial_theta_6 = df_W_out_6['initial_theta'].to_numpy()[0]
            delta_theta_6 = df_W_out_6['delta_theta'].to_numpy()[0]

            a = df_ab['a'].to_numpy()
            b = df_ab['b'].to_numpy()

            theta_3 = np.linspace(initial_theta_3, initial_theta_3 + (length - 1) * delta_theta_3, length)
            theta_4 = np.linspace(initial_theta_4, initial_theta_4 + (length - 1) * delta_theta_4, length)
            theta_5 = np.linspace(initial_theta_5, initial_theta_5 + (length - 1) * delta_theta_5, length)
            theta_6 = np.linspace(initial_theta_6, initial_theta_6 + (length - 1) * delta_theta_6, length)

            t = np.linspace(0, 0 + (length - 1) * 0.01, length)

            gt_coef_matrix = np.zeros((7,length))

            coef_matrix_3 = np.zeros((7,length))
            coef_matrix_4 = np.zeros((7,length))
            coef_matrix_5 = np.zeros((7,length))
            coef_matrix_6 = np.zeros((7,length))

            coef_matrix_ab_3 = np.zeros((7,length))
            coef_matrix_ab_4 = np.zeros((7,length))
            coef_matrix_ab_5 = np.zeros((7,length))
            coef_matrix_ab_6 = np.zeros((7,length))

            coef_matrix_t_3 = np.zeros((7,length))
            coef_matrix_t_4 = np.zeros((7,length))
            coef_matrix_t_5 = np.zeros((7,length))
            coef_matrix_t_6 = np.zeros((7,length))

            gt_coef_matrix[0,:] = a
            gt_coef_matrix[1,:] = b
            gt_coef_matrix[3,:] = -1*one

            #####################################################################################

            W_out_cte_3 = df_W_out_3[['cte','p','p^2','p^3']].to_numpy()[0]
            W_out_x_3 = df_W_out_3[['x','px','p^2x']].to_numpy()[0]
            W_out_x2_3 = df_W_out_3[['x^2','px^2']].to_numpy()[0]
            W_out_x3_3 = df_W_out_3[['x^3']].to_numpy()[0]

            coef_cte_3 = W_out_cte_3[0]*one + W_out_cte_3[1]*theta_3 + W_out_cte_3[2]*theta_3**2 + W_out_cte_3[3]*theta_3**3
            coef_x_3 = W_out_x_3[0]*one + W_out_x_3[1]*theta_3 + W_out_x_3[2]*theta_3**2
            coef_x2_3 = W_out_x2_3[0]*one + W_out_x2_3[1]*theta_3
            coef_x3_3 = W_out_x3_3[0]*one

            coef_matrix_3[0,:] = coef_cte_3
            coef_matrix_3[1,:] = coef_x_3
            coef_matrix_3[2,:] = coef_x2_3
            coef_matrix_3[3,:] = coef_x3_3

            smape_3 = sMAPE(coef_matrix_3,gt_coef_matrix)

            #####################################################################################

            W_out_cte_4 = df_W_out_4[['cte','p','p^2','p^3','p^4']].to_numpy()[0]
            W_out_x_4 = df_W_out_4[['x','px','p^2x','p^3x']].to_numpy()[0]
            W_out_x2_4 = df_W_out_4[['x^2','px^2','p^2x^2']].to_numpy()[0]
            W_out_x3_4 = df_W_out_4[['x^3','px^3']].to_numpy()[0]
            W_out_x4_4 = df_W_out_4[['x^4']].to_numpy()[0]

            coef_cte_4 = W_out_cte_4[0]*one + W_out_cte_4[1]*theta_4 + W_out_cte_4[2]*theta_4**2 + W_out_cte_4[3]*theta_4**3 + W_out_cte_4[4]*theta_4**4
            coef_x_4 = W_out_x_4[0]*one + W_out_x_4[1]*theta_4 + W_out_x_4[2]*theta_4**2 + W_out_x_4[3]*theta_4**3
            coef_x2_4 = W_out_x2_4[0]*one + W_out_x2_4[1]*theta_4 + W_out_x2_4[2]*theta_4**2
            coef_x3_4 = W_out_x3_4[0]*one + W_out_x3_4[1]*theta_4
            coef_x4_4 = W_out_x4_4[0]*one

            coef_matrix_4[0,:] = coef_cte_4
            coef_matrix_4[1,:] = coef_x_4
            coef_matrix_4[2,:] = coef_x2_4
            coef_matrix_4[3,:] = coef_x3_4
            coef_matrix_4[4,:] = coef_x4_4

            smape_4 = sMAPE(coef_matrix_4,gt_coef_matrix)

            #####################################################################################

            W_out_cte_5 = df_W_out_5[['cte','p','p^2','p^3','p^4','p^5']].to_numpy()[0]
            W_out_x_5 = df_W_out_5[['x','px','p^2x','p^3x','p^4x']].to_numpy()[0]
            W_out_x2_5 = df_W_out_5[['x^2','px^2','p^2x^2','p^3x^2']].to_numpy()[0]
            W_out_x3_5 = df_W_out_5[['x^3','px^3','p^2x^3']].to_numpy()[0]
            W_out_x4_5 = df_W_out_5[['x^4','px^4']].to_numpy()[0]
            W_out_x5_5 = df_W_out_5[['x^5']].to_numpy()[0]

            coef_cte_5 = W_out_cte_5[0]*one + W_out_cte_5[1]*theta_5 + W_out_cte_5[2]*theta_5**2 + W_out_cte_5[3]*theta_5**3 + W_out_cte_5[4]*theta_5**4 + W_out_cte_5[5]*theta_5**5
            coef_x_5 = W_out_x_5[0]*one + W_out_x_5[1]*theta_5 + W_out_x_5[2]*theta_5**2 + W_out_x_5[3]*theta_5**3 + W_out_x_5[4]*theta_5**4
            coef_x2_5 = W_out_x2_5[0]*one + W_out_x2_5[1]*theta_5 + W_out_x2_5[2]*theta_5**2 + W_out_x2_5[3]*theta_5**3
            coef_x3_5 = W_out_x3_5[0]*one + W_out_x3_5[1]*theta_5 + W_out_x3_5[2]*theta_5**2
            coef_x4_5 = W_out_x4_5[0]*one + W_out_x4_5[1]*theta_5
            coef_x5_5 = W_out_x5_5[0]*one

            coef_matrix_5[0,:] = coef_cte_5
            coef_matrix_5[1,:] = coef_x_5
            coef_matrix_5[2,:] = coef_x2_5
            coef_matrix_5[3,:] = coef_x3_5
            coef_matrix_5[4,:] = coef_x4_5
            coef_matrix_5[5,:] = coef_x5_5

            smape_5 = sMAPE(coef_matrix_5,gt_coef_matrix)

            #####################################################################################

            W_out_cte_6 = df_W_out_6[['cte','p','p^2','p^3','p^4','p^5','p^6']].to_numpy()[0]
            W_out_x_6 = df_W_out_6[['x','px','p^2x','p^3x','p^4x','p^5x']].to_numpy()[0]
            W_out_x2_6 = df_W_out_6[['x^2','px^2','p^2x^2','p^3x^2','p^4x^2']].to_numpy()[0]
            W_out_x3_6 = df_W_out_6[['x^3','px^3','p^2x^3','p^3x^3']].to_numpy()[0]
            W_out_x4_6 = df_W_out_6[['x^4','px^4','p^2x^4']].to_numpy()[0]
            W_out_x5_6 = df_W_out_6[['x^5','px^5']].to_numpy()[0]
            W_out_x6_6 = df_W_out_6[['x^6']].to_numpy()[0]

            coef_cte_6 = W_out_cte_6[0]*one + W_out_cte_6[1]*theta_6 + W_out_cte_6[2]*theta_6**2 + W_out_cte_6[3]*theta_6**3 + W_out_cte_6[4]*theta_6**4 + W_out_cte_6[5]*theta_6**5 + W_out_cte_6[6]*theta_6**6
            coef_x_6 = W_out_x_6[0]*one + W_out_x_6[1]*theta_6 + W_out_x_6[2]*theta_6**2 + W_out_x_6[3]*theta_6**3 + W_out_x_6[4]*theta_6**4 + W_out_x_6[5]*theta_6**5
            coef_x2_6 = W_out_x2_6[0]*one + W_out_x2_6[1]*theta_6 + W_out_x2_6[2]*theta_6**2 + W_out_x2_6[3]*theta_6**3 + W_out_x2_6[4]*theta_6**4
            coef_x3_6 = W_out_x3_6[0]*one + W_out_x3_6[1]*theta_6 + W_out_x3_6[2]*theta_6**2 + W_out_x3_6[3]*theta_6**3
            coef_x4_6 = W_out_x4_6[0]*one + W_out_x4_6[1]*theta_6 + W_out_x4_6[2]*theta_6**2
            coef_x5_6 = W_out_x5_6[0]*one + W_out_x5_6[1]*theta_6
            coef_x6_6 = W_out_x6_6[0]*one

            coef_matrix_6[0,:] = coef_cte_6
            coef_matrix_6[1,:] = coef_x_6
            coef_matrix_6[2,:] = coef_x2_6
            coef_matrix_6[3,:] = coef_x3_6
            coef_matrix_6[4,:] = coef_x4_6
            coef_matrix_6[5,:] = coef_x5_6
            coef_matrix_6[6,:] = coef_x6_6

            smape_6 = sMAPE(coef_matrix_6,gt_coef_matrix)

            #####################################################################################

            W_out_cte_ab_3 = df_W_out_ab_3[['cte','a','b','a^2','ab','b^2','a^3','a^2b','ab^2','b^3']].to_numpy()[0]
            W_out_x_ab_3 = df_W_out_ab_3[['x','ax','bx','a^2x','abx','b^2x']].to_numpy()[0]
            W_out_x2_ab_3 = df_W_out_ab_3[['x^2','ax^2','bx^2']].to_numpy()[0]
            W_out_x3_ab_3 = df_W_out_ab_3[['x^3']].to_numpy()[0]

            coef_cte_ab_3 = W_out_cte_ab_3[0]*one + W_out_cte_ab_3[1]*a + W_out_cte_ab_3[2]*b + W_out_cte_ab_3[3]*a**2 + W_out_cte_ab_3[4]*a*b + W_out_cte_ab_3[5]*b**2 + W_out_cte_ab_3[6]*a**3 + W_out_cte_ab_3[7]*a**2*b + W_out_cte_ab_3[8]*a*b**2 + W_out_cte_ab_3[9]*b**3
            coef_x_ab_3 = W_out_x_ab_3[0]*one + W_out_x_ab_3[1]*a + W_out_x_ab_3[2]*b + W_out_x_ab_3[3]*a**2 + W_out_x_ab_3[4]*a*b + W_out_x_ab_3[5]*b**2
            coef_x2_ab_3 = W_out_x2_ab_3[0]*one + W_out_x2_ab_3[1]*a + W_out_x2_ab_3[2]*b
            coef_x3_ab_3 = W_out_x3_ab_3[0]*one

            coef_matrix_ab_3[0,:] = coef_cte_ab_3
            coef_matrix_ab_3[1,:] = coef_x_ab_3
            coef_matrix_ab_3[2,:] = coef_x2_ab_3
            coef_matrix_ab_3[3,:] = coef_x3_ab_3

            smape_ab_3 = sMAPE(coef_matrix_ab_3,gt_coef_matrix)

            #####################################################################################

            W_out_cte_ab_4 = df_W_out_ab_4[['cte','a','b','a^2','ab','b^2','a^3','a^2b','ab^2','b^3','a^4','a^3b','a^2b^2','ab^3','b^4']].to_numpy()[0]
            W_out_x_ab_4 = df_W_out_ab_4[['x','ax','bx','a^2x','abx','b^2x','a^3x','a^2bx','ab^2x','b^3x']].to_numpy()[0]
            W_out_x2_ab_4 = df_W_out_ab_4[['x^2','ax^2','bx^2','a^2x^2','abx^2','b^2x^2']].to_numpy()[0]
            W_out_x3_ab_4 = df_W_out_ab_4[['x^3','ax^3','bx^3']].to_numpy()[0]
            W_out_x4_ab_4 = df_W_out_ab_4[['x^4']].to_numpy()[0]

            coef_cte_ab_4 = (W_out_cte_ab_4[0]*one + W_out_cte_ab_4[1]*a + W_out_cte_ab_4[2]*b + W_out_cte_ab_4[3]*a**2 + W_out_cte_ab_4[4]*a*b + W_out_cte_ab_4[5]*b**2 + W_out_cte_ab_4[6]*a**3 + W_out_cte_ab_4[7]*a**2*b + W_out_cte_ab_4[8]*a*b**2 + W_out_cte_ab_4[9]*b**3
                            + W_out_cte_ab_4[10]*a**4 + W_out_cte_ab_4[11]*a**3*b + W_out_cte_ab_4[12]*a**2*b**2 + W_out_cte_ab_4[13]*a*b**3 + W_out_cte_ab_4[14]*b**4)
            coef_x_ab_4 = (W_out_x_ab_4[0]*one + W_out_x_ab_4[1]*a + W_out_x_ab_4[2]*b + W_out_x_ab_4[3]*a**2 + W_out_x_ab_4[4]*a*b + W_out_x_ab_4[5]*b**2
                        + W_out_x_ab_4[6]*a**3 + W_out_x_ab_4[7]*a**2*b + W_out_x_ab_4[8]*a*b**2 + W_out_x_ab_4[9]*b**3)
            coef_x2_ab_4 = W_out_x2_ab_4[0]*one + W_out_x2_ab_4[1]*a + W_out_x2_ab_4[2]*b + W_out_x2_ab_4[3]*a**2 + W_out_x2_ab_4[4]*a*b + W_out_x2_ab_4[5]*b**2
            coef_x3_ab_4 = W_out_x3_ab_4[0]*one + W_out_x3_ab_4[1]*a + W_out_x3_ab_4[2]*b
            coef_x4_ab_4 = W_out_x4_ab_4[0]*one

            coef_matrix_ab_4[0,:] = coef_cte_ab_4
            coef_matrix_ab_4[1,:] = coef_x_ab_4
            coef_matrix_ab_4[2,:] = coef_x2_ab_4
            coef_matrix_ab_4[3,:] = coef_x3_ab_4
            coef_matrix_ab_4[4,:] = coef_x4_ab_4

            smape_ab_4 = sMAPE(coef_matrix_ab_4,gt_coef_matrix)

            #####################################################################################

            W_out_cte_ab_5 = df_W_out_ab_5[['cte','a','b','a^2','ab','b^2','a^3','a^2b','ab^2','b^3','a^4','a^3b','a^2b^2','ab^3','b^4','a^5','a^4b','a^3b^2','a^2b^3','ab^4','b^5']].to_numpy()[0]
            W_out_x_ab_5 = df_W_out_ab_5[['x','ax','bx','a^2x','abx','b^2x','a^3x','a^2bx','ab^2x','b^3x','a^4x','a^3bx','a^2b^2x','ab^3x','b^4x']].to_numpy()[0]
            W_out_x2_ab_5 = df_W_out_ab_5[['x^2','ax^2','bx^2','a^2x^2','abx^2','b^2x^2','a^3x^2','a^2bx^2','ab^2x^2','b^3x^2']].to_numpy()[0]
            W_out_x3_ab_5 = df_W_out_ab_5[['x^3','ax^3','bx^3','a^2x^3','abx^3','b^2x^3']].to_numpy()[0]
            W_out_x4_ab_5 = df_W_out_ab_5[['x^4','ax^4','bx^4']].to_numpy()[0]
            W_out_x5_ab_5 = df_W_out_ab_5[['x^5']].to_numpy()[0]

            coef_cte_ab_5 = (W_out_cte_ab_5[0]*one + W_out_cte_ab_5[1]*a + W_out_cte_ab_5[2]*b + W_out_cte_ab_5[3]*a**2 + W_out_cte_ab_5[4]*a*b + W_out_cte_ab_5[5]*b**2 + W_out_cte_ab_5[6]*a**3 + W_out_cte_ab_5[7]*a**2*b + W_out_cte_ab_5[8]*a*b**2 + W_out_cte_ab_5[9]*b**3
                            + W_out_cte_ab_5[10]*a**4 + W_out_cte_ab_5[11]*a**3*b + W_out_cte_ab_5[12]*a**2*b**2 + W_out_cte_ab_5[13]*a*b**3 + W_out_cte_ab_5[14]*b**4
                            + W_out_cte_ab_5[15]*a**5 + W_out_cte_ab_5[16]*a**4*b + W_out_cte_ab_5[17]*a**3*b**2 + W_out_cte_ab_5[18]*a**2*b**3 + W_out_cte_ab_5[19]*a*b**4 + W_out_cte_ab_5[20]*b**5)
            coef_x_ab_5 = (W_out_x_ab_5[0]*one + W_out_x_ab_5[1]*a + W_out_x_ab_5[2]*b + W_out_x_ab_5[3]*a**2 + W_out_x_ab_5[4]*a*b + W_out_x_ab_5[5]*b**2
                           + W_out_x_ab_5[6]*a**3 + W_out_x_ab_5[7]*a**2*b + W_out_x_ab_5[8]*a*b**2 + W_out_x_ab_5[9]*b**3
                           + W_out_x_ab_5[10]*a**4 + W_out_x_ab_5[11]*a**3*b + W_out_x_ab_5[12]*a**2*b**2 + W_out_x_ab_5[13]*a*b**3 + W_out_x_ab_5[14]*b**4)
            coef_x2_ab_5 = W_out_x2_ab_5[0]*one + W_out_x2_ab_5[1]*a + W_out_x2_ab_5[2]*b + W_out_x2_ab_5[3]*a**2 + W_out_x2_ab_5[4]*a*b + W_out_x2_ab_5[5]*b**2 + W_out_x2_ab_5[6]*a**3 + W_out_x2_ab_5[7]*a**2*b + W_out_x2_ab_5[8]*a*b**2 + W_out_x2_ab_5[9]*b**3
            coef_x3_ab_5 = W_out_x3_ab_5[0]*one + W_out_x3_ab_5[1]*a + W_out_x3_ab_5[2]*b + W_out_x3_ab_5[3]*a**2 + W_out_x3_ab_5[4]*a*b + W_out_x3_ab_5[5]*b**2
            coef_x4_ab_5 = W_out_x4_ab_5[0]*one + W_out_x4_ab_5[1]*a + W_out_x4_ab_5[2]*b
            coef_x5_ab_5 = W_out_x5_ab_5[0]*one

            coef_matrix_ab_5[0,:] = coef_cte_ab_5
            coef_matrix_ab_5[1,:] = coef_x_ab_5
            coef_matrix_ab_5[2,:] = coef_x2_ab_5
            coef_matrix_ab_5[3,:] = coef_x3_ab_5
            coef_matrix_ab_5[4,:] = coef_x4_ab_5
            coef_matrix_ab_5[5,:] = coef_x5_ab_5

            smape_ab_5 = sMAPE(coef_matrix_ab_5,gt_coef_matrix)

            #####################################################################################

            W_out_cte_ab_6 = df_W_out_ab_6[['cte','a','b','a^2','ab','b^2','a^3','a^2b','ab^2','b^3','a^4','a^3b','a^2b^2','ab^3','b^4','a^5','a^4b','a^3b^2','a^2b^3','ab^4','b^5','a^6','a^5b','a^4b^2','a^3b^3','a^2b^4','ab^5','b^6']].to_numpy()[0]
            W_out_x_ab_6 = df_W_out_ab_6[['x','ax','bx','a^2x','abx','b^2x','a^3x','a^2bx','ab^2x','b^3x','a^4x','a^3bx','a^2b^2x','ab^3x','b^4x','a^5x','a^4bx','a^3b^2x','a^2b^3x','ab^4x','b^5x']].to_numpy()[0]
            W_out_x2_ab_6 = df_W_out_ab_6[['x^2','ax^2','bx^2','a^2x^2','abx^2','b^2x^2','a^3x^2','a^2bx^2','ab^2x^2','b^3x^2','a^4x^2','a^3bx^2','a^2b^2x^2','ab^3x^2','b^4x^2']].to_numpy()[0]
            W_out_x3_ab_6 = df_W_out_ab_6[['x^3','ax^3','bx^3','a^2x^3','abx^3','b^2x^3','a^3x^3','a^2bx^3','ab^2x^3','b^3x^3']].to_numpy()[0]
            W_out_x4_ab_6 = df_W_out_ab_6[['x^4','ax^4','bx^4','a^2x^4','abx^4','b^2x^4']].to_numpy()[0]
            W_out_x5_ab_6 = df_W_out_ab_6[['x^5','ax^5','bx^5']].to_numpy()[0]
            W_out_x6_ab_6 = df_W_out_ab_6[['x^6']].to_numpy()[0]

            coef_cte_ab_6 = (W_out_cte_ab_6[0]*one + W_out_cte_ab_6[1]*a + W_out_cte_ab_6[2]*b + W_out_cte_ab_6[3]*a**2 + W_out_cte_ab_6[4]*a*b + W_out_cte_ab_6[5]*b**2 + W_out_cte_ab_6[6]*a**3 + W_out_cte_ab_6[7]*a**2*b + W_out_cte_ab_6[8]*a*b**2 + W_out_cte_ab_6[9]*b**3
                            + W_out_cte_ab_6[10]*a**4 + W_out_cte_ab_6[11]*a**3*b + W_out_cte_ab_6[12]*a**2*b**2 + W_out_cte_ab_6[13]*a*b**3 + W_out_cte_ab_6[14]*b**4
                            + W_out_cte_ab_6[15]*a**5 + W_out_cte_ab_6[16]*a**4*b + W_out_cte_ab_6[17]*a**3*b**2 + W_out_cte_ab_6[18]*a**2*b**3 + W_out_cte_ab_6[19]*a*b**4 + W_out_cte_ab_6[20]*b**5
                            + W_out_cte_ab_6[21]*a**6 + W_out_cte_ab_6[22]*a**5*b + W_out_cte_ab_6[23]*a**4*b**2 + W_out_cte_ab_6[24]*a**3*b**3 + W_out_cte_ab_6[25]*a**2*b**4 + W_out_cte_ab_6[26]*a*b**5 + W_out_cte_ab_6[27]*b**6)
            coef_x_ab_6 = (W_out_x_ab_6[0]*one + W_out_x_ab_6[1]*a + W_out_x_ab_6[2]*b + W_out_x_ab_6[3]*a**2 + W_out_x_ab_6[4]*a*b + W_out_x_ab_6[5]*b**2
                           + W_out_x_ab_6[6]*a**3 + W_out_x_ab_6[7]*a**2*b + W_out_x_ab_6[8]*a*b**2 + W_out_x_ab_6[9]*b**3
                           + W_out_x_ab_6[10]*a**4 + W_out_x_ab_6[11]*a**3*b + W_out_x_ab_6[12]*a**2*b**2 + W_out_x_ab_6[13]*a*b**3 + W_out_x_ab_6[14]*b**4
                           + W_out_x_ab_6[15]*a**5 + W_out_x_ab_6[16]*a**4*b + W_out_x_ab_6[17]*a**3*b**2 + W_out_x_ab_6[18]*a**2*b**3 + W_out_x_ab_6[19]*a*b**4 + W_out_x_ab_6[20]*b**5)
            coef_x2_ab_6 = (W_out_x2_ab_6[0]*one + W_out_x2_ab_6[1]*a + W_out_x2_ab_6[2]*b + W_out_x2_ab_6[3]*a**2 + W_out_x2_ab_6[4]*a*b + W_out_x2_ab_6[5]*b**2 + W_out_x2_ab_6[6]*a**3 + W_out_x2_ab_6[7]*a**2*b + W_out_x2_ab_6[8]*a*b**2 + W_out_x2_ab_6[9]*b**3
                           + W_out_x2_ab_6[10]*a**4 + W_out_x2_ab_6[11]*a**3*b + W_out_x2_ab_6[12]*a**2*b**2 + W_out_x2_ab_6[13]*a*b**3 + W_out_x2_ab_6[14]*b**4)
            coef_x3_ab_6 = (W_out_x3_ab_6[0]*one + W_out_x3_ab_6[1]*a + W_out_x3_ab_6[2]*b + W_out_x3_ab_6[3]*a**2 + W_out_x3_ab_6[4]*a*b + W_out_x3_ab_6[5]*b**2
                           + W_out_x3_ab_6[6]*a**3 + W_out_x3_ab_6[7]*a**2*b + W_out_x3_ab_6[8]*a*b**2 + W_out_x3_ab_6[9]*b**3)
            coef_x4_ab_6 = W_out_x4_ab_6[0]*one + W_out_x4_ab_6[1]*a + W_out_x4_ab_6[2]*b + W_out_x4_ab_6[3]*a**2 + W_out_x4_ab_6[4]*a*b + W_out_x4_ab_6[5]*b**2
            coef_x5_ab_6 = W_out_x5_ab_6[0]*one + W_out_x5_ab_6[1]*a + W_out_x5_ab_6[2]*b
            coef_x6_ab_6 = W_out_x6_ab_6[0]*one

            coef_matrix_ab_6[0,:] = coef_cte_ab_6
            coef_matrix_ab_6[1,:] = coef_x_ab_6
            coef_matrix_ab_6[2,:] = coef_x2_ab_6
            coef_matrix_ab_6[3,:] = coef_x3_ab_6
            coef_matrix_ab_6[4,:] = coef_x4_ab_6
            coef_matrix_ab_6[5,:] = coef_x5_ab_6
            coef_matrix_ab_6[6,:] = coef_x6_ab_6

            smape_ab_6 = sMAPE(coef_matrix_ab_6,gt_coef_matrix)

            #####################################################################################

            W_out_cte_t_3 = df_W_out_t_3[['cte','p','p^2','p^3']].to_numpy()[0]
            W_out_x_t_3 = df_W_out_t_3[['x','px','p^2x']].to_numpy()[0]
            W_out_x2_t_3 = df_W_out_t_3[['x^2','px^2']].to_numpy()[0]
            W_out_x3_t_3 = df_W_out_t_3[['x^3']].to_numpy()[0]

            coef_cte_t_3 = W_out_cte_t_3[0]*one + W_out_cte_t_3[1]*t + W_out_cte_t_3[2]*t**2 + W_out_cte_t_3[3]*t**3
            coef_x_t_3 = W_out_x_t_3[0]*one + W_out_x_t_3[1]*t + W_out_x_t_3[2]*t**2
            coef_x2_t_3 = W_out_x2_t_3[0]*one + W_out_x2_t_3[1]*t
            coef_x3_t_3 = W_out_x3_t_3[0]*one

            coef_matrix_t_3[0,:] = coef_cte_t_3
            coef_matrix_t_3[1,:] = coef_x_t_3
            coef_matrix_t_3[2,:] = coef_x2_t_3
            coef_matrix_t_3[3,:] = coef_x3_t_3

            smape_t_3 = sMAPE(coef_matrix_t_3,gt_coef_matrix)

            #####################################################################################

            W_out_cte_t_4 = df_W_out_t_4[['cte','p','p^2','p^3','p^4']].to_numpy()[0]
            W_out_x_t_4 = df_W_out_t_4[['x','px','p^2x','p^3x']].to_numpy()[0]
            W_out_x2_t_4 = df_W_out_t_4[['x^2','px^2','p^2x^2']].to_numpy()[0]
            W_out_x3_t_4 = df_W_out_t_4[['x^3','px^3']].to_numpy()[0]
            W_out_x4_t_4 = df_W_out_t_4[['x^4']].to_numpy()[0]

            coef_cte_t_4 = W_out_cte_t_4[0]*one + W_out_cte_t_4[1]*t + W_out_cte_t_4[2]*t**2 + W_out_cte_t_4[3]*t**3 + W_out_cte_t_4[4]*t**4
            coef_x_t_4 = W_out_x_t_4[0]*one + W_out_x_t_4[1]*t + W_out_x_t_4[2]*t**2 + W_out_x_t_4[3]*t**3
            coef_x2_t_4 = W_out_x2_t_4[0]*one + W_out_x2_t_4[1]*t + W_out_x2_t_4[2]*t**2
            coef_x3_t_4 = W_out_x3_t_4[0]*one + W_out_x3_t_4[1]*t
            coef_x4_t_4 = W_out_x4_t_4[0]*one

            coef_matrix_t_4[0,:] = coef_cte_t_4
            coef_matrix_t_4[1,:] = coef_x_t_4
            coef_matrix_t_4[2,:] = coef_x2_t_4
            coef_matrix_t_4[3,:] = coef_x3_t_4
            coef_matrix_t_4[4,:] = coef_x4_t_4

            smape_t_4 = sMAPE(coef_matrix_t_4,gt_coef_matrix)

            #####################################################################################

            W_out_cte_t_5 = df_W_out_t_5[['cte','p','p^2','p^3','p^4','p^5']].to_numpy()[0]
            W_out_x_t_5 = df_W_out_t_5[['x','px','p^2x','p^3x','p^4x']].to_numpy()[0]
            W_out_x2_t_5 = df_W_out_t_5[['x^2','px^2','p^2x^2','p^3x^2']].to_numpy()[0]
            W_out_x3_t_5 = df_W_out_t_5[['x^3','px^3','p^2x^3']].to_numpy()[0]
            W_out_x4_t_5 = df_W_out_t_5[['x^4','px^4']].to_numpy()[0]
            W_out_x5_t_5 = df_W_out_t_5[['x^5']].to_numpy()[0]

            coef_cte_t_5 = W_out_cte_t_5[0]*one + W_out_cte_t_5[1]*t + W_out_cte_t_5[2]*t**2 + W_out_cte_t_5[3]*t**3 + W_out_cte_t_5[4]*t**4 + W_out_cte_t_5[5]*t**5
            coef_x_t_5 = W_out_x_t_5[0]*one + W_out_x_t_5[1]*t + W_out_x_t_5[2]*t**2 + W_out_x_t_5[3]*t**3 + W_out_x_t_5[4]*t**4
            coef_x2_t_5 = W_out_x2_t_5[0]*one + W_out_x2_t_5[1]*t + W_out_x2_t_5[2]*t**2 + W_out_x2_t_5[3]*t**3
            coef_x3_t_5 = W_out_x3_t_5[0]*one + W_out_x3_t_5[1]*t + W_out_x3_t_5[2]*t**2
            coef_x4_t_5 = W_out_x4_t_5[0]*one + W_out_x4_t_5[1]*t
            coef_x5_t_5 = W_out_x5_t_5[0]*one

            coef_matrix_t_5[0,:] = coef_cte_t_5
            coef_matrix_t_5[1,:] = coef_x_t_5
            coef_matrix_t_5[2,:] = coef_x2_t_5
            coef_matrix_t_5[3,:] = coef_x3_t_5
            coef_matrix_t_5[4,:] = coef_x4_t_5
            coef_matrix_t_5[5,:] = coef_x5_t_5

            smape_t_5 = sMAPE(coef_matrix_t_5,gt_coef_matrix)

            #####################################################################################

            W_out_cte_t_6 = df_W_out_t_6[['cte','p','p^2','p^3','p^4','p^5','p^6']].to_numpy()[0]
            W_out_x_t_6 = df_W_out_t_6[['x','px','p^2x','p^3x','p^4x','p^5x']].to_numpy()[0]
            W_out_x2_t_6 = df_W_out_t_6[['x^2','px^2','p^2x^2','p^3x^2','p^4x^2']].to_numpy()[0]
            W_out_x3_t_6 = df_W_out_t_6[['x^3','px^3','p^2x^3','p^3x^3']].to_numpy()[0]
            W_out_x4_t_6 = df_W_out_t_6[['x^4','px^4','p^2x^4']].to_numpy()[0]
            W_out_x5_t_6 = df_W_out_t_6[['x^5','px^5']].to_numpy()[0]
            W_out_x6_t_6 = df_W_out_t_6[['x^6']].to_numpy()[0]

            coef_cte_t_6 = W_out_cte_t_6[0]*one + W_out_cte_t_6[1]*t + W_out_cte_t_6[2]*t**2 + W_out_cte_t_6[3]*t**3 + W_out_cte_t_6[4]*t**4 + W_out_cte_t_6[5]*t**5 + W_out_cte_t_6[6]*t**6
            coef_x_t_6 = W_out_x_t_6[0]*one + W_out_x_t_6[1]*t + W_out_x_t_6[2]*t**2 + W_out_x_t_6[3]*t**3 + W_out_x_t_6[4]*t**4 + W_out_x_t_6[5]*t**5
            coef_x2_t_6 = W_out_x2_t_6[0]*one + W_out_x2_t_6[1]*t + W_out_x2_t_6[2]*t**2 + W_out_x2_t_6[3]*t**3 + W_out_x2_t_6[4]*t**4
            coef_x3_t_6 = W_out_x3_t_6[0]*one + W_out_x3_t_6[1]*t + W_out_x3_t_6[2]*t**2 + W_out_x3_t_6[3]*t**3
            coef_x4_t_6 = W_out_x4_t_6[0]*one + W_out_x4_t_6[1]*t + W_out_x4_t_6[2]*t**2
            coef_x5_t_6 = W_out_x5_t_6[0]*one + W_out_x5_t_6[1]*t
            coef_x6_t_6 = W_out_x6_t_6[0]*one

            coef_matrix_t_6[0,:] = coef_cte_t_6
            coef_matrix_t_6[1,:] = coef_x_t_6
            coef_matrix_t_6[2,:] = coef_x2_t_6
            coef_matrix_t_6[3,:] = coef_x3_t_6
            coef_matrix_t_6[4,:] = coef_x4_t_6
            coef_matrix_t_6[5,:] = coef_x5_t_6
            coef_matrix_t_6[6,:] = coef_x6_t_6

            smape_t_6 = sMAPE(coef_matrix_t_6,gt_coef_matrix)

            #####################################################################################

            smape_matrix_3[row_count,:] = smape_3
            smape_matrix_4[row_count,:] = smape_4
            smape_matrix_5[row_count,:] = smape_5
            smape_matrix_6[row_count,:] = smape_6

            smape_matrix_ab_3[row_count,:] = smape_ab_3
            smape_matrix_ab_4[row_count,:] = smape_ab_4
            smape_matrix_ab_5[row_count,:] = smape_ab_5
            smape_matrix_ab_6[row_count,:] = smape_ab_6

            smape_matrix_t_3[row_count,:] = smape_t_3
            smape_matrix_t_4[row_count,:] = smape_t_4
            smape_matrix_t_5[row_count,:] = smape_t_5
            smape_matrix_t_6[row_count,:] = smape_t_6

            row_count += 1

    smape_matrix_3 = np.mean(smape_matrix_3, axis=1)
    smape_matrix_4 = np.mean(smape_matrix_4, axis=1)
    smape_matrix_5 = np.mean(smape_matrix_5, axis=1)
    smape_matrix_6 = np.mean(smape_matrix_6, axis=1)
    
    smape_matrix_3 = smape_matrix_3[~np.isnan(smape_matrix_3)]
    smape_matrix_4 = smape_matrix_4[~np.isnan(smape_matrix_4)]
    smape_matrix_5 = smape_matrix_5[~np.isnan(smape_matrix_5)]
    smape_matrix_6 = smape_matrix_6[~np.isnan(smape_matrix_6)]

    smape_matrix_ab_3 = np.mean(smape_matrix_ab_3, axis=1)
    smape_matrix_ab_4 = np.mean(smape_matrix_ab_4, axis=1)
    smape_matrix_ab_5 = np.mean(smape_matrix_ab_5, axis=1)
    smape_matrix_ab_6 = np.mean(smape_matrix_ab_6, axis=1)

    smape_matrix_ab_3 = smape_matrix_ab_3[~np.isnan(smape_matrix_ab_3)]
    smape_matrix_ab_4 = smape_matrix_ab_4[~np.isnan(smape_matrix_ab_4)]
    smape_matrix_ab_5 = smape_matrix_ab_5[~np.isnan(smape_matrix_ab_5)]
    smape_matrix_ab_6 = smape_matrix_ab_6[~np.isnan(smape_matrix_ab_6)]

    smape_matrix_t_3 = np.mean(smape_matrix_t_3, axis=1)
    smape_matrix_t_4 = np.mean(smape_matrix_t_4, axis=1)
    smape_matrix_t_5 = np.mean(smape_matrix_t_5, axis=1)
    smape_matrix_t_6 = np.mean(smape_matrix_t_6, axis=1)

    smape_matrix_t_3 = smape_matrix_t_3[~np.isnan(smape_matrix_t_3)]
    smape_matrix_t_4 = smape_matrix_t_4[~np.isnan(smape_matrix_t_4)]
    smape_matrix_t_5 = smape_matrix_t_5[~np.isnan(smape_matrix_t_5)]
    smape_matrix_t_6 = smape_matrix_t_6[~np.isnan(smape_matrix_t_6)]

    z = stats.norm.ppf(0.95)

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

    means_6 = np.mean(smape_matrix_6, axis=0)
    std_errors_6 = stats.sem(smape_matrix_6, axis=0)
    ci_lower_6 = means_6 - z * std_errors_6
    ci_upper_6 = means_6 + z * std_errors_6

    means_list = list([means_3, means_4, means_5, means_6])
    ci_lower_list = list([ci_lower_3, ci_lower_4, ci_lower_5, ci_lower_6])
    ci_upper_list = list([ci_upper_3, ci_upper_4, ci_upper_5, ci_upper_6])

    #####################################################################################

    means_ab_3 = np.mean(smape_matrix_ab_3)
    std_errors_ab_3 = stats.sem(smape_matrix_ab_3)
    ci_lower_ab_3 = means_ab_3 - z * std_errors_ab_3
    ci_upper_ab_3 = means_ab_3 + z * std_errors_ab_3

    means_ab_4 = np.mean(smape_matrix_ab_4)
    std_errors_ab_4 = stats.sem(smape_matrix_ab_4)
    ci_lower_ab_4 = means_ab_4 - z * std_errors_ab_4
    ci_upper_ab_4 = means_ab_4 + z * std_errors_ab_4

    means_ab_5 = np.mean(smape_matrix_ab_5)
    std_errors_ab_5 = stats.sem(smape_matrix_ab_5)
    ci_lower_ab_5 = means_ab_5 - z * std_errors_ab_5
    ci_upper_ab_5 = means_ab_5 + z * std_errors_ab_5

    means_ab_6 = np.mean(smape_matrix_ab_6)
    std_errors_ab_6 = stats.sem(smape_matrix_ab_6)
    ci_lower_ab_6 = means_ab_6 - z * std_errors_ab_6
    ci_upper_ab_6 = means_ab_6 + z * std_errors_ab_6

    means_list_ab = list([means_ab_3, means_ab_4, means_ab_5, means_ab_6])
    ci_lower_list_ab = list([ci_lower_ab_3, ci_lower_ab_4, ci_lower_ab_5, ci_lower_ab_6])
    ci_upper_list_ab = list([ci_upper_ab_3, ci_upper_ab_4, ci_upper_ab_5, ci_upper_ab_6])

    #####################################################################################

    means_t_3 = np.mean(smape_matrix_t_3)
    std_errors_t_3 = stats.sem(smape_matrix_t_3)
    ci_lower_t_3 = means_t_3 - z * std_errors_t_3
    ci_upper_t_3 = means_t_3 + z * std_errors_t_3

    means_t_4 = np.mean(smape_matrix_t_4)
    std_errors_t_4 = stats.sem(smape_matrix_t_4)
    ci_lower_t_4 = means_t_4 - z * std_errors_t_4
    ci_upper_t_4 = means_t_4 + z * std_errors_t_4

    means_t_5 = np.mean(smape_matrix_t_5)
    std_errors_t_5 = stats.sem(smape_matrix_t_5)
    ci_lower_t_5 = means_t_5 - z * std_errors_t_5
    ci_upper_t_5 = means_t_5 + z * std_errors_t_5

    means_t_6 = np.mean(smape_matrix_t_6)
    std_errors_t_6 = stats.sem(smape_matrix_t_6)
    ci_lower_t_6 = means_t_6 - z * std_errors_t_6
    ci_upper_t_6 = means_t_6 + z * std_errors_t_6

    means_list_t = list([means_t_3, means_t_4, means_t_5, means_t_6])
    ci_lower_list_t = list([ci_lower_t_3, ci_lower_t_4, ci_lower_t_5, ci_lower_t_6])
    ci_upper_list_t = list([ci_upper_t_3, ci_upper_t_4, ci_upper_t_5, ci_upper_t_6])

    #####################################################################################

    dic_smape = {'library':[3,4,5,6], 
                'mean_smape':means_list, 'lower_smape':ci_lower_list, 'upper_smape':ci_upper_list,
                'mean_smape_ab':means_list_ab, 'lower_smape_ab':ci_lower_list_ab, 'upper_smape_ab':ci_upper_list_ab,
                'mean_smape_t':means_list_t, 'lower_smape_t':ci_lower_list_t, 'upper_smape_t':ci_upper_list_t}

    smape_out = pd.DataFrame(dic_smape)
    smape_out.to_csv('../results/sMAPE/library/cusp_smape_{}.csv'.format(tl),header = True, index=False)

