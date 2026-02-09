import numpy as np
import pandas as pd
import os

if not os.path.exists('../../results/cusp/library4/prediction'):
    os.makedirs('../../results/cusp/library4/prediction')

if not os.path.exists('../../results/cusp/library5/prediction'):
    os.makedirs('../../results/cusp/library5/prediction')

if not os.path.exists('../../results/cusp/library6/prediction'):
    os.makedirs('../../results/cusp/library6/prediction')

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
        
    return np.sum(sMAPE/M)

def fun4(x,a,b,c,d,e):
    return a + b*x + c*x**2 + d*x**3 + e*x**4

def fun5(x,a,b,c,d,e,f):
    return a + b*x + c*x**2 + d*x**3 + e*x**4 + f*x**5

def fun6(x,a,b,c,d,e,f,g):
    return a + b*x + c*x**2 + d*x**3 + e*x**4 + f*x**5 + g*x**6

def recov_fun4(x, b, c, d, e):
    return b + 2*c*x + 3*d*x**2 + 4*e*x**3

def recov_fun5(x, b, c, d, e, f):
    return b + 2*c*x + 3*d*x**2 + 4*e*x**3 + 5*f*x**4

def recov_fun6(x, b, c, d, e, f, g):
    return b + 2*c*x + 3*d*x**2 + 4*e*x**3 + 5*f*x**4 + 6*g*x**5

total_length = 1000

for tl in [500]:

    length = total_length - tl

    pred_x4 = np.zeros(length)
    pred_x5 = np.zeros(length)
    pred_x6 = np.zeros(length)
    rrate_x4 = np.zeros(length)
    rrate_x5 = np.zeros(length)
    rrate_x6 = np.zeros(length)

    pred_ab_x4 = np.zeros(length)
    pred_ab_x5 = np.zeros(length)
    pred_ab_x6 = np.zeros(length)
    rrate_ab_x4 = np.zeros(length)
    rrate_ab_x5 = np.zeros(length)
    rrate_ab_x6 = np.zeros(length)

    pred_t_x4 = np.zeros(length)
    pred_t_x5 = np.zeros(length)
    pred_t_x6 = np.zeros(length)
    rrate_t_x4 = np.zeros(length)
    rrate_t_x5 = np.zeros(length)
    rrate_t_x6 = np.zeros(length)

    for al in [0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2]:
        for bl in [4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5]:

            df_W_out_4 = pd.read_csv('../../results/cusp/library4/{}/cusp_W_out_{}_{}.csv'.format(tl,al,bl))
            df_W_out_5 = pd.read_csv('../../results/cusp/library5/{}/cusp_W_out_{}_{}.csv'.format(tl,al,bl))
            df_W_out_6 = pd.read_csv('../../results/cusp/library6/{}/cusp_W_out_{}_{}.csv'.format(tl,al,bl))

            df_W_out_ab_4 = pd.read_csv('../../results/cusp/library4/{}/cusp_W_out_ab_{}_{}.csv'.format(tl,al,bl))
            df_W_out_ab_5 = pd.read_csv('../../results/cusp/library5/{}/cusp_W_out_ab_{}_{}.csv'.format(tl,al,bl))
            df_W_out_ab_6 = pd.read_csv('../../results/cusp/library6/{}/cusp_W_out_ab_{}_{}.csv'.format(tl,al,bl))

            df_W_out_t_4 = pd.read_csv('../../results/cusp/library4/{}/cusp_W_out_t_{}_{}.csv'.format(tl,al,bl))
            df_W_out_t_5 = pd.read_csv('../../results/cusp/library5/{}/cusp_W_out_t_{}_{}.csv'.format(tl,al,bl))
            df_W_out_t_6 = pd.read_csv('../../results/cusp/library6/{}/cusp_W_out_t_{}_{}.csv'.format(tl,al,bl))

            df_xab = pd.read_csv('../../cusp/cusp_data/cusp_data_{}_{}.csv'.format(al,bl))

            one = np.ones(length)

            initial_theta_4 = df_W_out_4['initial_theta'].to_numpy()[0]
            delta_theta_4 = df_W_out_4['delta_theta'].to_numpy()[0]
            initial_theta_5 = df_W_out_5['initial_theta'].to_numpy()[0]
            delta_theta_5 = df_W_out_5['delta_theta'].to_numpy()[0]
            initial_theta_6 = df_W_out_6['initial_theta'].to_numpy()[0]
            delta_theta_6 = df_W_out_6['delta_theta'].to_numpy()[0]

            a = df_xab['a'].to_numpy()[tl:]
            b = df_xab['b'].to_numpy()[tl:]
            x = df_xab['x'].to_numpy()[tl:]

            x0 = x[0]
            pred_x4[0] = x0
            pred_x5[0] = x0
            pred_x6[0] = x0

            dt = 0.01
            t = np.linspace(0 + tl * dt, 0 + (total_length - 1) * dt, length)

            theta_4 = np.linspace(initial_theta_4+tl*delta_theta_4, initial_theta_4+(total_length-1)*delta_theta_4, length)
            theta_5 = np.linspace(initial_theta_5+tl*delta_theta_5, initial_theta_5+(total_length-1)*delta_theta_5, length)
            theta_6 = np.linspace(initial_theta_6+tl*delta_theta_6, initial_theta_6+(total_length-1)*delta_theta_6, length)

            gt_coef_matrix = np.zeros((7,length))

            coef_matrix_4 = np.zeros((7,length))
            coef_matrix_5 = np.zeros((7,length))
            coef_matrix_6 = np.zeros((7,length))

            coef_matrix_ab_4 = np.zeros((7,length))
            coef_matrix_ab_5 = np.zeros((7,length))
            coef_matrix_ab_6 = np.zeros((7,length))

            coef_matrix_t_4 = np.zeros((7,length))
            coef_matrix_t_5 = np.zeros((7,length))
            coef_matrix_t_6 = np.zeros((7,length))

            gt_coef_matrix[0,:] = a
            gt_coef_matrix[1,:] = b
            gt_coef_matrix[3,:] = -1*one

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

            if smape_4 > 0.01:
                for i in range(length-1):
                    pred_x4[i+1] = pred_x4[i] + fun4(pred_x4[i],coef_cte_4[i],coef_x_4[i],coef_x2_4[i],coef_x3_4[i],coef_x4_4[i])*dt
                    rrate_x4[i+1] = recov_fun4(pred_x4[i+1],coef_x_4[i+1],coef_x2_4[i+1],coef_x3_4[i+1],coef_x4_4[i+1])

                data_4 = {'Time': t[1:], 'traj': x[1:], 'pred': pred_x4[1:], 'rrate': rrate_x4[1:]}
                df_data_4 = pd.DataFrame(data_4)

                df_data_4.to_csv('../../results/cusp/library4/prediction/cusp_pred_{}_{}.csv'.format(al,bl),index=False)

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

            if smape_5 > 0.01:
                for i in range(length-1):
                    pred_x5[i+1] = pred_x5[i] + fun5(pred_x5[i],coef_cte_5[i],coef_x_5[i],coef_x2_5[i],coef_x3_5[i],coef_x4_5[i],coef_x5_5[i])*dt
                    rrate_x5[i+1] = recov_fun5(pred_x5[i+1],coef_x_5[i+1],coef_x2_5[i+1],coef_x3_5[i+1],coef_x4_5[i+1],coef_x5_5[i+1])

                data_5 = {'Time': t[1:], 'traj': x[1:], 'pred': pred_x5[1:], 'rrate': rrate_x5[1:]}
                df_data_5 = pd.DataFrame(data_5)

                df_data_5.to_csv('../../results/cusp/library5/prediction/cusp_pred_{}_{}.csv'.format(al,bl),index=False)

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

            if smape_6 > 0.01:
                for i in range(length-1):
                    pred_x6[i+1] = pred_x6[i] + fun6(pred_x6[i],coef_cte_6[i],coef_x_6[i],coef_x2_6[i],coef_x3_6[i],coef_x4_6[i],coef_x5_6[i],coef_x6_6[i])*dt
                    rrate_x6[i+1] = recov_fun6(pred_x6[i+1],coef_x_6[i+1],coef_x2_6[i+1],coef_x3_6[i+1],coef_x4_6[i+1],coef_x5_6[i+1],coef_x6_6[i+1])

                data_6 = {'Time': t[1:], 'traj': x[1:], 'pred': pred_x6[1:], 'rrate': rrate_x6[1:]}
                df_data_6 = pd.DataFrame(data_6)

                df_data_6.to_csv('../../results/cusp/library6/prediction/cusp_pred_{}_{}.csv'.format(al,bl),index=False)

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

            if smape_ab_4 > 0.01:
                for i in range(length-1):
                    pred_ab_x4[i+1] = pred_ab_x4[i] + fun4(pred_ab_x4[i],coef_cte_ab_4[i],coef_x_ab_4[i],coef_x2_ab_4[i],coef_x3_ab_4[i],coef_x4_ab_4[i])*dt
                    rrate_ab_x4[i+1] = recov_fun4(pred_ab_x4[i+1],coef_x_ab_4[i+1],coef_x2_ab_4[i+1],coef_x3_ab_4[i+1],coef_x4_ab_4[i+1])

                data_ab_4 = {'Time': t[1:], 'traj': x[1:], 'pred': pred_ab_x4[1:], 'rrate': rrate_ab_x4[1:]}
                df_data_ab_4 = pd.DataFrame(data_ab_4)

                df_data_ab_4.to_csv('../../results/cusp/library4/prediction/cusp_pred_ab_{}_{}.csv'.format(al,bl),index=False)

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
    
            if smape_ab_5 > 0.01:
                for i in range(length-1):
                    pred_ab_x5[i+1] = pred_ab_x5[i] + fun5(pred_ab_x5[i],coef_cte_ab_5[i],coef_x_ab_5[i],coef_x2_ab_5[i],coef_x3_ab_5[i],coef_x4_ab_5[i],coef_x5_ab_5[i])*dt
                    rrate_ab_x5[i+1] = recov_fun5(pred_ab_x5[i+1],coef_x_ab_5[i+1],coef_x2_ab_5[i+1],coef_x3_ab_5[i+1],coef_x4_ab_5[i+1],coef_x5_ab_5[i+1])

                data_ab_5 = {'Time': t[1:], 'traj': x[1:], 'pred': pred_ab_x5[1:], 'rrate': rrate_ab_x5[1:]}
                df_data_ab_5 = pd.DataFrame(data_ab_5)

                df_data_ab_5.to_csv('../../results/cusp/library5/prediction/cusp_pred_ab_{}_{}.csv'.format(al,bl),index=False)

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

            if smape_ab_6 > 0.01:
                for i in range(length-1):
                    pred_ab_x6[i+1] = pred_ab_x6[i] + fun6(pred_ab_x6[i],coef_cte_ab_6[i],coef_x_ab_6[i],coef_x2_ab_6[i],coef_x3_ab_6[i],coef_x4_ab_6[i],coef_x5_ab_6[i],coef_x6_ab_6[i])*dt
                    rrate_ab_x6[i+1] = recov_fun6(pred_ab_x6[i+1],coef_x_ab_6[i+1],coef_x2_ab_6[i+1],coef_x3_ab_6[i+1],coef_x4_ab_6[i+1],coef_x5_ab_6[i+1],coef_x6_ab_6[i+1])

                data_ab_6 = {'Time': t[1:], 'traj': x[1:], 'pred': pred_ab_x6[1:], 'rrate': rrate_ab_x6[1:]}
                df_data_ab_6 = pd.DataFrame(data_ab_6)

                df_data_ab_6.to_csv('../../results/cusp/library6/prediction/cusp_pred_ab_{}_{}.csv'.format(al,bl),index=False)

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

            if smape_t_4 > 0.01:
                for i in range(length-1):
                    pred_t_x4[i+1] = pred_t_x4[i] + fun4(pred_t_x4[i],coef_cte_t_4[i],coef_x_t_4[i],coef_x2_t_4[i],coef_x3_t_4[i],coef_x4_t_4[i])*dt
                    rrate_t_x4[i+1] = recov_fun4(pred_t_x4[i+1],coef_x_t_4[i+1],coef_x2_t_4[i+1],coef_x3_t_4[i+1],coef_x4_t_4[i+1])

                data_t_4 = {'Time': t[1:], 'traj': x[1:], 'pred': pred_t_x4[1:], 'rrate': rrate_t_x4[1:]}
                df_data_t_4 = pd.DataFrame(data_t_4)

                df_data_t_4.to_csv('../../results/cusp/library4/prediction/cusp_pred_t_{}_{}.csv'.format(al,bl),index=False)

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

            if smape_t_5 > 0.01:
                for i in range(length-1):
                    pred_t_x5[i+1] = pred_t_x5[i] + fun5(pred_t_x5[i],coef_cte_t_5[i],coef_x_t_5[i],coef_x2_t_5[i],coef_x3_t_5[i],coef_x4_t_5[i],coef_x5_t_5[i])*dt
                    rrate_t_x5[i+1] = recov_fun5(pred_t_x5[i+1],coef_x_t_5[i+1],coef_x2_t_5[i+1],coef_x3_t_5[i+1],coef_x4_t_5[i+1],coef_x5_t_5[i+1])

                data_t_5 = {'Time': t[1:], 'traj': x[1:], 'pred': pred_t_x5[1:], 'rrate': rrate_t_x5[1:]}
                df_data_t_5 = pd.DataFrame(data_t_5)

                df_data_t_5.to_csv('../../results/cusp/library5/prediction/cusp_pred_t_{}_{}.csv'.format(al,bl),index=False)

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

            if smape_t_6 > 0.01:
                for i in range(length-1):
                    pred_t_x6[i+1] = pred_t_x6[i] + fun6(pred_t_x6[i],coef_cte_t_6[i],coef_x_t_6[i],coef_x2_t_6[i],coef_x3_t_6[i],coef_x4_t_6[i],coef_x5_t_6[i],coef_x6_t_6[i])*dt
                    rrate_t_x6[i+1] = recov_fun6(pred_t_x6[i+1],coef_x_t_6[i+1],coef_x2_t_6[i+1],coef_x3_t_6[i+1],coef_x4_t_6[i+1],coef_x5_t_6[i+1],coef_x6_t_6[i+1])

                data_t_6 = {'Time': t[1:], 'traj': x[1:], 'pred': pred_t_x6[1:], 'rrate': rrate_t_x6[1:]}
                df_data_t_6 = pd.DataFrame(data_t_6)

                df_data_t_6.to_csv('../../results/cusp/library6/prediction/cusp_pred_t_{}_{}.csv'.format(al,bl),index=False)
