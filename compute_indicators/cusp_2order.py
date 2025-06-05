import numpy as np
import pandas as pd

# Model

# def fun(x,a,b,c):
#     return a + b*x + c*x**2

def recov_fun(x,b,c):
    rrate = b + 2*c*x
    return rrate

def comp_NED(pred_x,traj_x):

    error = pred_x - traj_x
    Eucli_d = np.sqrt(np.sum(error**2))
    normal_d = np.sqrt(np.sum(pred_x**2) + np.sum(traj_x**2))
    normal_Eucli_d = Eucli_d / normal_d

    NED_results = normal_Eucli_d

    return NED_results

bif_list = list()
bif_list_ab = list()
bif_list_t = list()

bif_index = list()
bif_index_ab = list()
bif_index_t = list()

NED_list_ods = list()
NED_list_ab = list()
NED_list_t = list()

ods_count = 0
ab_count = 0
t_count = 0

length = 1000
pred_start = 500

index = 0

for al in [0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2]:
    for bl in [4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5]:
        
        ods_end = 0
        ab_end = 0
        t_end = 0

        df_W_out = pd.read_csv('../results/cusp/miss/cusp_W_out_{}_{}.csv'.format(al,bl))
        df_W_out_ab = pd.read_csv('../results/cusp/miss/cusp_W_out_ab_{}_{}.csv'.format(al,bl))
        df_W_out_t = pd.read_csv('../results/cusp/miss/cusp_W_out_t_{}_{}.csv'.format(al,bl))

        df_pred = pd.read_csv('../results/cusp/miss/cusp_pred_{}_{}.csv'.format(al,bl))
        df_pred_ab = pd.read_csv('../results/cusp/miss/cusp_pred_ab_{}_{}.csv'.format(al,bl))
        df_pred_t = pd.read_csv('../results/cusp/miss/cusp_pred_t_{}_{}.csv'.format(al,bl))

        df_ab = pd.read_csv('../cusp/cusp_data/cusp_data_{}_{}.csv'.format(al,bl))

        a = df_ab['a'].to_numpy()
        b = df_ab['b'].to_numpy()

        t = np.linspace(0, 0 + (length - 1) * 0.01, length)

        #####################################################################################

        predx = df_pred['pred'].to_numpy()
        trajx = df_pred['traj'].to_numpy()

        #####################################################################################

        predx_ab = df_pred_ab['pred'].to_numpy()
        trajx_ab = df_pred_ab['traj'].to_numpy()

        #####################################################################################

        predx_t = df_pred_t['pred'].to_numpy()
        trajx_t = df_pred_t['traj'].to_numpy()

        #####################################################################################

        one = np.ones(length)

        #####################################################################################

        rrate = np.zeros(length)
        rrate_ab = np.zeros(length)
        rrate_t = np.zeros(length)

        #####################################################################################

        initial_theta = df_W_out['initial_theta'].to_numpy()[0]
        delta_theta = df_W_out['delta_theta'].to_numpy()[0]

        theta = np.linspace(initial_theta, initial_theta + (length - 1) * delta_theta, length)

        W_out_x = df_W_out[['x','px']].to_numpy()[0]
        W_out_x2 = df_W_out[['x^2']].to_numpy()[0]

        coef_x = W_out_x[0]*one + W_out_x[1]*theta
        coef_x2 = W_out_x2[0]*one

        rrate[0] = recov_fun(predx[0],coef_x[pred_start],coef_x2[pred_start])

        # Run simulation
        for i in range(pred_start,length-2):

            rrate[i+1-pred_start] = recov_fun(predx[i+1-pred_start],coef_x[i+1],coef_x2[i+1])

            if rrate[i-pred_start] < 0 and rrate[i+1-pred_start] > 0:

                ods_end = i+1

                break

        if ods_end != 0:

            ods_count += 1

            NED = comp_NED(predx[:ods_end-1-pred_start],trajx[:ods_end-1-pred_start])

            NED_list_ods.append(NED)
            bif_list.append(ods_end)
            bif_index.append(index)

        #####################################################################################

        W_out_x_ab = df_W_out_ab[['x','ax','bx']].to_numpy()[0]
        W_out_x2_ab = df_W_out_ab[['x^2']].to_numpy()[0]

        coef_x_ab = W_out_x_ab[0]*one + W_out_x_ab[1]*a + W_out_x_ab[2]*b
        coef_x2_ab = W_out_x2_ab[0]*one

        rrate_ab[0] = recov_fun(predx_ab[0],coef_x_ab[pred_start],coef_x2_ab[pred_start])

        # Run simulation
        for i in range(pred_start,length-2):

            rrate_ab[i+1-pred_start] = recov_fun(predx_ab[i+1-pred_start],coef_x_ab[i+1],coef_x2_ab[i+1])

            if rrate_ab[i-pred_start] < 0 and rrate_ab[i+1-pred_start] > 0:

                ab_end = i+1

                break

        if ab_end != 0:

            ab_count += 1

            NED_ab = comp_NED(predx_ab[:ab_end-1-pred_start],trajx_ab[:ab_end-1-pred_start])

            NED_list_ab.append(NED_ab)
            bif_list_ab.append(ab_end)
            bif_index_ab.append(index)        

        #####################################################################################

        W_out_x_t = df_W_out_t[['x','px']].to_numpy()[0]
        W_out_x2_t = df_W_out_t[['x^2']].to_numpy()[0]

        coef_x_t = W_out_x_t[0]*one + W_out_x_t[1]*t
        coef_x2_t = W_out_x2_t[0]*one

        rrate_t[0] = recov_fun(predx_t[0],coef_x_t[pred_start],coef_x2_t[pred_start])

        # Run simulation
        for i in range(pred_start,length-2):

            rrate_t[i+1-pred_start] = recov_fun(predx_t[i+1-pred_start],coef_x_t[i+1],coef_x2_t[i+1])

            if rrate_t[i-pred_start] < 0 and rrate_t[i+1-pred_start] > 0:

                t_end = i+1

                break

        if t_end != 0:

            t_count += 1

            NED_t = comp_NED(predx_t[:t_end-1-pred_start],trajx_t[:t_end-1-pred_start])

            NED_list_t.append(NED_t)
            bif_list_t.append(t_end)
            bif_index_t.append(index)

        #####################################################################################

        index += 1

df_bif_gt = pd.read_csv('../cusp/cusp_data/bif.csv')
bif_gt = df_bif_gt['bif_time'].to_numpy()

bif_error_ods = np.abs(bif_gt[bif_index] - np.array(bif_list))
bif_error_ab = np.abs(bif_gt[bif_index_ab] - np.array(bif_list_ab))
bif_error_t = np.abs(bif_gt[bif_index_t] - np.array(bif_list_t))

NED_list_ods = np.array(NED_list_ods)
NED_list_ab = np.array(NED_list_ab)
NED_list_t = np.array(NED_list_t)

#####################################################################################

Q1_bif_error_ods = np.percentile(bif_error_ods, 25)
Q3_bif_error_ods = np.percentile(bif_error_ods, 75)
IQR_bif_error_ods = Q3_bif_error_ods - Q1_bif_error_ods
median_bif_error_ods = np.median(bif_error_ods)

lower_bound_bif_error_ods = Q1_bif_error_ods - 1.5 * IQR_bif_error_ods
upper_bound_bif_error_ods = Q3_bif_error_ods + 1.5 * IQR_bif_error_ods

non_outliers_bif_error_ods = bif_error_ods[(bif_error_ods >= lower_bound_bif_error_ods) & (bif_error_ods <= upper_bound_bif_error_ods)]
whisker_min_bif_error_ods = np.min(non_outliers_bif_error_ods)
whisker_max_bif_error_ods = np.max(non_outliers_bif_error_ods)

outliers_bif_error_ods = bif_error_ods[(bif_error_ods < lower_bound_bif_error_ods) | (bif_error_ods > upper_bound_bif_error_ods)]

print("ods bif error")
print(f"Q1 (25th percentile)     = {Q1_bif_error_ods}")
print(f"Q3 (75th percentile)     = {Q3_bif_error_ods}")
print(f"IQR (Q3 - Q1)            = {IQR_bif_error_ods}")
print(f"Median                   = {median_bif_error_ods}")
print(f"Lower whisker (min)      = {whisker_min_bif_error_ods}")
print(f"Upper whisker (max)      = {whisker_max_bif_error_ods}")
print(f"Outliers                 = {outliers_bif_error_ods}")

print('\n')
#####################################################################################

Q1_NED_ods = np.percentile(NED_list_ods, 25)
Q3_NED_ods = np.percentile(NED_list_ods, 75)
IQR_NED_ods = Q3_NED_ods - Q1_NED_ods
median_NED_ods = np.median(NED_list_ods)

lower_bound_NED_ods = Q1_NED_ods - 1.5 * IQR_NED_ods
upper_bound_NED_ods = Q3_NED_ods + 1.5 * IQR_NED_ods

non_outliers_NED_ods = NED_list_ods[(NED_list_ods >= lower_bound_NED_ods) & (NED_list_ods <= upper_bound_NED_ods)]
whisker_min_NED_ods = np.min(non_outliers_NED_ods)
whisker_max_NED_ods = np.max(non_outliers_NED_ods)

outliers_NED_ods = NED_list_ods[(NED_list_ods < lower_bound_NED_ods) | (NED_list_ods > upper_bound_NED_ods)]

print("ods NED")
print(f"Q1 (25th percentile)     = {Q1_NED_ods}")
print(f"Q3 (75th percentile)     = {Q3_NED_ods}")
print(f"IQR (Q3 - Q1)            = {IQR_NED_ods}")
print(f"Median                   = {median_NED_ods}")
print(f"Lower whisker (min)      = {whisker_min_NED_ods}")
print(f"Upper whisker (max)      = {whisker_max_NED_ods}")
print(f"Outliers                 = {outliers_NED_ods}")

print('\n')
#####################################################################################

print("ods bif count")
print(f"bif count     = {ods_count}")

print('\n')
print('###################################')
print('\n')

Q1_bif_error_ab = np.percentile(bif_error_ab, 25)
Q3_bif_error_ab = np.percentile(bif_error_ab, 75)
IQR_bif_error_ab = Q3_bif_error_ab - Q1_bif_error_ab
median_bif_error_ab = np.median(bif_error_ab)

lower_bound_bif_error_ab = Q1_bif_error_ab - 1.5 * IQR_bif_error_ab
upper_bound_bif_error_ab = Q3_bif_error_ab + 1.5 * IQR_bif_error_ab

non_outliers_bif_error_ab = bif_error_ab[(bif_error_ab >= lower_bound_bif_error_ab) & (bif_error_ab <= upper_bound_bif_error_ab)]
whisker_min_bif_error_ab = np.min(non_outliers_bif_error_ab)
whisker_max_bif_error_ab = np.max(non_outliers_bif_error_ab)

outliers_bif_error_ab = bif_error_ab[(bif_error_ab < lower_bound_bif_error_ab) | (bif_error_ab > upper_bound_bif_error_ab)]

print("ab bif error")
print(f"Q1 (25th percentile)     = {Q1_bif_error_ab}")
print(f"Q3 (75th percentile)     = {Q3_bif_error_ab}")
print(f"IQR (Q3 - Q1)            = {IQR_bif_error_ab}")
print(f"Median                   = {median_bif_error_ab}")
print(f"Lower whisker (min)      = {whisker_min_bif_error_ab}")
print(f"Upper whisker (max)      = {whisker_max_bif_error_ab}")
print(f"Outliers                 = {outliers_bif_error_ab}")

print('\n')
#####################################################################################

Q1_NED_ab = np.percentile(NED_list_ab, 25)
Q3_NED_ab = np.percentile(NED_list_ab, 75)
IQR_NED_ab = Q3_NED_ab - Q1_NED_ab
median_NED_ab = np.median(NED_list_ab)

lower_bound_NED_ab = Q1_NED_ab - 1.5 * IQR_NED_ab
upper_bound_NED_ab = Q3_NED_ab + 1.5 * IQR_NED_ab

non_outliers_NED_ab = NED_list_ab[(NED_list_ab >= lower_bound_NED_ab) & (NED_list_ab <= upper_bound_NED_ab)]
whisker_min_NED_ab = np.min(non_outliers_NED_ab)
whisker_max_NED_ab = np.max(non_outliers_NED_ab)

outliers_NED_ab = NED_list_ab[(NED_list_ab < lower_bound_NED_ab) | (NED_list_ab > upper_bound_NED_ab)]

print("ab NED")
print(f"Q1 (25th percentile)     = {Q1_NED_ab}")
print(f"Q3 (75th percentile)     = {Q3_NED_ab}")
print(f"IQR (Q3 - Q1)            = {IQR_NED_ab}")
print(f"Median                   = {median_NED_ab}")
print(f"Lower whisker (min)      = {whisker_min_NED_ab}")
print(f"Upper whisker (max)      = {whisker_max_NED_ab}")
print(f"Outliers                 = {outliers_NED_ab}")

print('\n')
#####################################################################################

print("ab bif count")
print(f"bif count     = {ab_count}")

print('\n')
print('###################################')
print('\n')

Q1_bif_error_t = np.percentile(bif_error_t, 25)
Q3_bif_error_t = np.percentile(bif_error_t, 75)
IQR_bif_error_t = Q3_bif_error_t - Q1_bif_error_t
median_bif_error_t = np.median(bif_error_t)

lower_bound_bif_error_t = Q1_bif_error_t - 1.5 * IQR_bif_error_t
upper_bound_bif_error_t = Q3_bif_error_t + 1.5 * IQR_bif_error_t

non_outliers_bif_error_t = bif_error_t[(bif_error_t >= lower_bound_bif_error_t) & (bif_error_t <= upper_bound_bif_error_t)]
whisker_min_bif_error_t = np.min(non_outliers_bif_error_t)
whisker_max_bif_error_t = np.max(non_outliers_bif_error_t)

outliers_bif_error_t = bif_error_t[(bif_error_t < lower_bound_bif_error_t) | (bif_error_t > upper_bound_bif_error_t)]

print("t bif error")
print(f"Q1 (25th percentile)     = {Q1_bif_error_t}")
print(f"Q3 (75th percentile)     = {Q3_bif_error_t}")
print(f"IQR (Q3 - Q1)            = {IQR_bif_error_t}")
print(f"Median                   = {median_bif_error_t}")
print(f"Lower whisker (min)      = {whisker_min_bif_error_t}")
print(f"Upper whisker (max)      = {whisker_max_bif_error_t}")
print(f"Outliers                 = {outliers_bif_error_t}")

print('\n')
#####################################################################################

Q1_NED_t = np.percentile(NED_list_t, 25)
Q3_NED_t = np.percentile(NED_list_t, 75)
IQR_NED_t = Q3_NED_t - Q1_NED_t
median_NED_t = np.median(NED_list_t)

lower_bound_NED_t = Q1_NED_t - 1.5 * IQR_NED_t
upper_bound_NED_t = Q3_NED_t + 1.5 * IQR_NED_t

non_outliers_NED_t = NED_list_t[(NED_list_t >= lower_bound_NED_t) & (NED_list_t <= upper_bound_NED_t)]
whisker_min_NED_t = np.min(non_outliers_NED_t)
whisker_max_NED_t = np.max(non_outliers_NED_t)

outliers_NED_t = NED_list_t[(NED_list_t < lower_bound_NED_t) | (NED_list_t > upper_bound_NED_t)]

print("t NED")
print(f"Q1 (25th percentile)     = {Q1_NED_t}")
print(f"Q3 (75th percentile)     = {Q3_NED_t}")
print(f"IQR (Q3 - Q1)            = {IQR_NED_t}")
print(f"Median                   = {median_NED_t}")
print(f"Lower whisker (min)      = {whisker_min_NED_t}")
print(f"Upper whisker (max)      = {whisker_max_NED_t}")
print(f"Outliers                 = {outliers_NED_t}")

print('\n')
#####################################################################################

print("t bif count")
print(f"bif count     = {t_count}")