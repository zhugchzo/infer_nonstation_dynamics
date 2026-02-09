import numpy as np
import pandas as pd
import os

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

length = 499
start = 500

index_ods = 0
index_ab = 0
index_t = 0

num_ods = 100
num_ab = 100
num_t = 100

for al in [0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2]:
    for bl in [4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5]:
        
        ods_end = 0

        file_path = '../results/cusp/library6/prediction/cusp_pred_{}_{}.csv'.format(al, bl)
        
        if not os.path.exists(file_path):
            index_ods += 1
            num_ods -= 1
            continue

        df_pred = pd.read_csv(file_path)

        #####################################################################################

        predx = df_pred['pred'].to_numpy()
        trajx = df_pred['traj'].to_numpy()
        rrate = df_pred['rrate'].to_numpy()

        #####################################################################################

        for i in range(length-1):

            if rrate[i] < 0 and rrate[i+1] > 0:

                ods_end = i+1

                break

        if ods_end != 0:

            ods_count += 1

            NED = comp_NED(predx[:ods_end-1],trajx[:ods_end-1])

            NED_list_ods.append(NED)
            bif_list.append(ods_end+start)
            bif_index.append(index_ods)

        index_ods += 1  

for al in [0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2]:
    for bl in [4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5]:
        
        ab_end = 0

        file_path_ab = '../results/cusp/library6/prediction/cusp_pred_ab_{}_{}.csv'.format(al, bl)
        
        if not os.path.exists(file_path_ab):
            index_ab += 1
            num_ab -= 1
            continue

        df_pred_ab = pd.read_csv(file_path_ab)

        #####################################################################################

        predx_ab = df_pred_ab['pred'].to_numpy()
        trajx_ab = df_pred_ab['traj'].to_numpy()
        rrate_ab = df_pred_ab['rrate'].to_numpy()

        #####################################################################################

        for i in range(length-1):

            if rrate_ab[i] < 0 and rrate_ab[i+1] > 0:

                ab_end = i+1

                break

        if ab_end != 0:

            ab_count += 1

            NED_ab = comp_NED(predx_ab[:ab_end-1],trajx_ab[:ab_end-1])

            NED_list_ab.append(NED_ab)
            bif_list_ab.append(ab_end+start)
            bif_index_ab.append(index_ab)

        index_ab += 1     

for al in [0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2]:
    for bl in [4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5]:
        
        t_end = 0

        file_path_t = '../results/cusp/library6/prediction/cusp_pred_t_{}_{}.csv'.format(al, bl)
        
        if not os.path.exists(file_path_t):
            index_t += 1
            num_t -= 1
            continue

        df_pred_t = pd.read_csv(file_path_t)

        #####################################################################################

        predx_t = df_pred_t['pred'].to_numpy()
        trajx_t = df_pred_t['traj'].to_numpy()
        rrate_t = df_pred_t['rrate'].to_numpy()

        #####################################################################################

        for i in range(length-1):

            if rrate_t[i] < 0 and rrate_t[i+1] > 0:

                t_end = i+1

                break

        if t_end != 0:

            t_count += 1

            NED_t = comp_NED(predx_t[:t_end-1],trajx_t[:t_end-1])

            NED_list_t.append(NED_t)
            bif_list_t.append(t_end+start)
            bif_index_t.append(index_t)

        index_t += 1

        #####################################################################################

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

print("ods bif ratio")
print(f"bif ratio     = {ods_count/num_ods}")

print("ods surrogate num")
print(f"surrogate num     = {num_ods}")

print('\n')
print('###################################')
print('\n')

if len(NED_list_ab) != 0:

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

    print("ab bif ratio")
    print(f"bif ratio     = {ab_count/num_ab}")

    print("ab surrogate num")
    print(f"surrogate num     = {num_ab}")

    print('\n')
    print('###################################')
    print('\n')

else:

    print('\n')
    print('########## No bifurcation in ab ############')
    print('\n')

    print("ab surrogate num")
    print(f"surrogate num     = {num_ab}")

    print('\n')
    print('###################################')
    print('\n')

if len(NED_list_t) != 0:

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

    print("t bif ratio")
    print(f"bif ratio     = {t_count/num_t}")

    print("t surrogate num")
    print(f"surrogate num     = {num_t}")

else:

    print('\n')
    print('########## No bifurcation in t ############')
    print('\n')    

    print("t surrogate num")
    print(f"surrogate num     = {num_t}")