import numpy as np
import pandas as pd
import os
import scipy.stats as stats

if not os.path.exists('../results/NED'):
    os.makedirs('../results/NED')

def comp_NED1(pred_x,traj_x):

    error = pred_x - traj_x
    Eucli_d = np.sqrt(np.sum(error**2))
    normal_d = np.sqrt(np.sum(pred_x**2) + np.sum(traj_x**2))
    normal_Eucli_d = Eucli_d / normal_d

    NED_results = normal_Eucli_d

    return NED_results

def comp_NED2(pred_x,traj_x):

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

NED_list_ods = list()
NED_list_p = list()
NED_list_t = list()

length = 1000
pred_start = 300
pred_length = length - pred_start - 1

NED_matrix = np.zeros((100,pred_length))
NED_matrix_p = np.zeros((100,pred_length))
NED_matrix_t = np.zeros((100,pred_length))

row_count = 0

pred_col = ['predx_0','predx_1','predx_2','predx_3','predx_4','predx_5','predx_6','predx_7','predx_8','predx_9','predx_10']
traj_col = ['trajx_0','trajx_1','trajx_2','trajx_3','trajx_4','trajx_5','trajx_6','trajx_7','trajx_8','trajx_9','trajx_10']

for rand_seed in range(100):

    df_pred = pd.read_csv('../results/Koscillators/miss_cos-/Koscillators_pred_{}.csv'.format(rand_seed))
    df_pred_p = pd.read_csv('../results/Koscillators/miss_cos-/Koscillators_pred_p_{}.csv'.format(rand_seed))
    df_pred_t = pd.read_csv('../results/Koscillators/miss_cos-/Koscillators_pred_t_{}.csv'.format(rand_seed))

    t = np.linspace(0, 0 + (pred_length - 1) * 1, pred_length)

    #####################################################################################

    predx = df_pred[pred_col].to_numpy()
    trajx = df_pred[traj_col].to_numpy()

    NED1 = comp_NED1(predx,trajx)
    NED2 = comp_NED2(predx,trajx)

    #####################################################################################

    predx_p = df_pred_p[pred_col].to_numpy()
    trajx_p = df_pred_p[traj_col].to_numpy()

    NED1_p = comp_NED1(predx_p,trajx_p)
    NED2_p = comp_NED2(predx_p,trajx_p)

    #####################################################################################

    predx_t = df_pred_t[pred_col].to_numpy()
    trajx_t = df_pred_t[traj_col].to_numpy()

    NED1_t = comp_NED1(predx_t,trajx_t)
    NED2_t = comp_NED2(predx_t,trajx_t)

    NED_list_ods.append(NED1)
    NED_list_p.append(NED1_p)
    NED_list_t.append(NED1_t)

    NED_matrix[row_count,:] = NED2
    NED_matrix_p[row_count,:] = NED2_p
    NED_matrix_t[row_count,:] = NED2_t

    row_count += 1

NED_list_ods = np.array(NED_list_ods)
NED_list_p = np.array(NED_list_p)
NED_list_t = np.array(NED_list_t)

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

Q1_NED_p = np.percentile(NED_list_p, 25)
Q3_NED_p = np.percentile(NED_list_p, 75)
IQR_NED_p = Q3_NED_p - Q1_NED_p
median_NED_p = np.median(NED_list_p)

lower_bound_NED_p = Q1_NED_p - 1.5 * IQR_NED_p
upper_bound_NED_p = Q3_NED_p + 1.5 * IQR_NED_p

non_outliers_NED_p = NED_list_p[(NED_list_p >= lower_bound_NED_p) & (NED_list_p <= upper_bound_NED_p)]
whisker_min_NED_p = np.min(non_outliers_NED_p)
whisker_max_NED_p = np.max(non_outliers_NED_p)

outliers_NED_p = NED_list_p[(NED_list_p < lower_bound_NED_p) | (NED_list_p > upper_bound_NED_p)]

print("p NED")
print(f"Q1 (25th percentile)     = {Q1_NED_p}")
print(f"Q3 (75th percentile)     = {Q3_NED_p}")
print(f"IQR (Q3 - Q1)            = {IQR_NED_p}")
print(f"Median                   = {median_NED_p}")
print(f"Lower whisker (min)      = {whisker_min_NED_p}")
print(f"Upper whisker (max)      = {whisker_max_NED_p}")
print(f"Outliers                 = {outliers_NED_p}")

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
NED_out.to_csv('../results/NED/Koscillators_NED_miss_cos-.csv',header = True, index=False)