import numpy as np
import pandas as pd

def f(x,coef):
    return coef[0] + coef[1]*x + coef[2]*x**2 + coef[3]*x**3

al = 0.2
bl = 4

df_coef = pd.read_csv('../results/sMAPE/cusp_sin_smape_{}_{}.csv'.format(al,bl))

true_coef = df_coef[['cte_true','x_true','x2_true','x3_true']].to_numpy()[:500]
infer_coef = df_coef[['cte_infer','x_infer','x2_infer','x3_infer']].to_numpy()[:500]

f_true = np.zeros(len(true_coef))
f_infer = np.zeros(len(true_coef))

for i in range(len(true_coef)):

    random_number = np.random.uniform(-4,-3)

    f_true[i] = f(random_number,true_coef[i])
    f_infer[i] = f(random_number,infer_coef[i])

f_results = {'f_true':f_true,'f_infer':f_infer}
df_f_results = pd.DataFrame(f_results)

df_f_results.to_csv('../results/sMAPE/check_taylor.csv',index=False)