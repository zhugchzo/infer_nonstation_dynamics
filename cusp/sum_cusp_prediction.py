import pandas as pd

all_data = []

for al in [0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2]:
    for bl in [4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5]:

        df_tseries = pd.read_csv('../results/cusp/robust/cusp_pred_{}_{}.csv'.format(al,bl))[['traj', 'pred']]
        all_data.append(df_tseries)

merged_df = pd.concat(all_data, ignore_index=True)

sample_frac = 0.3
sampled_df = merged_df.sample(frac=sample_frac, random_state=42)

# 输出抽样后的结果
sampled_df.to_csv('../results/cusp/cusp_traj_pred.csv', index=False)

