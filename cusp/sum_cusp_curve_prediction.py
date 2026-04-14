import pandas as pd

all_data = []

for curve in range(100):

    df_tseries = pd.read_csv('../results/cusp/cusp_curve/cusp_curve_pred_{}.csv'.format(curve))[['traj', 'pred']]

    # 如果'pred'列有NaN，就跳过这次循环，不添加到all_data
    if df_tseries['pred'].isna().any():
        continue

    # 如果没有NaN值，才添加到all_data
    all_data.append(df_tseries)

merged_df = pd.concat(all_data, ignore_index=True)

sample_frac = 0.3
sampled_df = merged_df.sample(frac=sample_frac, random_state=42)

# 输出抽样后的结果
sampled_df.to_csv('../results/cusp/cusp_curve_traj_pred.csv', index=False)