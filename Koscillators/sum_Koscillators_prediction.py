import pandas as pd

all_data = []

for rand_seed in range(100):

    df_tseries = pd.read_csv('../results/Koscillators/robust/Koscillators_pred_{}.csv'.format(rand_seed))
    df_network = pd.read_csv('Koscillators_data/Koscillators_network_{}.csv'.format(rand_seed),header=None)

    data_network = df_network.values

    # the number of node
    N = len(data_network)

    col_traj = []
    col_pred = []

    for node in range(N):
        col_traj.append('trajx_{}'.format(node))
        col_pred.append('predx_{}'.format(node))

    df_traj = df_tseries[col_traj]
    df_pred = df_tseries[col_pred]

    traj_long = df_traj.stack().reset_index(drop=True)
    pred_long = df_pred.stack().reset_index(drop=True)

    df_traj_pred = pd.DataFrame({'traj': traj_long.values, 'pred': pred_long.values})

    all_data.append(df_traj_pred)

merged_df = pd.concat(all_data, ignore_index=True)

sample_frac = 0.02
sampled_df = merged_df.sample(frac=sample_frac, random_state=42)

# 输出抽样后的结果
sampled_df.to_csv('../results/Koscillators/Koscillators_traj_pred.csv', index=False)


