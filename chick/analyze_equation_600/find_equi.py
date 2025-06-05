import numpy as np
import pandas as pd
import scipy.integrate as spi

df_data = pd.read_csv('../../results/chick/chick_600_gen.csv')
df_W_out = pd.read_csv('../../results/chick/chick_600_W_out.csv')

states_col = ['traj','theta']
f_elements = ['cte', 'x', 'p', 'x^2', 'px', 'p^2', 'x^3', 'px^2', 'p^2x', 'p^3']

states = df_data[states_col].values
f_coef = df_W_out[f_elements].values

def f_map(x, p, f0):
    polys = np.array([1, x, p, x**2, p*x, p**2, x**3, p*x**2, p**2*x, p**3])
    x_next = np.dot(f0, polys)
    return x_next

s0 = states[0][0]
p0 = states[0][1]
f0 = f_coef[0]

# 离散迭代步数
n_steps = 500
t = np.linspace(0, n_steps, n_steps+1)

# 保存轨道
trajectory = [s0]
x = s0

for _ in range(n_steps):
    x = f_map(x, p0, f0)
    trajectory.append(x)

# 转成 NumPy 数组
trajectory = np.array(trajectory)

# Put into pandas
df_traj = pd.DataFrame(trajectory, index=t, columns=['x'])

# Does the sysetm blow up?
if df_traj.abs().max().max() > 1e3:
    print('System blew up - run new model')

# Does the system contain Nan?
if df_traj.isna().values.any():
    print('System contains Nan value - run new model')
    
# Does the system contain inf?
if np.isinf(df_traj.values).any():
    print('System contains Inf value - run new model')

# Does the system converge?
# Difference between max and min of last 10 data points
diff = df_traj.iloc[-10:-1].max() - df_traj.iloc[-10:-1].min()
# L2 norm
norm = np.sqrt(np.square(diff).sum())
# Define convergence threshold
conv_thresh = 1e-8
if norm > conv_thresh:
    print('System does not converge - run new model')

# Export equilibrim data
equi = df_traj.iloc[-1].values
np.savetxt("equi.csv", equi, delimiter=",")

# Export parameter data
pars = np.concatenate([np.array([p0]),f0])
np.savetxt("pars.csv", pars, delimiter=",")
