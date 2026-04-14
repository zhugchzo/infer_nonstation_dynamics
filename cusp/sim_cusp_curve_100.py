# import python libraries
import numpy as np
import pandas as pd
import os

if not os.path.exists('cusp_curve_data'):
    os.makedirs('cusp_curve_data')

# Simulation parameters
dt = 0.01
t0 = 0
tburn = 500 # burn-in period

#----------------------------------
# Simulate model
#----------------------------------

# Model

def fun(x,a,b):
    return a + b*x - x**3

def recov_fun(x,b):
    rrate = b - 3*x**2
    return rrate

def generate_curve_family(n_curves=100, n_points=1000, a_min=1, a_max=2):
    """
    第一族：
        x 单调增加, y 先增后减
        x(t) = 1 + 2t
        y(t) = 4 - t + 4a t(1-t)

    返回：
        curves: shape = (n_curves, n_points, 2)
        a_values: shape = (n_curves,)
    """
    t = np.linspace(0, 1, n_points)
    a_values = np.linspace(a_min, a_max, n_curves)

    curves = []
    for a in a_values:
        x = 1 + 2 * t
        y = 4 - t + 4 * a * t * (1 - t)
        curves.append(np.column_stack([x, y]))

    return np.array(curves), a_values

curves, a_values = generate_curve_family(
    n_curves=100,
    n_points=1000,
    a_min=1,
    a_max=2
)

ii = 0

for curve in curves:

    sim_len = 10

    x0 = -1

    t = np.arange(t0,sim_len,dt)

    x = np.zeros(len(t))

    rrate = np.zeros(len(t))

    a = curve[:,0]
    b = curve[:,1]

    # Run burn-in period on x0
    for i in range(int(tburn/dt)):
        x0 = x0 + fun(x0,a[0],b[0])*dt

    rrate0 = recov_fun(x0,b[0])

    x[0] = x0
    rrate[0] = rrate0

    rrate_record = list()

    # Run simulation
    for i in range(len(t)-1):
        x[i+1] = x[i] + fun(x[i],a[i],b[i])*dt
        rrate[i+1] = recov_fun(x[i+1],b[i+1])

        if rrate[i] < 0 and rrate[i+1] > 0:
            rrate_record.append(i)

    if len(rrate_record) == 0:

        print('No bifurcation!')

    # Store series data in a temporary DataFrame
    data = {'x': x, 'a': a, 'b': b, 'rrate': rrate, 't': t}
    df_data = pd.DataFrame(data)

    df_data['Time'] = range(len(df_data))

    cols = ['Time'] + [col for col in df_data.columns if col != 'Time']

    df_data.to_csv('cusp_curve_data/cusp_curve_data_{}.csv'.format(ii),index=False)

    ii += 1