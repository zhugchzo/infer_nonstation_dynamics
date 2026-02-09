# import python libraries
import numpy as np
import pandas as pd
import os

if not os.path.exists('cusp_sin_data'):
    os.makedirs('cusp_sin_data')

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

# Model parameters
# ah = 3
# bh = 3

for al in [0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2]:
    for bl in [4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5]:

        sim_len = 10

        x0 = -1

        t = np.arange(t0,sim_len,dt)
        sint = np.sin(t)

        x = np.zeros(len(t))

        rrate = np.zeros(len(t))

        a = al + sint
        b = bl - sint

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

        df_data.to_csv('cusp_sin_data/cusp_sin_data_{}_{}.csv'.format(al,bl),index=False)