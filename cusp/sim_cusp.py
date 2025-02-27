# import python libraries
import numpy as np
import pandas as pd

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
al = 1
bl = 4
ah = 3
bh = 3

sim_len = 10

x0 = -1

t = np.arange(t0,sim_len,dt)

x = np.zeros(len(t))

rrate = np.zeros(len(t))

a = pd.Series(np.linspace(al,ah,len(t)),index=t)
b = pd.Series(np.linspace(bl,bh,len(t)),index=t)

# Run burn-in period on x0
for i in range(int(tburn/dt)):
    x0 = x0 + fun(x0,a[0],b[0])*dt

rrate0 = recov_fun(x0,b[0])

x[0] = x0
rrate[0] = rrate0

rrate_record = list()

# Run simulation
for i in range(len(t)-1):
    x[i+1] = x[i] + fun(x[i],a.iloc[i],b.iloc[i])*dt
    rrate[i+1] = recov_fun(x[i+1],b.iloc[i+1])

# Store series data in a temporary DataFrame
data = {'x': x, 'a': a, 'b': b, 'rrate': rrate, 't': t}
df_data = pd.DataFrame(data)

df_data['Time'] = range(len(df_data))

cols = ['Time'] + [col for col in df_data.columns if col != 'Time']

df_data.to_csv('cusp_data.csv',index=False)