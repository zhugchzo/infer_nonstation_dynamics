import networkx as nx
import numpy as np
import pandas as pd

# Simulation parameters
dt = 0.01
t0 = 0
tburn = 500 # burn-in period

n = 11
md = 5

# define dynamics
def de_fun(x, A, w, p, N):

    return w + p*np.sum(A*np.sin(x-x.reshape(N,1)),axis=1)

rand_seed = 23

pl = 0
ph = 1 # bifurcation parameter high
c_rate = 1e-3 # bifurcation parameter change rate

sim_len = int((ph-pl)*dt/c_rate)

np.random.seed(rand_seed)

g = nx.erdos_renyi_graph(n=n, p=md/(n-1), directed=False, seed=int(rand_seed))  # 使用示例参数生成网络

N = len(g)

A = nx.to_numpy_array(g) #element: out_link

x0 = np.random.rand(N) * 0.05*np.pi  #init
w_initial = np.random.rand(N) * 0.05*np.pi
sum_w = np.sum(w_initial)
w = w_initial - sum_w/N

t = np.arange(t0,sim_len,dt)

x = np.zeros((N,len(t)))

p = pd.Series(np.linspace(pl,ph,len(t)),index=t)

# Run burn-in period on x0
for i in range(int(tburn/dt)):
    x0 = x0 + de_fun(x0, A, w, p[0], N)*dt

# Initial condition post burn-in period
x[:,0] = x0

# Run simulation
for i in range(len(t)-1):
    x[:,i+1] = x[:,i] + de_fun(x[:,i], A, w, p.iloc[i], N)*dt

# Store series data in a temporary DataFrame
data = {'t':t,'p':p}
for node in range(N):
    data['x{}'.format(node)] = x[node,:]

df_data = pd.DataFrame(data)

#------------------------------------
# Export data 
#-----------------------------------

df_data.to_csv('Koscillators_data.csv',index=False)

np.savetxt('Koscillators_network.csv', A, delimiter=',', fmt='%.3f')

















