import networkx as nx
import numpy as np
import pandas as pd
import os

if not os.path.exists('Koscillators_data'):
    os.makedirs('Koscillators_data')

# Simulation parameters
dt = 0.01
t0 = 0
tburn = 500 # burn-in period

# define dynamics
def de_fun(x, A, w, p, N):

    return w + p*np.sum(A*np.sin(x-x.reshape(N,1)),axis=1)

pl = 0
ph = 1 # bifurcation parameter high
c_rate = 1e-3 # bifurcation parameter change rate

sim_len = int((ph-pl)*dt/c_rate)

for rand_seed in range(100):

    np.random.seed(rand_seed)

    n = np.random.randint(10, 50)

    g = nx.erdos_renyi_graph(n=n, p=0.2, directed=False, seed=int(rand_seed))  # 使用示例参数生成网络

    g.remove_nodes_from(list(nx.isolates(g)))

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

    df_data.to_csv('Koscillators_data/Koscillators_data_{}.csv'.format(rand_seed),index=False)

    np.savetxt('Koscillators_data/Koscillators_network_{}.csv'.format(rand_seed), A, delimiter=',', fmt='%.3f')

















