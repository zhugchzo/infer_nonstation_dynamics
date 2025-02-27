import numpy as np
import pandas as pd
dt = 0.01

# choose cte = 1 to include constant term
cte = 1
# input dimension of x
d = 1
# input dimension of theta
d_theta = 1
# input dimension of interaction (sum_sin(xj-xi))
dcop = 1

# size of linear part of feature vector (x1,...,xn,b)
dlin = d + d_theta + dcop
# size of nonlinear part of feature vector
dnonlin_2 = int(((dlin + 1)*dlin/2))
dnonlin = dnonlin_2
# total size of feature vector: linear + nonlinear
dtot = dlin + dnonlin

f_library = ['cte', 'x', 'p', 'sum_sin(xj-xi)', 'x^2', 'px', 'xsum_sin(xj-xi)', 'p^2', 'psum_sin(xj-xi)', 'sum_sin^2(xj-xi)']

col_x = ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10']

# the number of node
N = len(col_x)

pinv_error_list = list()

for rand_seed in range(100):

    # Load the saved data
    df_tseries = pd.read_csv('../../Koscillators/robust/Koscillators_data/Koscillators_data_{}.csv'.format(rand_seed))
    df_network = pd.read_csv('../../Koscillators/robust/Koscillators_data/Koscillators_network_{}.csv'.format(rand_seed),header=None)

    data_network = df_network.values

    data_tseries = df_tseries[col_x].values
    t_series = df_tseries['t'].values

    length = len(data_tseries)

    time = np.arange(1, 1 + length, 1)

    train_length = 300
    test_length = length - train_length - 1

    train_x_tseries = np.zeros((N,train_length))

    for node in range(N):
        train_x_tseries[node,:] = data_tseries[:train_length,node]

    t_series = t_series.reshape(-1,1)

    xt_tseries = np.hstack((data_tseries, t_series))

    t_train = t_series[:train_length+1]

    # create an array to hold the linear part of the feature vector
    x_train = np.zeros((dlin,train_length*N))
    y_train = np.zeros((d,train_length*N))

    out_train = np.ones((dtot + cte,train_length*N))

    for node in range(N):

        self_var_index = [node] 
        inter_var_index = list(np.where(data_network[:,node] == 1)[0])

        si_var_index = self_var_index + inter_var_index

        node_data = data_tseries[:,si_var_index][:train_length+1]

        # input number of interactive node
        k = len(si_var_index)

        node_data_theta = np.hstack((node_data, t_train))

        # fill in the linear part of the feature vector for all times
        for j in range(train_length):

            x_train[:d,train_length*node+j] = node_data_theta[j][:d]
            x_train[d:d+d_theta,train_length*node+j] = node_data_theta[j][-1]
            x_train[d+d_theta:,train_length*node+j] = np.sum(np.sin(node_data_theta[j][d:k*d]-np.tile(node_data_theta[j][:d],k-1)))
        
        # copy over the linear part (shift over by one to account for constant if needed)
        out_train[cte:dlin + cte, node*train_length : (node+1)*train_length] = x_train[:, node*train_length : (node+1)*train_length]

        y_train[0,train_length*node : train_length*(node+1)] = (node_data_theta[1:train_length+1,0] - node_data_theta[:train_length,0]).T/dt

    # fill in the non-linear part, order = 2
    cnt = 0
    for row1 in range(dlin):
        for row2 in range(row1,dlin):
            # shift by one for constant if needed
            out_train[dlin + cnt + cte] = x_train[row1,:] * x_train[row2,:]
            cnt += 1             
        
    # compute the pseudo-inverse matrix error

    pinv_error_matrix = np.linalg.pinv(out_train[:,:] @ out_train[:,:].T) @ (out_train[:,:] @ out_train[:,:].T) - np.identity(out_train.shape[0])
    pinv_error = np.sum(pinv_error_matrix**2) / out_train.shape[0]**2

    pinv_error_list.append(pinv_error)

mean_pinv_error = np.mean(pinv_error_list)

print('time pinv error: {}'.format(mean_pinv_error))