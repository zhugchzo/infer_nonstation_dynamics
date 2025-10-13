import numpy as np
import pandas as pd
import os

if not os.path.exists('../../results/Koscillators/miss_sin+'):
    os.makedirs('../../results/Koscillators/miss_sin+')

dt = 0.01

# choose cte = 1 to include constant term
cte = 1
# input dimension of x
d = 1
# input dimension of theta
d_theta = 1
# input dimension of interaction (sum_sin(xj+xi))
dcop = 1

# size of linear part of feature vector (x1,...,xn,b)
dlin = d + d_theta + dcop
# size of nonlinear part of feature vector
dnonlin_2 = int(((dlin + 1)*dlin/2))
dnonlin = dnonlin_2
# total size of feature vector: linear + nonlinear
dtot = dlin + dnonlin

f_library = ['cte', 'x', 'p', 'sum_sin(xj+xi)', 'x^2', 'px', 'xsum_sin(xj+xi)', 'p^2', 'psum_sin(xj+xi)', 'sum_sin^2(xj+xi)']

for rand_seed in range(100):

    # Load the saved data
    df_tseries = pd.read_csv('../Koscillators_data/Koscillators_dataN_{}.csv'.format(rand_seed))
    df_network = pd.read_csv('../Koscillators_data/Koscillators_networkN_{}.csv'.format(rand_seed),header=None)

    data_network = df_network.values

    # the number of node
    N = len(data_network)

    col_x = []

    for node in range(N):
        col_x.append('x{}'.format(node))

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
            x_train[d+d_theta:,train_length*node+j] = np.sum(np.sin(node_data_theta[j][d:k*d]+np.tile(node_data_theta[j][:d],k-1)))
        
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
        
    ##
    ## SINDy
    ##

    W_out_sparse = y_train @ out_train[:,:].T @ np.linalg.pinv(out_train[:,:] @ out_train[:,:].T)

    # Set the sparsification parameter lambda
    # Adjustable parameter, modify according to your data
    lambda_param = 1e-2

    # Perform sparsification, iterate multiple times to obtain a sparse solution
    for k in range(10):  # Number of iterations, can be adjusted as needed
        # Find the coefficients smaller than lambda_param and set them to zero
        smallinds = (np.abs(W_out_sparse) < lambda_param)
        W_out_sparse[smallinds] = 0
        
        # For each state dimension, perform least-squares regression again, only keeping the large coefficients
        for ind in range(d):  # Iterate over each state dimension
            biginds = ~smallinds[ind,:]  # Indices of large coefficients
            # Perform regression using only the non-zero coefficients
            W_out_sparse[ind, biginds] = y_train[ind,:] @ out_train[biginds,:].T @ np.linalg.pinv(out_train[biginds,:] @ out_train[biginds,:].T)

    #####################################################################################

    W_out = W_out_sparse[0]

    dic_W_out = pd.DataFrame()

    for j in range(dtot + cte):
        dic_W_out[f_library[j]] = [W_out[j]]

    dic_W_out.to_csv('../../results/Koscillators/miss_sin+/Koscillators_W_out_t_{}.csv'.format(rand_seed),header=True,index=False)

    ##
    ## prediction
    ##

    # create a place to store feature vectors for prediction
    out_test = np.ones((dtot + cte,N))        # full feature vector
    x_test = np.zeros((dlin,(test_length+1)*N))      # linear part

    x_test[d:d+d_theta,:] = np.tile(xt_tseries[train_length:, -1], N)

    test_traj = np.zeros((N,test_length))

    for node in range(N):

        inter_var_index = list(np.where(data_network[:,node] == 1)[0])
        k = len(inter_var_index)
        
        # copy over initial linear feature vector
        x_test[0,(test_length+1)*node] = xt_tseries[0,node]
        x_test[d+d_theta:,(test_length+1)*node] = np.sum(np.sin(xt_tseries[0,inter_var_index]+np.tile(xt_tseries[0,node],k)))

    for j in range(test_length):

        for node in range(N):

            # copy linear part into whole feature vector
            out_test[cte:dlin + cte,node] = x_test[:,(test_length+1)*node+j] # shift by one for constant

            # fill in the non-linear part

            cnt = 0
            # 2-order
            for row1 in range(dlin):
                for row2 in range(row1,dlin):
                    # shift by one for constant
                    out_test[dlin + cnt + cte,node] = x_test[row1,(test_length+1)*node+j] * x_test[row2,(test_length+1)*node+j]
                    cnt += 1

            # do a prediction
            x_test[0,(test_length+1)*node+j+1] = x_test[0,(test_length+1)*node+j] + (W_out @ out_test[:,node])*dt

            test_traj[node,j] = x_test[0,(test_length+1)*node+j+1]

        for node in range(N):

            inter_var_index = np.where(data_network[:,node] == 1)[0]
            k = len(inter_var_index)

            inter_var = x_test[0,(test_length+1)*inter_var_index+j+1]

            x_test[d+d_theta:,(test_length+1)*node+j+1] = np.sum(np.sin(inter_var+np.tile(x_test[0,(test_length+1)*node+j+1],k)))

    dic_pred = {'Time':t_series[train_length + 1:,0]}

    for node in range(N):
        dic_pred['trajx_{}'.format(node)] = data_tseries[train_length + 1:,node]
        dic_pred['predx_{}'.format(node)] = test_traj[node,:]

    pred_out = pd.DataFrame(dic_pred)
    pred_out.to_csv('../../results/Koscillators/miss_sin+/Koscillators_pred_t_{}.csv'.format(rand_seed),header = True, index=False)