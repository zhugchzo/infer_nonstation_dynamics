import numpy as np
import pandas as pd
import os

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
dnonlin_3 = int(((dlin + 2)*(dlin + 1)*dlin/6))
dnonlin = dnonlin_2 + dnonlin_3
# total size of feature vector: linear + nonlinear
dtot = dlin + dnonlin

f_library = ['cte', 'x', 'p', 'sin', 'x^2', 'px', 'xsin', 'p^2', 'psin', 'sin^2',
             'x^3', 'px^2', 'x^2sin', 'p^2x', 'pxsin', 'xsin^2', 'p^3', 'p^2sin', 'psin^2', 'sin^3']

for tl in [300, 350, 400, 450, 500]:

    if not os.path.exists('../../results/Koscillators/library3/{}'.format(tl)):
        os.makedirs('../../results/Koscillators/library3/{}'.format(tl))

    pinv_error_list = list()

    for rand_seed in range(100):

        # Load the saved data
        df_tseries = pd.read_csv('../Koscillators_data/Koscillators_data_{}.csv'.format(rand_seed))
        df_network = pd.read_csv('../Koscillators_data/Koscillators_network_{}.csv'.format(rand_seed),header=None)

        data_network = df_network.values

        # the number of node
        N = len(data_network)

        col_x = []

        for node in range(N):
            col_x.append('x{}'.format(node))

        data_tseries = df_tseries[col_x].values
        p_series = df_tseries['p'].values

        length = len(data_tseries)

        time = np.arange(1, 1 + length, 1)

        train_length = tl
        test_length = length - train_length - 1

        train_x_tseries = np.zeros((N,train_length))

        for node in range(N):
            train_x_tseries[node,:] = data_tseries[:train_length,node]

        p_series = p_series.reshape(-1,1)

        xp_tseries = np.hstack((data_tseries, p_series))

        p_train = p_series[:train_length+1]

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

            node_data_theta = np.hstack((node_data, p_train))

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

        # fill in the non-linear part, order = 3
        for row1 in range(dlin):
            for row2 in range(row1,dlin):
                for row3 in range(row2,dlin):
                    out_train[dlin + cnt + cte] = x_train[row1,:] * x_train[row2,:] * x_train[row3,:]
                    cnt += 1

        # compute the pseudo-inverse matrix error

        pinv_error_matrix = np.linalg.pinv(out_train[:,:] @ out_train[:,:].T) @ (out_train[:,:] @ out_train[:,:].T) - np.identity(out_train.shape[0])
        pinv_error = np.sum(pinv_error_matrix**2)

        pinv_error_list.append(pinv_error)            
            
        ##
        ## SINDy
        ##

        W_out_sparse = y_train @ out_train[:,:].T @ np.linalg.pinv(out_train[:,:] @ out_train[:,:].T)

        # Set the sparsification parameter lambda
        # Adjustable parameter, modify according to your data
        lambda_param = 5e-4

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

        dic_W_out.to_csv('../../results/Koscillators/library3/{}/Koscillators_W_out_p_{}.csv'.format(tl,rand_seed),header=True,index=False)

    mean_pinv_error = np.mean(pinv_error_list)