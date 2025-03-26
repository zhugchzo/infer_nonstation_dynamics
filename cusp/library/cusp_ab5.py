import numpy as np
import pandas as pd
import os

# choose cte = 1 to include constant term, cte = 0 to exclude it
cte = 1
# input dimension of x
d = 1
# size of linear part of feature vector (x1,...,xn,a,b)
dlin = d + 2
# size of nonlinear part of feature vector
dnonlin_2 = int(((dlin + 1)*dlin/2))
dnonlin_3 = int(((dlin + 2)*(dlin + 1)*dlin/6))
dnonlin_4 = int(((dlin + 3)*(dlin + 2)*(dlin + 1)*dlin/24))
dnonlin_5 = int(((dlin + 4)*(dlin + 3)*(dlin + 2)*(dlin + 1)*dlin/120))
dnonlin = dnonlin_2 + dnonlin_3 + dnonlin_4 + dnonlin_5
# total size of feature vector: linear + nonlinear
dtot = dlin + dnonlin

polynomial = ['cte', 'x', 'a', 'b', 'x^2', 'ax', 'bx', 'a^2', 'ab', 'b^2', 'x^3', 'ax^2', 'bx^2', 'a^2x', 'abx', 'b^2x', 'a^3', 'a^2b', 'ab^2', 'b^3',
              'x^4', 'ax^3', 'bx^3', 'a^2x^2', 'abx^2', 'b^2x^2', 'a^3x', 'a^2bx', 'ab^2x', 'b^3x', 'a^4', 'a^3b', 'a^2b^2', 'ab^3', 'b^4',
              'x^5', 'ax^4', 'bx^4', 'a^2x^3', 'abx^3', 'b^2x^3', 'a^3x^2', 'a^2bx^2', 'ab^2x^2', 'b^3x^2', 'a^4x', 'a^3bx', 'a^2b^2x', 'ab^3x', 'b^4x',
              'a^5', 'a^4b', 'a^3b^2', 'a^2b^3', 'ab^4', 'b^5']

dt = 0.01

for tl in [500, 550, 600, 650, 700]:

    if not os.path.exists('../../results/cusp/library5/{}'.format(tl)):
        os.makedirs('../../results/cusp/library5/{}'.format(tl))

    pinv_error_list = list()

    for al in [0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2]:
        for bl in [4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5]:

            df_tseries = pd.read_csv('../cusp_data/cusp_data_{}_{}.csv'.format(al,bl))
            keep_col_tseries = ['x','a','b']
            new_f_tseries = df_tseries[keep_col_tseries]
            values_tseries = new_f_tseries.values

            x_tseries = np.array(values_tseries[:,0])
            ab_series = np.array(values_tseries[:,1:])

            length = len(x_tseries)

            train_length = tl
            test_length = length - train_length - 1

            train_x_tseries = x_tseries[:train_length]

            x_tseries = x_tseries.reshape(-1,1)
            ab_series = ab_series.reshape(-1,2)

            xab_tseries = np.hstack((x_tseries, ab_series))

            # create an array to hold the linear part of the feature vector
            x_train = np.zeros((dlin,train_length + 1))

            # fill in the linear part of the feature vector for all times
            for j in range(train_length + 1):
                x_train[:,j] = xab_tseries[j]

            # create an array to hold the full feature vector for training time
            # (use ones so the constant term is already 1)
            out_train = np.ones((dtot + cte,train_length))  

            # copy over the linear part (shift over by one to account for constant if needed)
            out_train[cte:dlin + cte,:] = x_train[:,:train_length]

            # fill in the non-linear part, order = 2
            cnt = 0
            for row1 in range(dlin):
                for row2 in range(row1,dlin):
                    # shift by one for constant if needed
                    out_train[dlin + cnt + cte] = x_train[row1,:train_length] * x_train[row2,:train_length]
                    cnt += 1

            # fill in the non-linear part, order = 3
            for row1 in range(dlin):
                for row2 in range(row1,dlin):
                    for row3 in range(row2,dlin):
                        out_train[dlin + cnt + cte] = x_train[row1,:train_length] * x_train[row2,:train_length] * x_train[row3,:train_length]
                        cnt += 1

            # fill in the non-linear part, order = 4
            for row1 in range(dlin):
                for row2 in range(row1,dlin):
                    for row3 in range(row2,dlin):
                        for row4 in range(row3,dlin):
                            out_train[dlin + cnt + cte] = x_train[row1,:train_length] * x_train[row2,:train_length] * x_train[row3,:train_length] * x_train[row4,:train_length]
                            cnt += 1

            # fill in the non-linear part, order = 5
            for row1 in range(dlin):
                for row2 in range(row1,dlin):
                    for row3 in range(row2,dlin):
                        for row4 in range(row3,dlin):
                            for row5 in range(row4,dlin):
                                out_train[dlin + cnt + cte] = x_train[row1,:train_length] * x_train[row2,:train_length] * x_train[row3,:train_length] * x_train[row4,:train_length] * x_train[row5,:train_length]
                                cnt += 1

            # compute the pseudo-inverse matrix error

            pinv_error_matrix = np.linalg.pinv(out_train[:,:] @ out_train[:,:].T) @ (out_train[:,:] @ out_train[:,:].T) - np.identity(out_train.shape[0])
            pinv_error = np.sum(pinv_error_matrix**2)

            pinv_error_list.append(pinv_error)

            ##
            ## SINDy
            ##

            # Initialize the regression coefficient matrix W_out
            W_out_sparse = (x_train[0:d,1:train_length + 1] - x_train[0:d,:train_length])/dt @ out_train[:, :].T @ np.linalg.pinv(out_train[:, :] @ out_train[:, :].T)  # Initial least-squares solution

            # Set the sparsification parameter lambda
            lambda_param = 2.5e-2  # Adjustable parameter, modify according to your data

            # Perform sparsification, iterate multiple times to obtain a sparse solution
            for k in range(10):  # Number of iterations, can be adjusted as needed
                # Find the coefficients smaller than lambda_param and set them to zero
                smallinds = (np.abs(W_out_sparse) < lambda_param)
                W_out_sparse[smallinds] = 0
                
                # For each state dimension, perform least-squares regression again, only keeping the large coefficients
                for ind in range(d):  # Iterate over each state dimension
                    biginds = ~smallinds[ind,:]  # Indices of large coefficients
                    # Perform regression using only the non-zero coefficients
                    W_out_sparse[ind, biginds] = (x_train[ind,1: train_length + 1] - x_train[ind,:train_length])/dt @ out_train[biginds,:].T @ np.linalg.pinv(out_train[biginds,:] @ out_train[biginds,:].T)

            W_out = W_out_sparse[0]

            dic_W_out = pd.DataFrame()

            for j in range(dtot + cte):
                dic_W_out[polynomial[j]] = [W_out[j]]

            dic_W_out.to_csv('../../results/cusp/library5/{}/cusp_W_out_ab_{}_{}.csv'.format(tl,al,bl),header=True,index=False)

    mean_pinv_error = np.mean(pinv_error_list)