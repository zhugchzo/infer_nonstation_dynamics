import numpy as np
import pandas as pd

df_tseries = pd.read_csv('../../mitochondria/mitochondria_data.csv')
keep_col_tseries = ['Relative fluorescence ratio','Time']
new_f_tseries = df_tseries[keep_col_tseries]
values_tseries = new_f_tseries.values

x_tseries = np.array(values_tseries[:,0])
t_series = np.array(values_tseries[:,1])

dt = 0.011111111

length = len(x_tseries)

train_length = 181
test_length = length - train_length - 1

train_x_tseries = x_tseries[:train_length]

# choose cte = 1 to include constant term, cte = 0 to exclude it
cte = 1
# input dimension of x
d = 1
# size of linear part of feature vector (x1,...,xn,b)
dlin = d + 1
# size of nonlinear part of feature vector
dnonlin_2 = int(((dlin + 1)*dlin/2))
dnonlin_3 = int(((dlin + 2)*(dlin + 1)*dlin/6))
dnonlin = dnonlin_2 + dnonlin_3
# total size of feature vector: linear + nonlinear
dtot = dlin + dnonlin
        
x_tseries = x_tseries.reshape(-1, 1)
t_series = t_series.reshape(-1, 1)

xt_tseries = np.hstack((x_tseries, t_series))

# create an array to hold the linear part of the feature vector
x_train = np.zeros((dlin,train_length + 1))

# fill in the linear part of the feature vector for all times
for j in range(train_length + 1):
    x_train[:,j] = xt_tseries[j]

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
# compute the pseudo-inverse matrix error

pinv_error_matrix = np.linalg.pinv(out_train[:,:] @ out_train[:,:].T) @ (out_train[:,:] @ out_train[:,:].T) - np.identity(out_train.shape[0])
pinv_error = np.sum(pinv_error_matrix**2)# / out_train.shape[0]**2

##
## SINDy
##

# Initialize the regression coefficient matrix W_out
W_out_sparse = (x_train[0:d,1:train_length + 1] - x_train[0:d,:train_length])/dt @ out_train[:, :].T @ np.linalg.pinv(out_train[:, :] @ out_train[:, :].T)  # Initial least-squares solution

# Set the sparsification parameter lambda
lambda_param = 1e3  # Adjustable parameter, modify according to your data

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

##
## Ridge
##

# ridge parameter for regression
ridge_param = 1e-5

# ridge regression: train W_out to map out_train to x[t+1] - x[t]
W_out_ridge = (x_train[0:d,1:train_length + 1] - x_train[0:d,:train_length])/dt @ out_train[:,:].T @ np.linalg.pinv(out_train[:,:] @ out_train[:,:].T + ridge_param*np.identity(dtot+ cte))

#####################################################################################

# generation, compute MSE, SINDy

# create a place to store feature vectors for prediction
out_gen_sparse = np.ones(dtot + cte)        # full feature vector
x_gen_sparse = np.zeros((dlin,train_length))      # linear part

x_gen_sparse[-1,:] = xt_tseries[:train_length,-1]

# copy over initial linear feature vector
x_gen_sparse[:,0] = xt_tseries[0,:]

gen_traj_sparse = np.zeros((d,train_length))
gen_traj_sparse[:,0] = x_gen_sparse[0:d,0]

for j in range(train_length-1):

    # copy linear part into whole feature vector
    out_gen_sparse[cte:dlin + cte] = x_gen_sparse[:,j] # shift by one for constant

    # fill in the non-linear part

    cnt = 0
    # 2-order
    for row1 in range(dlin):
        for row2 in range(row1,dlin):
            # shift by one for constant
            out_gen_sparse[dlin + cnt + cte] = x_gen_sparse[row1,j] * x_gen_sparse[row2,j]
            cnt += 1

    # 3-order
    for row1 in range(dlin):
        for row2 in range(row1,dlin):
            for row3 in range(row2,dlin):
                # shift by one for constant
                out_gen_sparse[dlin + cnt + cte] = x_gen_sparse[row1,j] * x_gen_sparse[row2,j] * x_gen_sparse[row3,j]
                cnt += 1

    # do a prediction
    x_gen_sparse[0:d,j+1] = x_gen_sparse[0:d,j] + (W_out_sparse @ out_gen_sparse[:])*dt

    gen_traj_sparse[:,j+1] = x_gen_sparse[0:d,j+1]

mse_sparse = (train_x_tseries - gen_traj_sparse.T)**2
sparse_MSE = np.mean(mse_sparse)

# confirm virtual forcing parameter in governing equations
p_indexes = [2,4,5,7,8,9]
if np.sum(W_out_sparse[0][p_indexes]) == 0:
    sparse_MSE = np.nan

# generation, compute MSE, Ridge

# create a place to store feature vectors for prediction
out_gen_ridge = np.ones(dtot + cte)        # full feature vector
x_gen_ridge = np.zeros((dlin,train_length))      # linear part

x_gen_ridge[-1,:] = xt_tseries[:train_length,-1]

# copy over initial linear feature vector
x_gen_ridge[:,0] = xt_tseries[0,:]

gen_traj_ridge = np.zeros((d,train_length))
gen_traj_ridge[:,0] = x_gen_ridge[0:d,0]

for j in range(train_length-1):

    # copy linear part into whole feature vector
    out_gen_ridge[cte:dlin + cte] = x_gen_ridge[:,j] # shift by one for constant

    # fill in the non-linear part

    cnt = 0
    # 2-order
    for row1 in range(dlin):
        for row2 in range(row1,dlin):
            # shift by one for constant
            out_gen_ridge[dlin + cnt + cte] = x_gen_ridge[row1,j] * x_gen_ridge[row2,j]
            cnt += 1

    # 3-order
    for row1 in range(dlin):
        for row2 in range(row1,dlin):
            for row3 in range(row2,dlin):
                # shift by one for constant
                out_gen_ridge[dlin + cnt + cte] = x_gen_ridge[row1,j] * x_gen_ridge[row2,j] * x_gen_ridge[row3,j]
                cnt += 1

    # do a prediction
    x_gen_ridge[0:d,j+1] = x_gen_ridge[0:d,j] + (W_out_ridge @ out_gen_ridge[:])*dt

    gen_traj_ridge[:,j+1] = x_gen_ridge[0:d,j+1]

mse_ridge = (train_x_tseries - gen_traj_ridge.T)**2
ridge_MSE = np.mean(mse_ridge)

#####################################################################################

if sparse_MSE < ridge_MSE:
    MSE = sparse_MSE
else:
    MSE = ridge_MSE

print('time MSE: {}'.format(MSE))
print('time pinv error: {}'.format(pinv_error))