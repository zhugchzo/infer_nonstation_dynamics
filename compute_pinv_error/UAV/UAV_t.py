import numpy as np
import pandas as pd

# Load the saved data
data_file = pd.read_csv('../../UAV/UAV_data.csv')
data = data_file[['Time','x','y']].values

# Extract time and positions
t_series = data[:, 0]
x_tseries = data[:,1:3]

dt = 0.038

length = len(t_series)

train_length = 160
test_length = length - train_length - 1

train_x_tseries = x_tseries[:train_length]

# choose cte = 1 to include constant term, cte = 0 to exclude it
cte = 1
# input dimension of x
d = 2
# input dimension of theta
d_theta = 1
# size of linear part of feature vector (x1,...,xn,b)
dlin = d + d_theta
# size of nonlinear part of feature vector
dnonlin_2 = int(((dlin + 1)*dlin/2))
dnonlin_3 = int(((dlin + 2)*(dlin + 1)*dlin/6))
dnonlin = dnonlin_2 + dnonlin_3
# total size of feature vector: linear + nonlinear
dtot = dlin + dnonlin

f1_elements = ['cte_f1', 'x_f1', 'y_f1', 'p_f1', 'x^2_f1', 'xy_f1', 'px_f1', 'y^2_f1', 'py_f1', 'p^2_f1', 'x^3_f1', 'x^2y_f1', 'px^2_f1', 'xy^2_f1', 'pxy_f1', 'p^2x_f1', 'y^3_f1', 'py^2_f1', 'p^2y_f1', 'p^3_f1']
f2_elements = ['cte_f2', 'x_f2', 'y_f2', 'p_f2', 'x^2_f2', 'xy_f2', 'px_f2', 'y^2_f2', 'py_f2', 'p^2_f2', 'x^3_f2', 'x^2y_f2', 'px^2_f2', 'xy^2_f2', 'pxy_f2', 'p^2x_f2', 'y^3_f2', 'py^2_f2', 'p^2y_f2', 'p^3_f2']

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

# fill in the non-linear part, order = 2. x^2, xy, bx, y^2, by, b^2
cnt = 0
for row1 in range(dlin):
    for row2 in range(row1,dlin):
        # shift by one for constant if needed
        out_train[dlin + cnt + cte] = x_train[row1,:train_length] * x_train[row2,:train_length]
        cnt += 1

# fill in the non-linear part, order = 3. x^3, x^2y, bx^2, xy^2, bxy, b^2x, y^3, by^2, b^2y, b^3
for row1 in range(dlin):
    for row2 in range(row1,dlin):
        for row3 in range(row2,dlin):
            out_train[dlin + cnt + cte] = x_train[row1,:train_length] * x_train[row2,:train_length] * x_train[row3,:train_length]
            cnt += 1

ridge_param = 1e-5

# compute the pseudo-inverse matrix error

pinv_error_matrix = np.linalg.pinv(out_train[:,:] @ out_train[:,:].T + ridge_param*np.identity(dtot+ cte)) @ (out_train[:,:] @ out_train[:,:].T + ridge_param*np.identity(dtot+ cte)) - np.identity(out_train.shape[0])
pinv_error = np.sum(pinv_error_matrix**2) / out_train.shape[0]

##
## SINDy
##

# Initialize the regression coefficient matrix W_out
W_out_sparse = (x_train[0:d,1:train_length + 1] - x_train[0:d,:train_length])/dt @ out_train[:, :].T @ np.linalg.pinv(out_train[:, :] @ out_train[:, :].T)  # Initial least-squares solution

# Set the sparsification parameter lambda
lambda_param = 1e-2  # Adjustable parameter, modify according to your data

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

column_sums_sparse = np.sum(mse_sparse, axis=1)
sparse_MSE = np.mean(column_sums_sparse)

# confirm virtual forcing parameter in governing equations
p_indexes = [i for i, item in enumerate(f1_elements) if 'p' in item]
if np.sum(W_out_sparse[0,:][p_indexes]) + np.sum(W_out_sparse[1,:][p_indexes]) == 0:
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

column_sums_ridge = np.sum(mse_ridge, axis=1)
ridge_MSE = np.mean(column_sums_ridge)

if sparse_MSE < ridge_MSE:
    MSE = sparse_MSE
else:
    MSE = ridge_MSE

print('time MSE: {}'.format(MSE))
print('time pinv error: {}'.format(pinv_error))