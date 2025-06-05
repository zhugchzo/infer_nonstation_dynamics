import numpy as np
import pandas as pd
import os

def comp_egv(x,p,W_out):

    dom_egv = np.dot(np.array([0, 1, 0, 2*x, p, 0, 3*x**2, 2*p*x, p**2, 0]),W_out)

    return dom_egv

def F(x,p,W_out):

    F_next = np.dot(np.array([1, x, p, x**2, p*x, p**2, x**3, p*x**2, p**2*x, p**3]),W_out)

    return F_next

if not os.path.exists('../results/chick'):
    os.makedirs('../results/chick')

df_tseries = pd.read_csv('chick_data_270.csv')
keep_col_tseries = ['IBI (s)','Beat number']
new_f_tseries = df_tseries[keep_col_tseries]
values_tseries = new_f_tseries.values

x_tseries = np.array(values_tseries[:,0])
tseries = np.array(values_tseries[:,1])

length = len(x_tseries)

train_length = 150
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
t_series = tseries.reshape(-1, 1)

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

##
## SINDy
##

# Initialize the regression coefficient matrix W_out
W_out_sparse = x_train[0:d,1:train_length + 1] @ out_train[:, :].T @ np.linalg.pinv(out_train[:, :] @ out_train[:, :].T)  # Initial least-squares solution

# Set the sparsification parameter lambda
lambda_param = 1e1  # Adjustable parameter, modify according to your data

# Perform sparsification, iterate multiple times to obtain a sparse solution
for k in range(10):  # Number of iterations, can be adjusted as needed
    # Find the coefficients smaller than lambda_param and set them to zero
    smallinds = (np.abs(W_out_sparse) < lambda_param)
    W_out_sparse[smallinds] = 0
    
    # For each state dimension, perform least-squares regression again, only keeping the large coefficients
    for ind in range(d):  # Iterate over each state dimension
        biginds = ~smallinds[ind,:]  # Indices of large coefficients
        # Perform regression using only the non-zero coefficients
        W_out_sparse[ind, biginds] = x_train[ind,1: train_length + 1] @ out_train[biginds,:].T @ np.linalg.pinv(out_train[biginds,:] @ out_train[biginds,:].T)

##
## Ridge
##

# ridge parameter for regression
ridge_param = 1e-4

# ridge regression: train W_out to map out_train to x[t+1] - x[t]
W_out_ridge = x_train[0:d,1:train_length + 1] @ out_train[:,:].T @ np.linalg.pinv(out_train[:,:] @ out_train[:,:].T + ridge_param*np.identity(dtot+ cte))


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
    x_gen_sparse[0:d,j+1] = W_out_sparse @ out_gen_sparse[:]

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
    x_gen_ridge[0:d,j+1] = W_out_ridge @ out_gen_ridge[:]

    gen_traj_ridge[:,j+1] = x_gen_ridge[0:d,j+1]

mse_ridge = (train_x_tseries - gen_traj_ridge.T)**2
ridge_MSE = np.mean(mse_ridge)


#####################################################################################

# compute AIC

AIC_sparse = train_length*np.log(sparse_MSE) + 2*np.count_nonzero(W_out_sparse)
AIC_ridge = train_length*np.log(ridge_MSE) + 2*np.count_nonzero(W_out_ridge)

if np.isnan(AIC_sparse):
    AIC_sparse = 1e5
if np.isnan(AIC_ridge):
    AIC_ridge = 1e5

if AIC_sparse < AIC_ridge:
    W_out = W_out_sparse[0]
else:
    W_out = W_out_ridge[0]

#####################################################################################

polynomial = ['cte', 'x', 'p', 'x^2', 'px', 'p^2', 'x^3', 'px^2', 'p^2x', 'p^3']

dic_W_out = pd.DataFrame()

for j in range(dtot + cte):
    dic_W_out[polynomial[j]] = [W_out[j]]

dic_W_out.to_csv('../results/chick/chick_270_W_out_t.csv',header=True,index=False)

##
## prediction
##

# create a place to store feature vectors for prediction
out_test = np.ones(dtot + cte)               # full feature vector
x_test = np.zeros((dlin,test_length+1))      # linear part

x_test[-1,:] = xt_tseries[:,-1][train_length:]

# copy over initial linear feature vector
x_test[:,0] = x_train[:,-1]

predict_traj = list()
egv_list = list()

pd_bif_time = 0
pd_bif = 0

iterations = 100
last = 4

for j in range(test_length):

    dom_egv = comp_egv(x_test[0,j],x_test[1,j],W_out)

    egv_list.append(dom_egv)

    if j > 1:

        if egv_list[j-1] > -1 and egv_list[j] < -1:

            pd_bif_time = j + train_length
            pd_bif = x_test[1,j]

            break

    # copy linear part into whole feature vector
    out_test[cte:dlin + cte] = x_test[:,j] # shift by one for constant

    # fill in the non-linear part

    cnt = 0
    # 2-order
    for row1 in range(dlin):
        for row2 in range(row1,dlin):
            # shift by one for constant
            out_test[dlin + cnt + cte] = x_test[row1,j] * x_test[row2,j]
            cnt += 1

    # 3-order
    for row1 in range(dlin):
        for row2 in range(row1,dlin):
            for row3 in range(row2,dlin):
                # shift by one for constant
                out_test[dlin + cnt + cte] = x_test[row1,j] * x_test[row2,j] * x_test[row3,j]
                cnt += 1

    # do a prediction
    x_test[0:d,j+1] = W_out @ out_test[:]

    predict_traj.append(x_test[0,j+1])

if pd_bif_time != 0:

    print('---------------- Find pb bif ----------------')

    t_chaos = list()
    x_chaos = list()

    t_chaos.append(tseries[pd_bif_time])
    x_chaos.append(predict_traj[-1])

    for j in range(pd_bif_time,length):

        x = x_chaos[-1]
        p = x_test[1,j-train_length]

        for _ in range(iterations - last):
            x = F(x, p, W_out)

        for _ in range(last):
            x = F(x, p, W_out)
            t_chaos.append(tseries[j])
            x_chaos.append(x)

    dic_pred_2 = {'Time':t_chaos[last+1:],'pred':x_chaos[last+1:]}
    pred_2_out = pd.DataFrame(dic_pred_2)
    pred_2_out.to_csv('../results/chick/chick_270_pred_2_t.csv',header = True, index=False)

else:

    print('---------------- Find no pb bif ----------------')

dic_pred_1 = {'Time':tseries[train_length + 1:train_length + 1 + len(predict_traj)],
              'traj':x_tseries[train_length + 1:train_length + 1 + len(predict_traj),0],
              'pred':predict_traj}

pred_1_out = pd.DataFrame(dic_pred_1)
pred_1_out.to_csv('../results/chick/chick_270_pred_1_t.csv',header = True, index=False)

# generation

# create a place to store feature vectors for prediction
out_gen = np.ones(dtot + cte)               # full feature vector
x_gen = np.zeros((dlin,length))      # linear part

x_gen[-1,:] = xt_tseries[:,-1]

# copy over initial linear feature vector
x_gen[:,0] = xt_tseries[0,:]

gen_traj = list()

for j in range(length-1):

    # copy linear part into whole feature vector
    out_gen[cte:dlin + cte] = x_gen[:,j] # shift by one for constant

    # fill in the non-linear part

    cnt = 0
    # 2-order
    for row1 in range(dlin):
        for row2 in range(row1,dlin):
            # shift by one for constant
            out_gen[dlin + cnt + cte] = x_gen[row1,j] * x_gen[row2,j]
            cnt += 1

    # 3-order
    for row1 in range(dlin):
        for row2 in range(row1,dlin):
            for row3 in range(row2,dlin):
                # shift by one for constant
                out_gen[dlin + cnt + cte] = x_gen[row1,j] * x_gen[row2,j] * x_gen[row3,j]
                cnt += 1

    # do a prediction
    x_gen[0:d,j+1] = W_out @ out_gen[:]

    gen_traj.append(x_gen[0,j+1])

dic_gen = {'Time':t_series[1:,0],'traj':x_tseries[1:,0],'gen':gen_traj}

gen_out = pd.DataFrame(dic_gen)
gen_out.to_csv('../results/chick/chick_270_gen_t.csv',header = True, index=False)