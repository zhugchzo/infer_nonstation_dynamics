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

print('time pinv error: {}'.format(pinv_error))