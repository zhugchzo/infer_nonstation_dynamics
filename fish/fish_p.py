import numpy as np
import pandas as pd
import os
        
def generate_dynamic_function(M, d, k, type): # this function is used to generate dynamical functions 
                                              # using different basis functions in our function library

    if type == 1:
        index_max = 3*k*d
    elif type == 2:
        index_max = (3*k-1)*d
    elif type == 3:
        index_max = (2*k+1)*d
    elif type == 4:
        index_max = 2*k*d
    
    S = range(d)                           
    Y = range(d, index_max) if k>1 else []       
    P = index_max                               

    NN = M.shape[1]
    
    #-----------------------
    # Degree 1 terms
    # x_i and p
    degree1_terms = []
    degree1_names = []
    for i in S:
        degree1_terms.append(M[i,:])    # x_i
        degree1_names.append(f"x_{i+1}")
    # p
    degree1_terms.append(M[P,:])
    degree1_names.append("p")
    
    degree1_matrix = np.vstack(degree1_terms) # shape: (d+1, N)

    #-----------------------
    # Degree 2 terms
    degree2_terms = []
    degree2_names = []
    
    SP_indices = list(S) + [P]
    for i in range(len(SP_indices)):
        idx1 = SP_indices[i]
        term_val = M[idx1,:]**2
        var_name = "p" if idx1==P else f"x_{idx1+1}"
        degree2_terms.append(term_val)
        degree2_names.append(f"{var_name}^2")
        for j in range(i+1, len(SP_indices)):
            idx2 = SP_indices[j]
            term_val = M[idx1,:]*M[idx2,:]
            var_name1 = "p" if idx1==P else f"x_{idx1+1}"
            var_name2 = "p" if idx2==P else f"x_{idx2+1}"
            degree2_terms.append(term_val)
            degree2_names.append(f"{var_name1}*{var_name2}")
    
    for i in S:
        for j in range(d, index_max):
            term_val = M[i,:]*M[j,:]
            degree2_terms.append(term_val)
            degree2_names.append(f"x_{i+1}*y_{j-d+1}")
    
    degree2_matrix = np.vstack(degree2_terms) # shape: (num_2nd_terms, N)

    #-----------------------
    # Degree 3 terms
    degree3_terms = []
    degree3_names = []
    
    from itertools import product
    for combo in product(SP_indices, repeat=3):
        sorted_combo = sorted(combo)
        val = M[sorted_combo[0],:]*M[sorted_combo[1],:]*M[sorted_combo[2],:]
        var_names = []
        for c in sorted_combo:
            var_names.append("p" if c==P else f"x_{c+1}")
        term_name = "*".join(var_names)

        degree3_terms.append(val)
        degree3_names.append(term_name)

    unique_terms = {}
    for name, arr in zip(degree3_names, degree3_terms):
        if name not in unique_terms:
            unique_terms[name] = arr
        else:
            pass
    degree3_names = list(unique_terms.keys())
    degree3_terms = [unique_terms[n] for n in degree3_names]

    two_vars_xp = []
    two_vars_xp_names = []
    for i in range(len(SP_indices)):
        idx1 = SP_indices[i]
        for j in range(i, len(SP_indices)):
            idx2 = SP_indices[j]
            arr = M[idx1,:]*M[idx2,:]
            vars_ = [( "p" if idx1==P else f"x_{idx1+1}" ),
                     ( "p" if idx2==P else f"x_{idx2+1}" )]
            name_ = "*".join(sorted(vars_))
            if idx1 in S or idx2 in S:
                two_vars_xp.append(arr)
                two_vars_xp_names.append(name_)
    
    for j in range(d, index_max):
        y_name = f"y_{j-d+1}"
        for arr2, n2 in zip(two_vars_xp, two_vars_xp_names):
            val = M[j,:]*arr2
            name_ = y_name + "*" + n2
            if name_ not in unique_terms:
                unique_terms[name_] = val

    for i_s in S:
        x_name = f"x_{i_s+1}"
        for idx_j1 in range(d, index_max):
            for idx_j2 in range(idx_j1, index_max):
                arr = M[i_s,:]*M[idx_j1,:]*M[idx_j2,:]
                y_name1 = f"y_{idx_j1-d+1}"
                y_name2 = f"y_{idx_j2-d+1}"
                sorted_y = sorted([y_name1,y_name2])
                name_ = x_name + "*" + "*".join(sorted_y)
                if name_ not in unique_terms:
                    unique_terms[name_] = arr

    degree3_names = list(unique_terms.keys())
    degree3_terms = [unique_terms[n] for n in degree3_names]

    degree3_matrix = np.vstack(degree3_terms) if len(degree3_terms)>0 else np.empty((0,NN))

    return degree1_matrix, degree2_matrix, degree3_matrix, ['cte']+degree1_names+degree2_names+degree3_names

if not os.path.exists('../results/fish'):
    os.makedirs('../results/fish')

# Load the saved data
df_tseries = pd.read_csv('fish_data.csv')
df_network = pd.read_csv('fish_network.csv')

col_T = ['surf.t', 'bot.t']
col_fish = ['Aurelia.sp', 'Plotosus.japonicus', 'Sebastes.cheni', 'Trachurus.japonicus', 'Girella.punctata',
       'Pseudolabrus.sieboldi', 'Parajulis.poecilopterus', 'Halichoeres.tenuispinnis', 'Chaenogobius.gulosus',
       'Pterogobius.zonoleucus', 'Tridentiger.trigonocephalus', 'Siganus.fuscescens', 'Sphyraena.pinguis', 'Rudarius.ercodes']

data_network = df_network[col_fish].values

# choose cte = 1 to include constant term
cte = 1
# input dimension of x
d = 1
# input dimension of theta
d_theta = 1

# the number of node
N = len(col_fish)

data_tseries = df_tseries[col_fish].values
sbT_tseries = df_tseries[col_T].values
T_tseries = sbT_tseries[:,0]

length = len(data_tseries)

time = np.arange(1, 1 + length, 1)

train_length = 254
test_length = length - train_length - 1

period = 24

a = 1
b = 1

theta_train = T_tseries[:train_length+1]
theta_valid = theta_train.copy()
theta_train = theta_train.reshape(-1, 1)

#####################################################################################

sparse_node_dictionary_list = list()
ridge_node_dictionary_list = list()

sparse_MSE = 0
ridge_MSE = 0

sparse_nozero = 0
ridge_nozero = 0

sparse_p_zero_count = 0
sparse_self_zero_count = 0
sparse_inter_zero_count = 0

for node in range(N):

    node_dictionary_sparse = {}
    node_dictionary_ridge = {}
    
    self_var_index = list(np.where(data_network[:,node] == 2)[0])
    inter_var_index = list(np.where(data_network[:,node] == 1)[0])

    si_var_index = self_var_index + inter_var_index

    node_dictionary_sparse['si_var_index'] = si_var_index
    node_dictionary_ridge['si_var_index'] = si_var_index

    node_data = data_tseries[:,si_var_index][:train_length+1]

    # input number of interactive node
    k = len(si_var_index)
    # size of linear part of feature vector (x1,...,xn,b)
    dlin = d + d_theta

    if node in [2,10]:
        dcop = (3*k-2)*d

    elif node in [4,5,9,12,13]:
        dcop = 2*k*d

    elif node in [0,3,8,11]:
        dcop = (2*k-1)*d

    else:
        dcop = (3*k-1)*d
    
    # size of nonlinear part of feature vector
    dnonlin_2 = int(((dlin + 1)*dlin/2) + (d*dcop))
    dnonlin_3 = int(((dlin + 2)*(dlin + 1)*dlin/6) + (((dlin + 1)*dlin/2-1)*dcop) + (((dcop + 1)*dcop/2)*d))
    dnonlin = dnonlin_2 + dnonlin_3
    # total size of feature vector: linear + nonlinear
    dtot = dlin + dnonlin

    node_data_theta = np.hstack((node_data, theta_train))

    # create an array to hold the linear part of the feature vector
    x_train = np.zeros((dlin + dcop,train_length + 1))

    # fill in the linear part of the feature vector for all times
    for j in range(train_length + 1):

        if node in [2,10]:
            x_train[:d,j] = node_data_theta[j][:d]
            x_train[d:2*d,j] = 1/(a+node_data_theta[j][:d])
            x_train[2*d:(1+k)*d,j] = node_data_theta[j][d:k*d]
            x_train[(1+k)*d:(2*k)*d,j] = 1/(a+node_data_theta[j][d:k*d])
            x_train[(2*k)*d:(3*k-1)*d,j] = 1/(b+np.exp(-node_data_theta[j][d:k*d]+np.tile(node_data_theta[j][:d],k-1)))
            x_train[-1,j] = node_data_theta[j][-1]

        elif node in [4,5,9,12,13]:
            x_train[:d,j] = node_data_theta[j][:d]
            x_train[d:2*d,j] = 1/(a+node_data_theta[j][:d])
            x_train[2*d:3*d,j] = 1/(b+np.exp(-node_data_theta[j][:d]))
            x_train[3*d:(k+2)*d,j] = node_data_theta[j][d:k*d]
            x_train[(k+2)*d:(2*k+1)*d,j] = 1/(b+np.exp(-node_data_theta[j][d:k*d]+np.tile(node_data_theta[j][:d],k-1)))
            x_train[-1,j] = node_data_theta[j][-1]

        elif node in [0]:
            x_train[:d,j] = node_data_theta[j][:d]
            x_train[d:2*d,j] = 1/(b+np.exp(-node_data_theta[j][:d]))
            x_train[2*d:(1+k)*d,j] = node_data_theta[j][d:k*d]
            x_train[(1+k)*d:2*k*d,j] = 1/(a+node_data_theta[j][d:k*d])
            x_train[-1,j] = node_data_theta[j][-1]

        elif node in [3,8,11]:
            x_train[:d,j] = node_data_theta[j][:d]
            x_train[d:2*d,j] = 1/(b+np.exp(-node_data_theta[j][:d]))
            x_train[2*d:(1+k)*d,j] = node_data_theta[j][d:k*d]
            x_train[(1+k)*d:2*k*d,j] = 1/(b+np.exp(-node_data_theta[j][d:k*d]+np.tile(node_data_theta[j][:d], k-1)))
            x_train[-1,j] = node_data_theta[j][-1]

        else:
            x_train[:d,j] = node_data_theta[j][:d]
            x_train[d:2*d,j] = 1/(a+node_data_theta[j][:d])
            x_train[2*d:3*d,j] = 1/(b+np.exp(-node_data_theta[j][:d]))
            x_train[3*d:(2+k)*d,j] = node_data_theta[j][d:k*d]
            x_train[(2+k)*d:(2*k+1)*d,j] = 1/(a+node_data_theta[j][d:k*d])
            x_train[(2*k+1)*d:(3*k)*d,j] = 1/(b+np.exp(-node_data_theta[j][d:k*d]+np.tile(node_data_theta[j][:d],k-1)))
            x_train[-1,j] = node_data_theta[j][-1]

    out_train = np.ones((dtot + cte,train_length))
        
    if node in [2,10]:
        out_train[cte:dlin + cte,:] = generate_dynamic_function(x_train[:,:train_length], d, k, 2)[0]  
        out_train[dlin + cte : dlin + dnonlin_2 + cte, :] = generate_dynamic_function(x_train[:,:train_length], d, k, 2)[1]
        out_train[dlin + cte + dnonlin_2:, :] = generate_dynamic_function(x_train[:,:train_length], d, k, 2)[2]
        f_library = generate_dynamic_function(x_train[:,:train_length], d, k, 2)[3]

    elif node in [4,5,9,12,13]:
        out_train[cte:dlin + cte,:] = generate_dynamic_function(x_train[:,:train_length], d, k, 3)[0]  
        out_train[dlin + cte : dlin + dnonlin_2 + cte, :] = generate_dynamic_function(x_train[:,:train_length], d, k, 3)[1]
        out_train[dlin + cte + dnonlin_2:, :] = generate_dynamic_function(x_train[:,:train_length], d, k, 3)[2]
        f_library = generate_dynamic_function(x_train[:,:train_length], d, k, 3)[3]

    elif node in [0,3,8,11]:
        out_train[cte:dlin + cte,:] = generate_dynamic_function(x_train[:,:train_length], d, k, 4)[0]  
        out_train[dlin + cte : dlin + dnonlin_2 + cte, :] = generate_dynamic_function(x_train[:,:train_length], d, k, 4)[1]
        out_train[dlin + cte + dnonlin_2:, :] = generate_dynamic_function(x_train[:,:train_length], d, k, 4)[2]
        f_library = generate_dynamic_function(x_train[:,:train_length], d, k, 4)[3]
        
    else:
        out_train[cte:dlin + cte,:] = generate_dynamic_function(x_train[:,:train_length], d, k, 1)[0]  
        out_train[dlin + cte : dlin + dnonlin_2 + cte, :] = generate_dynamic_function(x_train[:,:train_length], d, k, 1)[1]
        out_train[dlin + cte + dnonlin_2:, :] = generate_dynamic_function(x_train[:,:train_length], d, k, 1)[2]
        f_library = generate_dynamic_function(x_train[:,:train_length], d, k, 1)[3]
    
    ##
    ## SINDy
    ##
    
    W_out_sparse = x_train[0:d,1:train_length + 1] @ out_train[:,:].T @ np.linalg.pinv(out_train[:,:] @ out_train[:,:].T)

    # Set the sparsification parameter lambda
    # Adjustable parameter, modify according to your data
    if node in [1]:
        lambda_param = 0.02
    else:
        lambda_param = 0.1

    # Perform sparsification, iterate multiple times to obtain a sparse solution
    for k in range(10):  # Number of iterations, can be adjusted as needed
        # Find the coefficients smaller than lambda_param and set them to zero
        smallinds = (np.abs(W_out_sparse) < lambda_param)
        W_out_sparse[smallinds] = 0
        
        # For each state dimension, perform least-squares regression again, only keeping the large coefficients
        for ind in range(d):  # Iterate over each state dimension
            biginds = ~smallinds[ind, :]  # Indices of large coefficients
            # Perform regression using only the non-zero coefficients
            W_out_sparse[ind, biginds] = x_train[ind,1:train_length + 1] @ out_train[biginds,:].T @ np.linalg.pinv(out_train[biginds,:] @ out_train[biginds,:].T)

    node_dictionary_sparse['W_out'] = W_out_sparse[0]
    sparse_nozero += np.count_nonzero(W_out_sparse[0])

    sparse_node_dictionary_list.append(node_dictionary_sparse)

    ##
    ## Ridge
    ##
    
    ridge_param = 1e-5

    W_out_ridge = x_train[0:d,1:train_length + 1] @ out_train[:,:].T @ np.linalg.pinv(out_train[:,:] @ out_train[:,:].T + ridge_param*np.identity(dtot+ cte))

    node_dictionary_ridge['W_out'] = W_out_ridge[0]
    ridge_nozero += np.count_nonzero(W_out_ridge[0])

    ridge_node_dictionary_list.append(node_dictionary_ridge)
    
    p_indexes = [i for i, item in enumerate(f_library) if 'p' in item]
    if np.sum(W_out_sparse[0][p_indexes]) == 0:
        sparse_p_zero_count += 1

    if node in [0,2,3,8,10,11]:
        self_var_name = ['x_1','x_1^2','x_1*p','x_1*y_1','x_1*x_1*x_1','x_1*x_1*p','x_1*p*p','y_1*x_1*x_1','y_1*p*x_1','x_1*y_1*y_1']
        self_indexes = [i for i, item in enumerate(f_library) if item in self_var_name]
        if self_indexes and np.sum(W_out_sparse[0][self_indexes]) == 0:
            sparse_self_zero_count += 1                

    else:
        self_var_name = ['x_1', 'x_1^2', 'x_1*p', 'x_1*y_1', 'x_1*y_2', 'x_1*x_1*x_1', 'x_1*x_1*p', 'x_1*p*p', 'y_1*x_1*x_1', 'y_1*p*x_1', 'y_2*x_1*x_1', 'y_2*p*x_1', 'x_1*y_1*y_1', 'x_1*y_1*y_2', 'x_1*y_2*y_2']
        inter_var_name = ['y_{}'.format(3+i) for i in range(int(2*len(inter_var_index)))]
        self_indexes = [i for i, item in enumerate(f_library) if item in self_var_name]
        if self_indexes and np.sum(W_out_sparse[0][self_indexes]) == 0:
            sparse_self_zero_count += 1   

    if node in [2,10]:
        inter_var_name = ['y_{}'.format(2+i) for i in range(int(3*len(inter_var_index)))]
        inter_indexes = [i for i, item in enumerate(f_library) if any(name in item for name in inter_var_name)]
        if inter_indexes and np.sum(W_out_sparse[0][inter_indexes]) == 0:
            sparse_inter_zero_count += 1
    
    elif node in [4,5,9,12,13]:
        inter_var_name = ['y_{}'.format(3+i) for i in range(int(2*len(inter_var_index)))]
        inter_indexes = [i for i, item in enumerate(f_library) if any(name in item for name in inter_var_name)]
        if inter_indexes and np.sum(W_out_sparse[0][inter_indexes]) == 0:
            sparse_inter_zero_count += 1
    
    elif node in [0]:
        inter_var_name = ['y_{}'.format(2+i) for i in range(int(2*len(inter_var_index)))]
        inter_indexes = [i for i, item in enumerate(f_library) if any(name in item for name in inter_var_name)]
        if inter_indexes and np.sum(W_out_sparse[0][inter_indexes]) == 0:
            sparse_inter_zero_count += 1

    elif node in [3,8,11]:
        inter_var_name = ['y_{}'.format(2+i) for i in range(int(2*len(inter_var_index)))]
        inter_indexes = [i for i, item in enumerate(f_library) if any(name in item for name in inter_var_name)]
        if inter_indexes and np.sum(W_out_sparse[0][inter_indexes]) == 0:
            sparse_inter_zero_count += 1

    else:
        inter_var_name = ['y_{}'.format(3+i) for i in range(int(3*len(inter_var_index)))]
        inter_indexes = [i for i, item in enumerate(f_library) if any(name in item for name in inter_var_name)]
        if inter_indexes and np.sum(W_out_sparse[0][inter_indexes]) == 0:
            sparse_inter_zero_count += 1
    
    # confirm virtual forcing parameter in governing equations
    if sparse_p_zero_count == N:
        sparse_MSE = np.nan
    
    # confirm self dynamics in governing equations
    if sparse_self_zero_count > 1:
        sparse_MSE = np.nan
    
    # confirm interaction dynamics in governing equations
    if sparse_inter_zero_count > 1:
        sparse_MSE = np.nan
    
        # confirm there is no governing equation being zero
    if np.sum(W_out_sparse[0]) == 0:
        sparse_MSE = np.nan

#####################################################################################

# compute MSE

next_x_train = data_tseries[1:train_length+1]
MSE_x_train = data_tseries[0:train_length]

splits = np.split(MSE_x_train, indices_or_sections=np.arange(period, MSE_x_train.shape[0], period), axis=0)

# SINDy

sparse_valid_matrix = np.zeros((N+d_theta,train_length+1))  # linear part
sparse_valid_matrix[N+d_theta-1,:] = theta_valid

MSE_index = 0

for one_period in splits:

    period_test_length = len(one_period)

    sparse_valid_matrix[:N+d_theta-1,0] = one_period[0,:]

    # do validation
    for j in range(period_test_length):

        for node in range(N):

            node_dictionary = sparse_node_dictionary_list[node]

            si_var_index = node_dictionary['si_var_index']
            W_out = node_dictionary['W_out']

            # input number of interactive node
            k = len(si_var_index)
            # size of linear part of feature vector (x1,...,xn,b)
            dlin = d + d_theta

            if node in [2,10]:
                dcop = (3*k-2)*d

            elif node in [4,5,9,12,13]:
                dcop = 2*k*d  

            elif node in [0,3,8,11]:
                dcop = (2*k-1)*d

            else:
                dcop = (3*k-1)*d
            
            # size of nonlinear part of feature vector
            dnonlin_2 = int(((dlin + 1)*dlin/2) + (d*dcop))
            dnonlin_3 = int(((dlin + 2)*(dlin + 1)*dlin/6) + (((dlin + 1)*dlin/2-1)*dcop) + (((dcop + 1)*dcop/2)*d))
            dnonlin = dnonlin_2 + dnonlin_3
            # total size of feature vector: linear + nonlinear
            dtot = dlin + dnonlin

            # create a place to store feature vectors for prediction
            sparse_out_valid = np.ones((dtot + cte,1))  # full feature vector
            
            var_index = si_var_index.copy()
            var_index.append(N+d_theta-1)
            
            sparse_x_valid = np.zeros(dlin + dcop)

            if node in [2,10]:
                sparse_x_valid[:d] = sparse_valid_matrix[var_index,j][:d]
                sparse_x_valid[d:2*d] = 1/(a+sparse_valid_matrix[var_index,j][:d])
                sparse_x_valid[2*d:(1+k)*d] = sparse_valid_matrix[var_index,j][d:k*d]
                sparse_x_valid[(1+k)*d:(2*k)*d] = 1/(a+sparse_valid_matrix[var_index,j][d:k*d])
                sparse_x_valid[(2*k)*d:(3*k-1)*d] = 1/(b+np.exp(-sparse_valid_matrix[var_index,j][d:k*d]+np.tile(sparse_valid_matrix[var_index,j][:d],k-1)))
                sparse_x_valid[-1] = sparse_valid_matrix[var_index,j][-1]

            elif node in [4,5,9,12,13]:
                sparse_x_valid[:d] = sparse_valid_matrix[var_index,j][:d]
                sparse_x_valid[d:2*d] = 1/(a+sparse_valid_matrix[var_index,j][:d])
                sparse_x_valid[2*d:3*d] = 1/(b+np.exp(-sparse_valid_matrix[var_index,j][:d]))
                sparse_x_valid[3*d:(k+2)*d] = sparse_valid_matrix[var_index,j][d:k*d]
                sparse_x_valid[(k+2)*d:(2*k+1)*d] = 1/(b+np.exp(-sparse_valid_matrix[var_index,j][d:k*d]+np.tile(sparse_valid_matrix[var_index,j][:d],k-1)))
                sparse_x_valid[-1] = sparse_valid_matrix[var_index,j][-1]

            elif node in [0]:
                sparse_x_valid[:d] = sparse_valid_matrix[var_index,j][:d]
                sparse_x_valid[d:2*d] = 1/(b+np.exp(-sparse_valid_matrix[var_index,j][:d]))
                sparse_x_valid[2*d:(1+k)*d] = sparse_valid_matrix[var_index,j][d:k*d]
                sparse_x_valid[(1+k)*d:2*k*d] = 1/(a+sparse_valid_matrix[var_index,j][d:k*d])
                sparse_x_valid[-1] = sparse_valid_matrix[var_index,j][-1]

            elif node in [3,8,11]:
                sparse_x_valid[:d] = sparse_valid_matrix[var_index,j][:d]
                sparse_x_valid[d:2*d] = 1/(b+np.exp(-sparse_valid_matrix[var_index,j][:d]))
                sparse_x_valid[2*d:(1+k)*d] = sparse_valid_matrix[var_index,j][d:k*d]
                sparse_x_valid[(1+k)*d:2*k*d] = 1/(b+np.exp(-sparse_valid_matrix[var_index,j][d:k*d]+np.tile(sparse_valid_matrix[var_index,j][:d],k-1)))
                sparse_x_valid[-1] = sparse_valid_matrix[var_index,j][-1]
            
            else:
                sparse_x_valid[:d] = sparse_valid_matrix[var_index,j][:d]
                sparse_x_valid[d:2*d] = 1/(a+sparse_valid_matrix[var_index,j][:d])
                sparse_x_valid[2*d:3*d] = 1/(b+np.exp(-sparse_valid_matrix[var_index,j][:d]))
                sparse_x_valid[3*d:(2+k)*d] = sparse_valid_matrix[var_index,j][d:k*d]
                sparse_x_valid[(2+k)*d:(2*k+1)*d] = 1/(a+sparse_valid_matrix[var_index,j][d:k*d])
                sparse_x_valid[(2*k+1)*d:(3*k)*d] = 1/(b+np.exp(-sparse_valid_matrix[var_index,j][d:k*d]+np.tile(sparse_valid_matrix[var_index,j][:d],k-1)))
                sparse_x_valid[-1] = sparse_valid_matrix[var_index,j][-1]

            sparse_x_valid = sparse_x_valid.reshape(-1,1)

            if node in [2,10]:
                sparse_out_valid[cte:dlin + cte] = generate_dynamic_function(sparse_x_valid, d, k, 2)[0]
                sparse_out_valid[dlin + cte : dlin + dnonlin_2 + cte, :] = generate_dynamic_function(sparse_x_valid, d, k, 2)[1]
                sparse_out_valid[dlin + cte + dnonlin_2:, :] = generate_dynamic_function(sparse_x_valid, d, k, 2)[2]

            elif node in [4,5,9,12,13]:
                sparse_out_valid[cte:dlin + cte] = generate_dynamic_function(sparse_x_valid, d, k, 3)[0]
                sparse_out_valid[dlin + cte : dlin + dnonlin_2 + cte, :] = generate_dynamic_function(sparse_x_valid, d, k, 3)[1]
                sparse_out_valid[dlin + cte + dnonlin_2:, :] = generate_dynamic_function(sparse_x_valid, d, k, 3)[2]
            
            elif node in [0,3,8,11]:
                sparse_out_valid[cte:dlin + cte] = generate_dynamic_function(sparse_x_valid, d, k, 4)[0]
                sparse_out_valid[dlin + cte : dlin + dnonlin_2 + cte, :] = generate_dynamic_function(sparse_x_valid, d, k, 4)[1]
                sparse_out_valid[dlin + cte + dnonlin_2:, :] = generate_dynamic_function(sparse_x_valid, d, k, 4)[2]           
            
            else:
                sparse_out_valid[cte:dlin + cte] = generate_dynamic_function(sparse_x_valid, d, k, 1)[0]
                sparse_out_valid[dlin + cte : dlin + dnonlin_2 + cte, :] = generate_dynamic_function(sparse_x_valid, d, k, 1)[1]
                sparse_out_valid[dlin + cte + dnonlin_2:, :] = generate_dynamic_function(sparse_x_valid, d, k, 1)[2]

            prediction = (W_out @ sparse_out_valid[:])[0]

            if prediction < 0:
                prediction = 0

            sparse_valid_matrix[node,MSE_index+j+1] = prediction
    
    MSE_index += period_test_length

mse_sparse = (next_x_train - sparse_valid_matrix[:N+d_theta-1,1:].T)**2
mse_sparse_normalized = (mse_sparse - np.min(mse_sparse, axis=0)) / (np.max(mse_sparse, axis=0) - np.min(mse_sparse, axis=0))

column_sums_sparse = np.sum(mse_sparse_normalized, axis=1)

sparse_MSE += np.mean(column_sums_sparse)

# Ridge

ridge_valid_matrix = np.zeros((N+d_theta,train_length+1))  # linear part
ridge_valid_matrix[N+d_theta-1,:] = theta_valid

MSE_index = 0

for one_period in splits:

    period_test_length = len(one_period)

    ridge_valid_matrix[:N+d_theta-1,0] = one_period[0,:]

    # do validation
    for j in range(period_test_length):

        for node in range(N):

            node_dictionary = ridge_node_dictionary_list[node]

            si_var_index = node_dictionary['si_var_index']
            W_out = node_dictionary['W_out']

            # input number of interactive node
            k = len(si_var_index)
            # size of linear part of feature vector (x1,...,xn,b)
            dlin = d + d_theta

            if node in [2,10]:
                dcop = (3*k-2)*d

            elif node in [4,5,9,12,13]:
                dcop = 2*k*d  

            elif node in [0,3,8,11]:
                dcop = (2*k-1)*d

            else:
                dcop = (3*k-1)*d
            
            # size of nonlinear part of feature vector
            dnonlin_2 = int(((dlin + 1)*dlin/2) + (d*dcop))
            dnonlin_3 = int(((dlin + 2)*(dlin + 1)*dlin/6) + (((dlin + 1)*dlin/2-1)*dcop) + (((dcop + 1)*dcop/2)*d))
            dnonlin = dnonlin_2 + dnonlin_3
            # total size of feature vector: linear + nonlinear
            dtot = dlin + dnonlin

            # create a place to store feature vectors for prediction
            ridge_out_valid = np.ones((dtot + cte,1))  # full feature vector
            
            var_index = si_var_index.copy()
            var_index.append(N+d_theta-1)
            
            ridge_x_valid = np.zeros(dlin + dcop)

            if node in [2,10]:
                ridge_x_valid[:d] = ridge_valid_matrix[var_index,j][:d]
                ridge_x_valid[d:2*d] = 1/(a+ridge_valid_matrix[var_index,j][:d])
                ridge_x_valid[2*d:(1+k)*d] = ridge_valid_matrix[var_index,j][d:k*d]
                ridge_x_valid[(1+k)*d:(2*k)*d] = 1/(a+ridge_valid_matrix[var_index,j][d:k*d])
                ridge_x_valid[(2*k)*d:(3*k-1)*d] = 1/(b+np.exp(-ridge_valid_matrix[var_index,j][d:k*d]+np.tile(ridge_valid_matrix[var_index,j][:d],k-1)))
                ridge_x_valid[-1] = ridge_valid_matrix[var_index,j][-1]

            elif node in [4,5,9,12,13]:
                ridge_x_valid[:d] = ridge_valid_matrix[var_index,j][:d]
                ridge_x_valid[d:2*d] = 1/(a+ridge_valid_matrix[var_index,j][:d])
                ridge_x_valid[2*d:3*d] = 1/(b+np.exp(-ridge_valid_matrix[var_index,j][:d]))
                ridge_x_valid[3*d:(k+2)*d] = ridge_valid_matrix[var_index,j][d:k*d]
                ridge_x_valid[(k+2)*d:(2*k+1)*d] = 1/(b+np.exp(-ridge_valid_matrix[var_index,j][d:k*d]+np.tile(ridge_valid_matrix[var_index,j][:d],k-1)))
                ridge_x_valid[-1] = ridge_valid_matrix[var_index,j][-1]

            elif node in [0]:
                ridge_x_valid[:d] = ridge_valid_matrix[var_index,j][:d]
                ridge_x_valid[d:2*d] = 1/(b+np.exp(-ridge_valid_matrix[var_index,j][:d]))
                ridge_x_valid[2*d:(1+k)*d] = ridge_valid_matrix[var_index,j][d:k*d]
                ridge_x_valid[(1+k)*d:2*k*d] = 1/(a+ridge_valid_matrix[var_index,j][d:k*d])
                ridge_x_valid[-1] = ridge_valid_matrix[var_index,j][-1]

            elif node in [3,8,11]:
                ridge_x_valid[:d] = ridge_valid_matrix[var_index,j][:d]
                ridge_x_valid[d:2*d] = 1/(b+np.exp(-ridge_valid_matrix[var_index,j][:d]))
                ridge_x_valid[2*d:(1+k)*d] = ridge_valid_matrix[var_index,j][d:k*d]
                ridge_x_valid[(1+k)*d:2*k*d] = 1/(b+np.exp(-ridge_valid_matrix[var_index,j][d:k*d]+np.tile(ridge_valid_matrix[var_index,j][:d],k-1)))
                ridge_x_valid[-1] = ridge_valid_matrix[var_index,j][-1]
            
            else:
                ridge_x_valid[:d] = ridge_valid_matrix[var_index,j][:d]
                ridge_x_valid[d:2*d] = 1/(a+ridge_valid_matrix[var_index,j][:d])
                ridge_x_valid[2*d:3*d] = 1/(b+np.exp(-ridge_valid_matrix[var_index,j][:d]))
                ridge_x_valid[3*d:(2+k)*d] = ridge_valid_matrix[var_index,j][d:k*d]
                ridge_x_valid[(2+k)*d:(2*k+1)*d] = 1/(a+ridge_valid_matrix[var_index,j][d:k*d])
                ridge_x_valid[(2*k+1)*d:(3*k)*d] = 1/(b+np.exp(-ridge_valid_matrix[var_index,j][d:k*d]+np.tile(ridge_valid_matrix[var_index,j][:d],k-1)))
                ridge_x_valid[-1] = ridge_valid_matrix[var_index,j][-1]

            ridge_x_valid = ridge_x_valid.reshape(-1,1)

            if node in [2,10]:
                ridge_out_valid[cte:dlin + cte] = generate_dynamic_function(ridge_x_valid, d, k, 2)[0]
                ridge_out_valid[dlin + cte : dlin + dnonlin_2 + cte, :] = generate_dynamic_function(ridge_x_valid, d, k, 2)[1]
                ridge_out_valid[dlin + cte + dnonlin_2:, :] = generate_dynamic_function(ridge_x_valid, d, k, 2)[2]

            elif node in [4,5,9,12,13]:
                ridge_out_valid[cte:dlin + cte] = generate_dynamic_function(ridge_x_valid, d, k, 3)[0]
                ridge_out_valid[dlin + cte : dlin + dnonlin_2 + cte, :] = generate_dynamic_function(ridge_x_valid, d, k, 3)[1]
                ridge_out_valid[dlin + cte + dnonlin_2:, :] = generate_dynamic_function(ridge_x_valid, d, k, 3)[2]
            
            elif node in [0,3,8,11]:
                ridge_out_valid[cte:dlin + cte] = generate_dynamic_function(ridge_x_valid, d, k, 4)[0]
                ridge_out_valid[dlin + cte : dlin + dnonlin_2 + cte, :] = generate_dynamic_function(ridge_x_valid, d, k, 4)[1]
                ridge_out_valid[dlin + cte + dnonlin_2:, :] = generate_dynamic_function(ridge_x_valid, d, k, 4)[2]         
            
            else:
                ridge_out_valid[cte:dlin + cte] = generate_dynamic_function(ridge_x_valid, d, k, 1)[0]
                ridge_out_valid[dlin + cte : dlin + dnonlin_2 + cte, :] = generate_dynamic_function(ridge_x_valid, d, k, 1)[1]
                ridge_out_valid[dlin + cte + dnonlin_2:, :] = generate_dynamic_function(ridge_x_valid, d, k, 1)[2]

            prediction = (W_out @ ridge_out_valid[:])[0]

            if prediction < 0:
                prediction = 0

            ridge_valid_matrix[node,MSE_index+j+1] = prediction
    
    MSE_index += period_test_length

mse_ridge = (next_x_train - ridge_valid_matrix[:N+d_theta-1,1:].T)**2
mse_ridge_normalized = (mse_ridge - np.min(mse_ridge, axis=0)) / (np.max(mse_ridge, axis=0) - np.min(mse_ridge, axis=0))

column_sums_ridge = np.sum(mse_ridge_normalized, axis=1)

ridge_MSE += np.mean(column_sums_ridge)

#####################################################################################

# compute AIC

AIC_sparse = train_length*np.log(sparse_MSE) + 2*sparse_nozero
AIC_ridge = train_length*np.log(ridge_MSE) + 2*ridge_nozero

if np.isnan(AIC_sparse):
    AIC_sparse = 1e5
if np.isnan(AIC_ridge):
    AIC_ridge = 1e5

if AIC_sparse < AIC_ridge:
    node_dictionary_list = sparse_node_dictionary_list
else:
    node_dictionary_list = ridge_node_dictionary_list

##
## prediction
##

test_matrix = np.zeros((N+d_theta,test_length+1))  # linear part

theta_test = T_tseries[train_length:]

test_matrix[N+d_theta-1,:] = theta_test
# copy over initial linear feature vector
test_matrix[:N+d_theta-1,0] = data_tseries[train_length]

# do prediction
for j in range(test_length):

    for node in range(N):

        node_dictionary = node_dictionary_list[node]

        si_var_index = node_dictionary['si_var_index']
        W_out = node_dictionary['W_out']

        # input number of interactive node
        k = len(si_var_index)
        # size of linear part of feature vector (x1,...,xn,b)
        dlin = d + d_theta

        if node in [2,10]:
            dcop = (3*k-2)*d

        elif node in [4,5,9,12,13]:
            dcop = 2*k*d  

        elif node in [0,3,8,11]:
            dcop = (2*k-1)*d

        else:
            dcop = (3*k-1)*d
        
        # size of nonlinear part of feature vector
        dnonlin_2 = int(((dlin + 1)*dlin/2) + (d*dcop))
        dnonlin_3 = int(((dlin + 2)*(dlin + 1)*dlin/6) + (((dlin + 1)*dlin/2-1)*dcop) + (((dcop + 1)*dcop/2)*d))
        dnonlin = dnonlin_2 + dnonlin_3
        # total size of feature vector: linear + nonlinear
        dtot = dlin + dnonlin

        # create a place to store feature vectors for prediction
        out_test = np.ones((dtot + cte,1))  # full feature vector
        
        var_index = si_var_index.copy()
        var_index.append(N+d_theta-1)
        
        x_test = np.zeros(dlin + dcop)

        if node in [2,10]:
            x_test[:d] = test_matrix[var_index,j][:d]
            x_test[d:2*d] = 1/(a+test_matrix[var_index,j][:d])
            x_test[2*d:(1+k)*d] = test_matrix[var_index,j][d:k*d]
            x_test[(1+k)*d:(2*k)*d] = 1/(a+test_matrix[var_index,j][d:k*d])
            x_test[(2*k)*d:(3*k-1)*d] = 1/(b+np.exp(-test_matrix[var_index,j][d:k*d]+np.tile(test_matrix[var_index,j][:d],k-1)))
            x_test[-1] = test_matrix[var_index,j][-1]

        elif node in [4,5,9,12,13]:
            x_test[:d] = test_matrix[var_index,j][:d]
            x_test[d:2*d] = 1/(a+test_matrix[var_index,j][:d])
            x_test[2*d:3*d] = 1/(b+np.exp(-test_matrix[var_index,j][:d]))
            x_test[3*d:(k+2)*d] = test_matrix[var_index,j][d:k*d]
            x_test[(k+2)*d:(2*k+1)*d] = 1/(b+np.exp(-test_matrix[var_index,j][d:k*d]+np.tile(test_matrix[var_index,j][:d],k-1)))
            x_test[-1] = test_matrix[var_index,j][-1]

        elif node in [0]:
            x_test[:d] = test_matrix[var_index,j][:d]
            x_test[d:2*d] = 1/(b+np.exp(-test_matrix[var_index,j][:d]))
            x_test[2*d:(1+k)*d] = test_matrix[var_index,j][d:k*d]
            x_test[(1+k)*d:2*k*d] = 1/(a+test_matrix[var_index,j][d:k*d])
            x_test[-1] = test_matrix[var_index,j][-1]

        elif node in [3,8,11]:
            x_test[:d] = test_matrix[var_index,j][:d]
            x_test[d:2*d] = 1/(b+np.exp(-test_matrix[var_index,j][:d]))
            x_test[2*d:(1+k)*d] = test_matrix[var_index,j][d:k*d]
            x_test[(1+k)*d:2*k*d] = 1/(b+np.exp(-test_matrix[var_index,j][d:k*d]+np.tile(test_matrix[var_index,j][:d],k-1)))
            x_test[-1] = test_matrix[var_index,j][-1]
        
        else:
            x_test[:d] = test_matrix[var_index,j][:d]
            x_test[d:2*d] = 1/(a+test_matrix[var_index,j][:d])
            x_test[2*d:3*d] = 1/(b+np.exp(-test_matrix[var_index,j][:d]))
            x_test[3*d:(2+k)*d] = test_matrix[var_index,j][d:k*d]
            x_test[(2+k)*d:(2*k+1)*d] = 1/(a+test_matrix[var_index,j][d:k*d])
            x_test[(2*k+1)*d:(3*k)*d] = 1/(b+np.exp(-test_matrix[var_index,j][d:k*d]+np.tile(test_matrix[var_index,j][:d],k-1)))
            x_test[-1] = test_matrix[var_index,j][-1]
        
        x_test = x_test.reshape(-1,1)

        if node in [2,10]:
            out_test[cte:dlin + cte] = generate_dynamic_function(x_test, d, k, 2)[0]
            out_test[dlin + cte : dlin + dnonlin_2 + cte, :] = generate_dynamic_function(x_test, d, k, 2)[1]
            out_test[dlin + cte + dnonlin_2:, :] = generate_dynamic_function(x_test, d, k, 2)[2]
            f_library = generate_dynamic_function(x_test, d, k, 2)[3]

        elif node in [4,5,9,12,13]:
            out_test[cte:dlin + cte] = generate_dynamic_function(x_test, d, k, 3)[0]
            out_test[dlin + cte : dlin + dnonlin_2 + cte, :] = generate_dynamic_function(x_test, d, k, 3)[1]
            out_test[dlin + cte + dnonlin_2:, :] = generate_dynamic_function(x_test, d, k, 3)[2]
            f_library = generate_dynamic_function(x_test, d, k, 3)[3]
        
        elif node in [0,3,8,11]:
            out_test[cte:dlin + cte] = generate_dynamic_function(x_test, d, k, 4)[0]
            out_test[dlin + cte : dlin + dnonlin_2 + cte, :] = generate_dynamic_function(x_test, d, k, 4)[1]
            out_test[dlin + cte + dnonlin_2:, :] = generate_dynamic_function(x_test, d, k, 4)[2]
            f_library = generate_dynamic_function(x_test, d, k, 4)[3]            
        
        else:
            out_test[cte:dlin + cte] = generate_dynamic_function(x_test, d, k, 1)[0]
            out_test[dlin + cte : dlin + dnonlin_2 + cte, :] = generate_dynamic_function(x_test, d, k, 1)[1]
            out_test[dlin + cte + dnonlin_2:, :] = generate_dynamic_function(x_test, d, k, 1)[2]
            f_library = generate_dynamic_function(x_test, d, k, 1)[3]

        prediction = (W_out @ out_test[:])[0]

        if prediction < 0:
            prediction = 0

        test_matrix[node,j+1] = prediction

dic_pred = {'Time':time[train_length + 1:]}

for node in range(N):

    dic_pred['traj_{}'.format(col_fish[node])] = data_tseries[train_length + 1:,node]
    dic_pred['pred_{}'.format(col_fish[node])] = test_matrix[node,1:]

csv_out = pd.DataFrame(dic_pred)
csv_out.to_csv('../results/fish/fish_pred_p.csv',header = True)