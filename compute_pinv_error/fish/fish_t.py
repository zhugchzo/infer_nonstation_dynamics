import numpy as np
import pandas as pd
import os

def encode_date_tag(tag,min,delta): # this function is used to transform date to time variable

    month, part = tag.split('-')[-2], tag.split('-')[-1]
    month = int(month)  

    if 2 <= month < 8:
        if part == "early":
            return (month - 2) * 2*delta + min  # early
        elif part == "late":
            return (month - 2) * 2*delta + min + delta  # late
        
    elif 9 <= month <= 12:
        if part == "early":
            return -(month - 8) * 2*delta + min + 12*delta  # early
        elif part == "late":
            return -(month - 8) * 2*delta + min + 11*delta  # late

    elif month == 1:
        if part == "early":
            return min + 2*delta  # early
        elif part == "late":
            return min + 1*delta  # late 
            
    elif month == 8:
        if part == "early":
            return min + 12*delta  # early
        elif part == "late":
            return min + 11*delta  # late
        
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

# Load the saved data
df_tseries = pd.read_csv('../../fish/fish_data.csv')
df_network = pd.read_csv('../../fish/fish_network.csv')

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
t_tseries = (df_tseries['date_tag'].apply(lambda x: encode_date_tag(x, min=0, delta=0.5))).values

length = len(data_tseries)

time = np.arange(1, 1 + length, 1)

train_length = 254
test_length = length - train_length - 1

period = 24

a = 1
b = 1

theta_train = t_tseries[:train_length+1]
theta_valid = theta_train.copy()
theta_train = theta_train.reshape(-1, 1)

#####################################################################################

pinv_error = 0

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
    
    # compute the pseudo-inverse matrix error

    pinv_error_matrix = np.linalg.pinv(out_train[:,:] @ out_train[:,:].T) @ (out_train[:,:] @ out_train[:,:].T) - np.identity(out_train.shape[0])
    node_pinv_error = np.sum(pinv_error_matrix**2) / out_train.shape[0]**2

    pinv_error += node_pinv_error / N

print('time pinv error: {}'.format(pinv_error))