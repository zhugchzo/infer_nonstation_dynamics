import numpy as np
import pandas as pd
import sympy as sp

def encode_date_tag(tag,min,delta):

    month, part = tag.split('-')[-2], tag.split('-')[-1]
    month = int(month)  

    if 2 <= month < 8:
        if part == 'early':
            return (month - 2) * 2*delta + min  # early
        elif part == 'late':
            return (month - 2) * 2*delta + min + delta  # late
        
    elif 9 <= month <= 12:
        if part == 'early':
            return -(month - 8) * 2*delta + min + 12*delta  # early
        elif part == 'late':
            return -(month - 8) * 2*delta + min + 11*delta  # late

    elif month == 1:
        if part == 'early':
            return min + 2*delta  # early
        elif part == 'late':
            return min + 1*delta  # late 
            
    elif month == 8:
        if part == 'early':
            return min + 12*delta  # early
        elif part == 'late':
            return min + 11*delta  # late

col = ['Aurelia.sp', 'Plotosus.japonicus', 'Sebastes.cheni', 'Trachurus.japonicus', 'Girella.punctata',
       'Pseudolabrus.sieboldi', 'Parajulis.poecilopterus', 'Halichoeres.tenuispinnis', 'Chaenogobius.gulosus',
       'Pterogobius.zonoleucus', 'Tridentiger.trigonocephalus', 'Siganus.fuscescens', 'Sphyraena.pinguis', 'Rudarius.ercodes']

N = len(col)
length = 285

df_tseries = pd.read_csv('fish_data.csv')
df_network = pd.read_csv('fish_network.csv')

data_tseries = df_tseries[col].values
data_network = df_network[col].values

initial_theta = 0
delta_theta = 1
encoded_pars = (df_tseries['date_tag'].apply(lambda x: encode_date_tag(x, min=initial_theta, delta=delta_theta))).values
theta_list = encoded_pars

dom_eval_list = list()

for t in range(length):

    jacob_matrix = np.zeros((N,N))

    for node in range(N):
        
        inter_var_index = list(np.where(data_network[:,node] == 1)[0])
        k = len(inter_var_index)

        df_W_out = pd.read_csv('../results/fish/equations/{}_{}.csv'.format(node,col[node]))

        columns_list = df_W_out.columns.tolist()
        W_out = df_W_out.values
        W_out = W_out.reshape(-1)

        if node in [0]:

            x, y, p = sp.symbols('x y p')
            vector = {'cte':1, 'x_1':x, 'p':p, 'y_1':1/(1+sp.exp(-x)), 'y_2':y, 'y_3':1/(1+y)}

            f_terms = []
            for term in columns_list:
                factors = term.split('*')
                product = 1
                for factor in factors:
                    if '^' in factor:
                        base, exp = factor.split('^')
                        exp = int(exp)
                        product *= vector[base]**exp
                    else:
                        product *= vector[factor]
                f_terms.append(product)

            f_terms = np.array(f_terms)
            f = np.dot(W_out,f_terms)

            df_dx = sp.diff(f, x)
            df_dy = sp.diff(f, y)
        
            x_t = data_tseries[t,node]
            y_t = data_tseries[t,inter_var_index[0]]
            p_t = theta_list[t]

            df_dx_value = df_dx.subs({x:x_t, y:y_t, p:p_t})
            df_dy_value = df_dy.subs({x:x_t, y:y_t, p:p_t})

            jacob_matrix[node,node] = df_dx_value
            jacob_matrix[node,inter_var_index[0]] = df_dy_value

        elif node in [2,10]:
            
            x, y, p = sp.symbols('x y p')
            vector = {'cte':1, 'x_1':x, 'p':p, 'y_1':1/(1+x), 'y_2':y, 'y_3':1/(1+y), 'y_4':1/(1+sp.exp(-y+x))}

            f_terms = []
            for term in columns_list:
                factors = term.split('*') 
                product = 1
                for factor in factors:
                    if '^' in factor:
                        base, exp = factor.split('^')
                        exp = int(exp)
                        product *= vector[base]**exp
                    else:
                        product *= vector[factor]
                f_terms.append(product)

            f_terms = np.array(f_terms)
            f = np.dot(W_out,f_terms)

            df_dx = sp.diff(f, x)
            df_dy = sp.diff(f, y)
        
            x_t = data_tseries[t,node]
            y_t = data_tseries[t,inter_var_index[0]]
            p_t = theta_list[t]

            df_dx_value = df_dx.subs({x:x_t, y:y_t, p:p_t})
            df_dy_value = df_dy.subs({x:x_t, y:y_t, p:p_t})

            jacob_matrix[node,node] = df_dx_value
            jacob_matrix[node,inter_var_index[0]] = df_dy_value

        elif node in [5]:
            
            x, y, z, p = sp.symbols('x y z p')
            vector = {'cte':1, 'x_1':x, 'p':p, 'y_1':1/(1+x), 'y_2':1+(1+sp.exp(-x)), 'y_3':y, 'y_4':z, 'y_5':1/(1+sp.exp(-y+x)), 'y_6':1/(1+sp.exp(-z+x))}

            f_terms = []
            for term in columns_list:
                factors = term.split('*')
                product = 1
                for factor in factors:
                    if '^' in factor:
                        base, exp = factor.split('^')
                        exp = int(exp)
                        product *= vector[base]**exp
                    else:
                        product *= vector[factor]
                f_terms.append(product)

            f_terms = np.array(f_terms)
            f = np.dot(W_out,f_terms)

            df_dx = sp.diff(f, x)
            df_dy = sp.diff(f, y)
            df_dz = sp.diff(f ,z)
        
            x_t = data_tseries[t,node]
            y_t = data_tseries[t,inter_var_index[0]]
            z_t = data_tseries[t,inter_var_index[1]]
            p_t = theta_list[t]

            df_dx_value = df_dx.subs({x:x_t, y:y_t, z:z_t, p:p_t})
            df_dy_value = df_dy.subs({x:x_t, y:y_t, z:z_t, p:p_t})
            df_dz_value = df_dz.subs({x:x_t, y:y_t, z:z_t, p:p_t})

            jacob_matrix[node,node] = df_dx_value
            jacob_matrix[node,inter_var_index[0]] = df_dy_value
            jacob_matrix[node,inter_var_index[1]] = df_dz_value

        elif node in [3,8]:

            x, y, z, p = sp.symbols('x y z p')
            vector = {'cte':1, 'x_1':x, 'p':p, 'y_1':1/(1+sp.exp(-x)), 'y_2':y, 'y_3':z, 'y_4':1/(1+sp.exp(-y+x)), 'y_5':1/(1+sp.exp(-z+x))}

            f_terms = []
            for term in columns_list:
                factors = term.split('*')
                product = 1
                for factor in factors:
                    if '^' in factor:
                        base, exp = factor.split('^')
                        exp = int(exp)
                        product *= vector[base]**exp
                    else:
                        product *= vector[factor]
                f_terms.append(product)

            f_terms = np.array(f_terms)
            f = np.dot(W_out,f_terms)

            df_dx = sp.diff(f, x)
            df_dy = sp.diff(f, y)
            df_dz = sp.diff(f ,z)
        
            x_t = data_tseries[t,node]
            y_t = data_tseries[t,inter_var_index[0]]
            z_t = data_tseries[t,inter_var_index[1]]
            p_t = theta_list[t]

            df_dx_value = df_dx.subs({x:x_t, y:y_t, z:z_t, p:p_t})
            df_dy_value = df_dy.subs({x:x_t, y:y_t, z:z_t, p:p_t})
            df_dz_value = df_dz.subs({x:x_t, y:y_t, z:z_t, p:p_t})

            jacob_matrix[node,node] = df_dx_value
            jacob_matrix[node,inter_var_index[0]] = df_dy_value
            jacob_matrix[node,inter_var_index[1]] = df_dz_value

        elif node in [6]:

            x, y, z, p = sp.symbols('x y z p')
            vector = {'cte':1, 'x_1':x, 'p':p, 'y_1':1/(1+x), 'y_2':1/(1+sp.exp(-x)), 'y_3':y, 'y_4':z,
                    'y_5':1/(1+y), 'y_6':1/(1+z), 'y_7':1/(1+sp.exp(-y+x)), 'y_8':1/(1+sp.exp(-z+x))}

            f_terms = []
            for term in columns_list:
                factors = term.split('*')
                product = 1
                for factor in factors:
                    if '^' in factor:
                        base, exp = factor.split('^')
                        exp = int(exp)
                        product *= vector[base]**exp
                    else:
                        product *= vector[factor]
                f_terms.append(product)

            f_terms = np.array(f_terms)
            f = np.dot(W_out,f_terms)

            df_dx = sp.diff(f, x)
            df_dy = sp.diff(f, y)
            df_dz = sp.diff(f ,z)
        
            x_t = data_tseries[t,node]
            y_t = data_tseries[t,inter_var_index[0]]
            z_t = data_tseries[t,inter_var_index[1]]
            p_t = theta_list[t]

            df_dx_value = df_dx.subs({x:x_t, y:y_t, z:z_t, p:p_t})
            df_dy_value = df_dy.subs({x:x_t, y:y_t, z:z_t, p:p_t})
            df_dz_value = df_dz.subs({x:x_t, y:y_t, z:z_t, p:p_t})

            jacob_matrix[node,node] = df_dx_value
            jacob_matrix[node,inter_var_index[0]] = df_dy_value
            jacob_matrix[node,inter_var_index[1]] = df_dz_value

        elif node in [9,12,13]:

            x, y, p = sp.symbols('x y p')
            vector = {'cte':1, 'x_1':x, 'p':p, 'y_1':1/(1+x), 'y_2':1/(1+sp.exp(-x)), 'y_3':y, 'y_4':1/(1+sp.exp(-y+x))}

            f_terms = []
            for term in columns_list:
                factors = term.split('*')
                product = 1
                for factor in factors:
                    if '^' in factor:
                        base, exp = factor.split('^')
                        exp = int(exp)
                        product *= vector[base]**exp
                    else:
                        product *= vector[factor]
                f_terms.append(product)

            f_terms = np.array(f_terms)
            f = np.dot(W_out,f_terms)

            df_dx = sp.diff(f, x)
            df_dy = sp.diff(f, y)
        
            x_t = data_tseries[t,node]
            y_t = data_tseries[t,inter_var_index[0]]
            p_t = theta_list[t]

            df_dx_value = df_dx.subs({x:x_t, y:y_t, p:p_t})
            df_dy_value = df_dy.subs({x:x_t, y:y_t, p:p_t})

            jacob_matrix[node,node] = df_dx_value
            jacob_matrix[node,inter_var_index[0]] = df_dy_value

    evals = np.linalg.eigvals(jacob_matrix)
    evals = [lam for lam in evals]
    dom_eval = max(evals,key=abs)
    dom_eval_module = abs(dom_eval)

    dom_eval_list.append(dom_eval_module)

time = [i for i in range(0,length)]

dic_dom_eval = {'Time':time, 'dom_eval':dom_eval_list}

csv_out = pd.DataFrame(dic_dom_eval)
csv_out.to_csv('../results/fish/dom_eval.csv', header = True, index = False)

    


