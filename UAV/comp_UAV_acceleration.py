import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def comp_acc(coef_x,coef_y,x,y,p):

    dxdt = (coef_x[0] + coef_x[1]*x + coef_x[2]*y + coef_x[3]*p + 
            coef_x[4]*x**2 + coef_x[5]*x*y + coef_x[6]*p*x + coef_x[7]*y**2 + 
            coef_x[8]*p*y + coef_x[9]*p**2 + coef_x[10]*x**3 + coef_x[11]*x**2*y + 
            coef_x[12]*p*x**2 + coef_x[13]*x*y**2 + coef_x[14]*p*x*y + coef_x[15]*p**2*x +
            coef_x[16]*y**3 + coef_x[17]*p*y**2 + coef_x[18]*p**2*y + coef_x[19]*p**3)
    
    dydt = (coef_y[0] + coef_y[1]*x + coef_y[2]*y + coef_y[3]*p + 
            coef_y[4]*x**2 + coef_y[5]*x*y + coef_y[6]*p*x + coef_y[7]*y**2 + 
            coef_y[8]*p*y + coef_y[9]*p**2 + coef_y[10]*x**3 + coef_y[11]*x**2*y + 
            coef_y[12]*p*x**2 + coef_y[13]*x*y**2 + coef_y[14]*p*x*y + coef_y[15]*p**2*x +
            coef_y[16]*y**3 + coef_y[17]*p*y**2 + coef_y[18]*p**2*y + coef_y[19]*p**3)
    
    fx = (coef_x[1] + 2*coef_x[4]*x + coef_x[5]*y + coef_x[6]*p + 3*coef_x[10]*x**2 +
          2*coef_x[11]*x*y + 2*coef_x[12]*p*x + coef_x[13]*y**2 + coef_x[14]*p*y + coef_x[15]*p**2)

    fy = (coef_x[2] + coef_x[5]*x + 2*coef_x[7]*y + coef_x[8]*p + coef_x[11]*x**2 + 
          2*coef_x[13]*x*y + coef_x[14]*p*x + 3*coef_x[16]*y**2 + 2*coef_x[17]*p*y + coef_x[18]*p**2)
    
    gx = (coef_y[1] + 2*coef_y[4]*x + coef_y[5]*y + coef_y[6]*p + 3*coef_y[10]*x**2 +
          2*coef_y[11]*x*y + 2*coef_y[12]*p*x + coef_y[13]*y**2 + coef_y[14]*p*y + coef_y[15]*p**2)

    gy = (coef_y[2] + coef_y[5]*x + 2*coef_y[7]*y + coef_y[8]*p + coef_y[11]*x**2 +
          2*coef_y[13]*x*y + coef_y[14]*p*x + 3*coef_y[16]*y**2 + 2*coef_y[17]*p*y + coef_y[18]*p**2)
    
    acc_x = fx*dxdt + fy*dydt
    acc_y = gx*dxdt + gy*dydt

    acc = np.sqrt(acc_x**2 + acc_y**2)
    acc_x_norm = acc_x / acc
    acc_y_norm = acc_y / acc

    return acc_x_norm, acc_y_norm

# Load the saved data
data_file = pd.read_csv('UAV_data.csv')
data = data_file[['Time','x','y']].values

# Extract time and positions
t_series = data[:, 0]
x_tseries = data[:,1:3]

length = len(t_series)

initial_p = 1
delta_p = 5e-3

p_tseries = np.linspace(initial_p, initial_p + (length - 1) * delta_p, length)

# Load the equations

df_coef = pd.read_csv('../results/UAV/UAV_W_out.csv')

f1_elements = ['cte_f1', 'x_f1', 'y_f1', 'p_f1', 'x^2_f1', 'xy_f1', 'px_f1', 'y^2_f1', 'py_f1', 'p^2_f1', 'x^3_f1', 'x^2y_f1', 'px^2_f1', 'xy^2_f1', 'pxy_f1', 'p^2x_f1', 'y^3_f1', 'py^2_f1', 'p^2y_f1', 'p^3_f1']
f2_elements = ['cte_f2', 'x_f2', 'y_f2', 'p_f2', 'x^2_f2', 'xy_f2', 'px_f2', 'y^2_f2', 'py_f2', 'p^2_f2', 'x^3_f2', 'x^2y_f2', 'px^2_f2', 'xy^2_f2', 'pxy_f2', 'p^2x_f2', 'y^3_f2', 'py^2_f2', 'p^2y_f2', 'p^3_f2']

coef_x = df_coef[f1_elements].to_numpy()[0]
coef_y = df_coef[f2_elements].to_numpy()[0]

accx_list = list()
accy_list = list()

for i in range(length):

    accx = comp_acc(coef_x,coef_y,x_tseries[i,0],x_tseries[i,1],p_tseries[i])[0]
    accy = comp_acc(coef_x,coef_y,x_tseries[i,0],x_tseries[i,1],p_tseries[i])[1]
    
    accx_list.append(accx)
    accy_list.append(accy)

dic_acc = {'Time':t_series,'accx':accx_list,'accy':accy_list}

csv_out = pd.DataFrame(dic_acc)
csv_out.to_csv('../results/UAV/UAV_acc.csv',header = True)

plt.plot(x_tseries[:,0],x_tseries[:,1])

# for i in range(1,length-1,11):
#     plt.quiver(x_tseries[i,0], x_tseries[i,1], ax_list[i], ax_list[i], angles='xy', scale_units='xy', scale=None, color='red', width=0.005)

for i in range(1,length-1,11):
    plt.quiver(x_tseries[i,0], x_tseries[i,1], accx_list[i], accy_list[i], angles='xy', scale_units='xy', scale=None, color='red', width=0.001)

plt.quiver(x_tseries[0,0], x_tseries[0,1], accx_list[0], accy_list[0], angles='xy', scale_units='xy', scale=None, color='red', width=0.001)
plt.quiver(x_tseries[-1,0], x_tseries[-1,1], accx_list[-1], accy_list[-1], angles='xy', scale_units='xy', scale=None, color='red', width=0.001)

# plt.xlim(-1,11.5)
# plt.ylim(-5.25,7.25)
plt.show()