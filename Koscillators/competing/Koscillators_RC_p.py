import numpy as np
import pandas as pd
import os

if not os.path.exists('../../results/Koscillators/competing'):
    os.makedirs('../../results/Koscillators/competing')

class ReservoirComputing:
    def __init__(self, input_size, reservoir_size, output_size, sparsity=0.1, input_scaling=0.1, spectral_radius=0.95):
        """
        初始化Reservoir Computing模型的参数
        """
        self.input_size = input_size
        self.reservoir_size = reservoir_size
        self.output_size = output_size
        self.sparsity = sparsity
        self.input_scaling = input_scaling
        self.spectral_radius = spectral_radius
        
        # 初始化水库权重和输入权重
        self.W_in = np.random.randn(self.reservoir_size, self.input_size) * self.input_scaling
        self.W_res = self._initialize_reservoir_weights(self.reservoir_size, self.sparsity)
        
        # 计算水库权重的谱半径
        rho_W = np.max(np.abs(np.linalg.eigvals(self.W_res)))
        self.W_res /= rho_W / self.spectral_radius
        
        # 初始化输出权重
        self.W_out = np.random.randn(self.output_size, self.reservoir_size)
        
        self.state = np.zeros(self.reservoir_size)

    def _initialize_reservoir_weights(self, size, sparsity):
        """
        初始化权重矩阵
        """
        W_res = np.random.randn(size, size)
        W_res[W_res < sparsity] = 0
        np.fill_diagonal(W_res, 0)
        return W_res
    
    def _update_state(self, input_data):
        """
        更新状态
        """
        self.state = np.tanh(np.dot(self.W_res, self.state) + np.dot(self.W_in, input_data))
    
    def train(self, X_train, Y_train, reg=1e-8):
        """
        训练模型 计算输出权重W_out
        """
        # 存储所有的水库状态
        states = np.zeros((X_train.shape[0], self.reservoir_size))
        
        # 通过输入数据更新水库状态
        for t in range(X_train.shape[0]):
            self._update_state(X_train[t])
            states[t] = self.state

        # 计算输出权重 W_out
        # 使用岭回归方法来计算W_out
        self.W_out = np.dot(np.linalg.pinv(states.T.dot(states) + reg * np.eye(self.reservoir_size)), np.dot(states.T, Y_train))

    def predict(self, X_current):
        """
        使用训练后的模型进行预测 基于当前输入预测下一个时间点的x值
        """
        self._update_state(X_current)
        return np.dot(self.W_out.T, self.state.reshape(-1, 1))

    def multi_step_predict(self, X_start, num_steps, p_series):
        """
        进行多步预测 每一步的预测结果作为下一步的输入
        """
        predictions = np.zeros((N,num_steps))
        last_x = X_start[:-1]  # 初始输入x值

        for t in range(num_steps):
            # 当前的输入包括x(t)和已知的p(t)，并预测x(t+1)
            current_input = np.concatenate((last_x.reshape(-1), p_series[t]))  # 输入是 x(t) 和 p(t)
            prediction = self.predict(current_input)  # 预测 x(t+1)
            predictions[:,t] = prediction.reshape(-1) # 保存预测的值
            
            # 更新 last_x 为新的预测值，用于下一步的预测
            last_x = prediction
        
        return np.array(predictions)  # 返回所有预测值

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
    t_series = df_tseries['t'].values
    p_series = df_tseries['p'].values

    length = len(data_tseries)

    # 设置训练和测试数据的长度
    train_length = 300
    test_length = length - train_length - 1

    train_x_tseries = np.zeros((N,train_length))

    for node in range(N):
        train_x_tseries[node,:] = data_tseries[:train_length,node]

    t_series = t_series.reshape(-1,1)

    # 重塑输入和目标序列为列向量
    x_tseries = data_tseries.reshape(-1, N)
    p_series = p_series.reshape(-1, 1)

    # 训练数据（前train_length部分）
    X_train = np.hstack([x_tseries[:train_length], p_series[:train_length]])  # 输入是(x, t)
    Y_train = x_tseries[1:train_length + 1]  # 目标是预测下一时刻的x

    # 测试数据（后test_length部分）
    X_test = np.hstack([x_tseries[train_length:-1], p_series[train_length:-1]])  # 输入是(x, t)
    Y_test = x_tseries[train_length + 1:]  # 目标是预测下一时刻的x

    # 初始化RC模型
    input_size = X_train.shape[1]
    reservoir_size = 1000
    output_size = Y_train.shape[1]

    rc = ReservoirComputing(input_size, reservoir_size, output_size)

    # 训练模型
    rc.train(X_train, Y_train, reg=1e-8)

    # 从测试数据中获取初始输入
    initial_input = X_test[0]

    # 使用多步预测方法进行预测
    predict_traj = rc.multi_step_predict(initial_input, num_steps=test_length, p_series=p_series[train_length:])

    dic_pred = {'Time':t_series[train_length + 1:,0]}

    for node in range(N):
        dic_pred['trajx_{}'.format(node)] = data_tseries[train_length + 1:,node]
        dic_pred['predx_{}'.format(node)] = predict_traj[node,:]

    pred_out = pd.DataFrame(dic_pred)
    pred_out.to_csv('../../results/Koscillators/competing/Koscillators_pred_RCp_{}.csv'.format(rand_seed),header = True, index=False)