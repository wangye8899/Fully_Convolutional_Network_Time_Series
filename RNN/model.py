import tensorflow as tf
from tensorflow.contrib import rnn
# 读配置文件，得到神经网络模型的参数

class model:
    def __init__(self,x,weights,biases,timesteps,num_hidden):
        self.x = x
        self.weights = weights
        self.biases = biases
        self.timesteps = timesteps
        self.num_hidden = num_hidden
    # 定义BIRNN神经网络结构
    def BiRNN(self):
    #     # 改变输入X的shape
        # print("dasda")
        x_ = tf.unstack(self.x,self.timesteps,1)
    #     # 前向传播
        # lstm_cell_fw = rnn.BasicLSTMCell(self.num_hidden)
        lstm_cell_fw = tf.nn.rnn_cell.LSTMCell(self.num_hidden)
        # lstm_cell_fw = tf.nn.rnn_cell.DropoutWrapper(lstm_cell_fw,output_keep_prob=0.5)
    #     # 反向传播
        lstm_cell_bw = tf.nn.rnn_cell.LSTMCell(self.num_hidden)
        # lstm_cell_bw = tf.nn.rnn_cell.DropoutWrapper(lstm_cell_bw,output_keep_prob=0.5)
        # lstm_cell_fw = tf.nn.rnn_cell.MultiRNNCell([lstm_cell() for i in range(2)])
        # lstm_cell_bw = tf.nn.rnn_cell.MultiRNNCell([lstm_cell() for i in range(2) ])
    
    #     # 产生输出
        try:
            outputs,_ ,_= rnn.static_bidirectional_rnn(lstm_cell_fw,lstm_cell_bw,x_,dtype=tf.float32)
        except Exception:
            outputs = rnn.static_bidirectional_rnn(lstm_cell_fw,lstm_cell_bw,x_,dtype=tf.float32)
    
        return tf.matmul(outputs[-1],self.weights['out'])+self.biases['out']     

    def MulTi_LSTM(self):
        x_ = tf.unstack(self.x,self.timesteps,1)
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.num_hidden)
        # lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell,output_keep_prob=0.5)
        # cell_list = [tf.nn.rnn_cell.LSTMCell(self.num_hidden) for _ in range(3)]
        # mul_rnn_lstm_cell = tf.nn.rnn_cell.MultiRNNCell(cell_list)
        outputs,states = rnn.static_rnn(lstm_cell,x_,dtype=tf.float32)
        return tf.matmul(outputs[-1],self.weights['out'])+self.biases['out']

    def Basic_Rnn(self):
        x_ = tf.unstack(self.x,self.timesteps,1)
        rnn_cell = tf.nn.rnn_cell.BasicRNNCell(self.num_hidden)
#         rnn_cell = tf.nn.rnn_cell.DropoutWrapper(rnn_cell,output_keep_prob=0.5)
        # cell_list = [ tf.nn.rnn_cell.BasicRNNCell(num_hidden) for _ in range(2)]    
        # mul_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(cell_list)
        # outputs,states = rnn.static_rnn(mul_rnn_cell,x,dtype=tf.float32)
        outputs , states = rnn.static_rnn(rnn_cell,x_,dtype=tf.float32)
        return tf.matmul(outputs[-1],self.weights['out'])+self.biases['out']


        
    def lstm_cell(self):
        cell = tf.nn.rnn_cell.LSTMCell(self.num_hidden)
        cell = tf.nn.rnn_cell.DropoutWrapper(cell,output_keep_prob=0.5)
        return cell



    def Fully_connected_network(self):
        layer_1 = tf.add(tf.matmul(self.x,self.weights['h1']),self.biases['b1'])
        layer_2 = tf.add(tf.matmul(layer_1,self.weights['h2']),self.biases['b2']) 
        out_layer = tf.add(tf.matmul(layer_2,self.weights['out']),self.biases['out'])
        return out_layer

   
   