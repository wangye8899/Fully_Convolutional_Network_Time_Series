import tensorflow as tf
import tensorflow.contrib as contrib
import numpy as np

class model_:
    def __init__(self,x,num_hiddens,weights,biases):
        self.x = x
        # self.x = tf.unstack(x,256,1) 
        self.num_hiddens = num_hiddens
        self.weights = weights
        self.biases = biases


    def modeling(self):
        # x_conv_input = tf.reshape(self.x,[None,30,1])
        # # 对于雷达信号数据，先使用一维卷积提取特征，将处理过后的特征映射输入至lstm中
        # conv1d = tf.nn.conv1d(x_conv_input,self.weights['conv1'],1,'SAME')
        # conv1d = tf.nn.bias_add(conv1d,self.biases['conv1'])
        # conv1d = tf.keras.layers.BatchNormalization()(conv1d)
        # conv1d = tf.nn.relu(conv1d)
        # conv1d_out = tf.matmul(conv1d,self.weights['conv1_out'])+self.biases['conv1_out']
        
        # # 修改维度
        # conv1d_out = tf.reshape(conv1d_out,[None,30,1])
        


        # 对于模型来说，每次输入的数据格式为[batch_size,256,30]
        # 也就是说对于一个LSTM循环体要展开256个单体，每个单体处理长度为30的序列
        # lstm_cell = contrib.rnn.LSTMCell(self.num_hiddens,initializer=tf.orthogonal_initializer(),activation=tf.nn.tanh)
        # lstm_cell = contrib.rnn.DropoutWrapper(cell=lstm_cell,input_keep_prob=1.0,output_keep_prob=0.8)
        lstm_cell = contrib.rnn.BasicLSTMCell(self.num_hiddens,activation=tf.nn.tanh)
        # multi_lstm = contrib.rnn.MultiRNNCell([contrib.rnn.LSTMCell(self.num_hiddens,initializer=tf.orthogonal_initializer(),activation=tf.nn.tanh) for i in range(3)])
        predict,state = tf.nn.dynamic_rnn(lstm_cell,self.x,dtype=tf.float32)
        # 此时predict的维度是[batch_size,256,num_hiddens]
        # 使用全连接层，将predict转换为label的格式，即[batch_size,1024,2]
        # predict = np.array(predict)
        predict = tf.reshape(predict,[-1,self.num_hiddens])
        fully_connected_layer_outputs = tf.layers.dense(predict,2)
        
        
        
        # 双向循环神经网络 结构代码
        # lstm_fw_cell = contrib.rnn.LSTMCell(self.num_hiddens,initializer=tf.orthogonal_initializer(),activation=tf.nn.tanh,forget_bias=1.0)
        # lstm_bw_cell = contrib.rnn.LSTMCell(self.num_hiddens,initializer=tf.orthogonal_initializer(),activation=tf.nn.tanh,forget_bias=1.0)
        # outputs ,_,_ = contrib.rnn.static_bidirectional_rnn(lstm_fw_cell,lstm_bw_cell,self.x,dtype=tf.float32)
        # print(np.array(outputs).shape)
        # fully_connected_layer_outputs = tf.add(tf.matmul(outputs[-1],self.weights['out']),self.biases['out'])       
        return fully_connected_layer_outputs