import tensorflow as tf
import tensorflow.contrib as contrib
import numpy as np

class model_:
    def __init__(self,x_cnn,x_lstm,num_hiddens,weights,biases):
        self.x_cnn = x_cnn
        self.x_lstm = x_lstm
        self.num_hiddens = num_hiddens
        self.weights = weights
        self.biases = biases
    def lstm_layers(self):
        input_x = self.x_lstm
        multi_lstm = contrib.rnn.MultiRNNCell([contrib.rnn.LSTMCell(self.num_hiddens,initializer=tf.orthogonal_initializer(),activation=tf.nn.tanh) for i in range(1)])
        predict,state = tf.nn.dynamic_rnn(multi_lstm,input_x,dtype=tf.float32)
        # 此时predict的维度是[batch_size,256,num_hiddens]
        # 使用全连接层，将predict转换为label的格式，即[batch_size,1024,2]
        # predict = np.array(predict)
        predict = tf.reshape(predict,[-1,self.num_hiddens])
        return predict
    def modeling(self):
        
        # x_conv_input = tf.transpose(self.x,[None,30,1])
        # 对于雷达信号数据，先使用一维卷积提取特征，将处理过后的特征映射输入至lstm中
        # 第一层卷积
        conv1d = tf.nn.conv1d(self.x_cnn,self.weights['conv1'],1,'SAME')
        conv1d = tf.nn.bias_add(conv1d,self.biases['conv1'])
        conv1d = tf.keras.layers.BatchNormalization()(conv1d)
        conv1d = tf.nn.relu(conv1d)
        # 第二层卷积
        conv2d = tf.nn.conv1d(conv1d,self.weights['conv2'],1,'SAME')
        conv2d = tf.nn.bias_add(conv2d,self.biases['conv2'])
        conv2d = tf.keras.layers.BatchNormalization()(conv2d)
        # 第三层卷积
        conv3d = tf.nn.conv1d(conv2d,self.weights['conv3'],1,'SAME')
        conv3d = tf.nn.bias_add(conv3d,self.biases['conv3'])
        conv3d = tf.keras.layers.BatchNormalization()(conv3d)
        conv3d = tf.nn.relu(conv3d)
        # print(conv1d)
        after_pooling_conv3d = tf.keras.layers.GlobalAveragePooling1D()(conv3d)
        # 连接
        conv1d_out = tf.reshape(after_pooling_conv3d,[256,128])
        lstm_out = self.lstm_layers()
        predict = tf.concat([conv1d_out,lstm_out],1)
        fully_connected_layer_outputs = tf.layers.dense(predict,2)
        
        
        
        # 双向循环神经网络 结构代码
        # lstm_fw_cell = contrib.rnn.LSTMCell(self.num_hiddens,initializer=tf.orthogonal_initializer(),activation=tf.nn.tanh,forget_bias=1.0)
        # lstm_bw_cell = contrib.rnn.LSTMCell(self.num_hiddens,initializer=tf.orthogonal_initializer(),activation=tf.nn.tanh,forget_bias=1.0)
        # outputs ,_,_ = contrib.rnn.static_bidirectional_rnn(lstm_fw_cell,lstm_bw_cell,self.x,dtype=tf.float32)
        # print(np.array(outputs).shape)
        # fully_connected_layer_outputs = tf.add(tf.matmul(outputs[-1],self.weights['out']),self.biases['out'])       
        return fully_connected_layer_outputs

    
