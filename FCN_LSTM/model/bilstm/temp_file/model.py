import tensorflow as tf
import tensorflow.contrib as contrib
import numpy as np

class model_:
    def __init__(self,x,num_hiddens):
        self.x = x 
        self.num_hiddens = num_hiddens

    def modeling(self):
        # 对于模型来说，每次输入的数据格式为[batch_size,1024,30]
        # 也就是说对于一个LSTM循环体要展开1024个单体，每个单体处理长度为30的序列
        lstm_cell = contrib.rnn.BasicLSTMCell(self.num_hiddens,activation=tf.nn.tanh)
        predict,state = tf.nn.dynamic_rnn(lstm_cell,self.x,dtype=tf.float32)
        # 此时predict的维度是[batch_size,1024,num_hiddens]
        # 使用全连接层，将predict转换为label的格式，即[batch_size,1024,2]
        # predict = np.array(predict)
        predict = tf.reshape(predict,[-1,self.num_hiddens])
        fully_connected_layer_outputs = tf.layers.dense(predict,2,activation=tf.nn.relu)

        # 此时经过全连接层的线性变换，最终输出的形状为[None,2]
        return fully_connected_layer_outputs