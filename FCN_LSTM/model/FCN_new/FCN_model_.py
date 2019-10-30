import  tensorflow as tf
import numpy as np
import matplotlib.pyplot as mlp
import tensorflow.keras as keras 
from tensorflow.contrib import rnn
from tensorflow.keras.models import Model , Sequential
import tensorflow as tf
import tensorflow.contrib as contrib
import numpy as np
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn

class LSTM_FCN:
    
    def __init__(self,x,weights,biases,num_hidden):
        self.x = x
        self.weights = weights
        self.biases = biases
        self.num_hidden = num_hidden 
        pass

    def FCN_(self):
        # 单个变量，多个时间步处理
        # x = tf.transpose(self.x,[0,2,1])
        # print(np.array(x))
        # 第一层卷积
        print(np.array(self.x))
        conv1 = tf.nn.conv1d(self.x,self.weights['conv1'],1,'SAME')
        conv1 = tf.nn.bias_add(conv1,self.biases['conv1'])
        conv1 = tf.keras.layers.BatchNormalization()(conv1)
        conv1 = tf.nn.relu(conv1)
        print(np.array(conv1))
        # 第二层卷积
        conv2 = tf.nn.conv1d(conv1,self.weights['conv2'],1,'SAME')
        conv2 = tf.nn.bias_add(conv2,self.biases['conv2'])
        conv2 = tf.keras.layers.BatchNormalization()(conv2)
        conv2 = tf.nn.relu(conv2)
        print(np.array(conv2))
        # 第三层卷积
        conv3 = tf.nn.conv1d(conv2,self.weights['conv3'],1,'SAME')
        conv3 = tf.nn.bias_add(conv3,self.biases['conv3'])
        conv3 = tf.keras.layers.BatchNormalization()(conv3)
        conv3 = tf.nn.relu(conv3)
        print(np.array(conv3))
        # 第四层卷积

        # conv4 = tf.nn.conv1d(conv3,self.weights['conv4'],1,'SAME')
        # conv4 = tf.nn.bias_add(conv4,self.biases['conv4'])
        # conv4 = tf.nn.relu(conv4)
        # # # # # 第五层卷积
        # conv5 = tf.nn.conv1d(conv4,self.weights['conv5'],1,'SAME')
        # conv5 = tf.nn.bias_add(conv5,self.biases['conv5'])
        # # # 第六层卷积
        # conv6 = tf.nn.conv1d(conv5,self.weights['conv6'],1,'SAME')
        # conv6 = tf.nn.bias_add(conv6,self.biases['conv6'])
        # # # 全局池化层和softmax分类器
        # FCN_out = tf.keras.layers.GlobalAveragePooling1D()(conv5)
        # conv4 = tf.nn.conv1d(conv3,self.weights['conv4'],1,'SAME')
        # conv4 = tf.nn.bias_add(conv4,self.biases['conv4'])
        # conv4 = tf.nn.relu(conv4)
        # # 第五层卷积
        # conv5 = tf.nn.conv1d(conv4,self.weights['conv5'],1,'SAME')
        # conv5 = tf.nn.bias_add(conv5,self.biases['conv5'])
        # 全局池化层和softmax分类器
        FCN_out = tf.keras.layers.GlobalAveragePooling1D()(conv3)
        
        print("全局池化之后的形状")
        print(np.array(FCN_out))
        # FCN_out = tf.add(tf.matmul(FCN_out,self.weights['out_w']),self.biases['out_b'])
        return FCN_out

    def Bi_LSTM_(self):
        # 卷积的输出为[256,128]
        input_ = self.FCN_()
        input_ = tf.reshape(input_,[-1,256,128])
        for _ in range(3):
            with tf.variable_scope(None, default_name="bidirectional-rnn"):
                fw_lstm_cell = contrib.rnn.LSTMCell(self.num_hidden,initializer=tf.orthogonal_initializer())
                bw_lstm_cell = contrib.rnn.LSTMCell(self.num_hidden,initializer=tf.orthogonal_initializer())
                # 目前均不加Dropout层
                outputs,state = tf.nn.bidirectional_dynamic_rnn(fw_lstm_cell,bw_lstm_cell,input_,dtype=tf.float32)
                input_ = tf.concat(outputs,2) 

        #此时3层BLSTM输出的形状为 [?,256,256]
        # 使用全连接层进行分类
        input_reshape = tf.reshape(input_,[-1,256])

        return tf.matmul(input_reshape,self.weights['out_w'])+self.biases['out_b'] 
    