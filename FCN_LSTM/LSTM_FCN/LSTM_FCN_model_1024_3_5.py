import  tensorflow as tf
import numpy as np
import matplotlib.pyplot as mlp
import tensorflow.keras as keras 
from tensorflow.contrib import rnn
from tensorflow.keras.models import Model , Sequential

class LSTM_FCN:
    
    def __init__(self,x,weights,biases,num_hidden):
        self.x = x
        self.weights = weights
        self.biases = biases
        self.num_hidden = num_hidden 
        pass

    def FCN_(self):
        # 单个变量，多个时间步处理
        x = tf.transpose(self.x,[0,2,1])
        print(np.array(x)) 
        # 第一层卷积
        conv1 = tf.nn.conv1d(x,self.weights['conv1'],1,'SAME')
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
        # conv4 = tf.keras.layers.BatchNormalization()(conv4)
        # conv4 = tf.nn.relu(conv4)

        # conv5 = tf.nn.conv1d(conv4,self.weights['conv5'],1,'SAME')
        # conv5 = tf.nn.bias_add(conv5,self.biases['conv5'])
        # conv5 = tf.keras.layers.BatchNormalization()(conv5)
        # conv5 = tf.nn.relu(conv5)

        # conv6 = tf.nn.conv1d(conv5,self.weights['conv6'],1,'SAME')
        # conv6 = tf.nn.bias_add(conv6,self.biases['conv6'])
        # conv6 = tf.keras.layers.BatchNormalization()(conv6)
        # conv6 = tf.nn.relu(conv6)
        
        
        FCN_out = tf.keras.layers.GlobalAveragePooling1D()(conv3)
        print("全局池化之后的形状")
        # print(np.array(FCN_out))
        # FCN_out = tf.add(tf.matmul(FCN_out,self.weights['out_w']),self.biases['out_b'])
        return FCN_out

    def LSTM_(self):
        print(self.num_hidden)

        x = tf.transpose(self.x,[0,2,1])
        lstm = tf.keras.layers.LSTM(self.num_hidden)(self.x)
        # lstm = tf.keras.layers.Dropout(0.8)(lstm)
        # lstm = tf.nn.rnn_cell.BasicLSTMCell(self.num_hidden)
        # lstm = tf.nn.rnn_cell.DropoutWrapper(lstm,output_keep_prob=0.8)
        # lstm_out , _ = tf.nn.dynamic_rnn(lstm,self.x,dtype=tf.float32)
        print("lstm处理之后产生的形状")
        print(np.array(lstm))
        # lstm_out = tf.add(tf.matmul(lstm_out[-1],self.weights['out_w']),self.biases['out_b'])
        return lstm

    def connect_FCN_LSTM(self):
        # fcn_out = self.FCN_()
        lstm_out = self.LSTM_()
        # print("测试一下")
        # print(fcn_out)
        # print(lstm_out)
        # connect_out = tf.concat([fcn_out,lstm_out],1)
        # print("连接后的矩阵形状")
        # print(np.array(connect_out))
        # print("最终的形状")
        # print(np.array(tf.matmul(connect_out,self.weights['out_w'])))
        connect_out = tf.add(tf.matmul(lstm_out,self.weights['out_w']),self.biases['out_b'])
        return connect_out

    def Bi_LSTM_(self):
        model = Sequential()
        model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.num_hidden)))(self.x)
        return model
        x = model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.num_hidden)))(self.x)
        return x
    # def ResNet_(self):

