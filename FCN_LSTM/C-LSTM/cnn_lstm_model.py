import tensorflow as tf
import numpy as np

class CNN_LSTM:
    
    def __init__(self,x,weights,biases,num_filters):
        self.x = x 
        self.weights = weights
        self.biases = biases
        self.num_filters = num_filters

    def CNN_(self):
        x = tf.transpose(self.x,[0,2,1])
        print(x.get_shape)
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
        # # 第三层卷积
        conv3 = tf.nn.conv1d(conv2,self.weights['conv3'],1,'SAME')
        conv3 = tf.nn.bias_add(conv3,self.biases['conv3'])
        conv3 = tf.keras.layers.BatchNormalization()(conv3)
        conv3 = tf.nn.relu(conv3)
        print(np.array(conv3))
        # 第四层
        # conv4 = tf.nn.conv1d(conv3,self.weights['conv4'],1,'SAME')
        # conv4 = tf.nn.bias_add(conv4,self.biases['conv4'])
        # conv4 = tf.keras.layers.BatchNormalization()(conv4)
        # conv4 = tf.nn.relu(conv4)
        # print(np.array(conv4))

        pool = tf.keras.layers.GlobalAveragePooling1D()(conv3)
        print(np.array(pool))
        return pool

    def LSTM_(self):
        cnn_out = self.CNN_()
        lstm = tf.keras.layers.LSTM(self.num_filters)(cnn_out)
        return lstm

    def Return_out(self):
        lstm_out = self.LSTM_()
        cnn_out = self.CNN_()
        return_out = tf.add(tf.matmul(cnn_out,self.weights['out_w']),self.biases['out_b'])

        return return_out