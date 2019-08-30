# 单独的python文件，构建全卷积神经网络
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras


class FCN_model:
   

    def __init__(self,x,weights,biases,strides,num_channels):
        self.x = x
        self.num_channels = num_channels
        self.weights = weights
        self.biases = biases
        self.strides = strides
    
    def Conv2D(self):
        # 第一层卷积层
        conv1_layer = tf.nn.conv2d(self.x,self.weights['conv1_w'],strides=[1,self.strides,self.strides,1],padding='SAME')
<<<<<<< HEAD:FCN/FCN_model.py
        conv1_layer = tf.nn.dropout(conv1_layer,0.8)
=======
        # conv1_layer = tf.nn.dropout(conv1_layer,0.8)
>>>>>>> 90200e08ccc8078e6456de3c9d879e4da2e082cf:FCN_model.py
        conv1_layer = tf.nn.bias_add(conv1_layer,self.biases['conv1_b'])
        conv1_layer = keras.layers.BatchNormalization()(conv1_layer)
        conv1_layer = tf.nn.relu(conv1_layer)
       
        # 第二层卷积层
        conv2_layer = tf.nn.conv2d(conv1_layer,self.weights['conv2_w'],strides=[1,self.strides,self.strides,1],padding='SAME')
<<<<<<< HEAD:FCN/FCN_model.py
        conv2_layer = tf.nn.dropout(conv2_layer,0.8)
=======
        # conv2_layer = tf.nn.dropout(conv2_layer,0.8)
>>>>>>> 90200e08ccc8078e6456de3c9d879e4da2e082cf:FCN_model.py
        conv2_layer = tf.nn.bias_add(conv2_layer,self.biases['conv2_b'])
        conv2_layer = keras.layers.BatchNormalization()(conv2_layer)
        conv2_layer = tf.nn.relu(conv2_layer) 

        # 第三层卷积层
        conv3_layer = tf.nn.conv2d(conv2_layer,self.weights['conv3_w'],strides=[1,self.strides,self.strides,1],padding='SAME')
<<<<<<< HEAD:FCN/FCN_model.py
        conv3_layer = tf.nn.dropout(conv3_layer,0.8)
=======
        # conv3_layer = tf.nn.dropout(conv3_layer,0.8)
>>>>>>> 90200e08ccc8078e6456de3c9d879e4da2e082cf:FCN_model.py
        conv3_layer = tf.nn.bias_add(conv3_layer,self.biases['conv3_b']) 
        conv3_layer = keras.layers.BatchNormalization()(conv3_layer)
        conv3_layer = tf.nn.relu(conv3_layer)

        # 第四层卷积层
        # conv4_layer = tf.nn.conv2d(conv3_layer,self.weights['conv4_w'],strides=[1,self.strides,self.strides,1],padding='SAME')
        # conv4_layer = tf.nn.bias_add(conv4_layer,self.biases['conv4_b'])
        # conv4_layer = keras.layers.BatchNormalization()(conv4_layer)
        # conv4_layer = tf.nn.relu(conv4_layer)
        # 全局池化层和softmax分类器
        gap = keras.layers.GlobalAveragePooling2D()(conv3_layer)
        outputs = tf.add(tf.matmul(gap,self.weights['out_w']),self.biases['out_b'])

        

        return outputs
