import tensorflow as tf
import tensorflow.keras as keras
import os
import numpy as np
import random

# 定义训练参数
# num_hidden = 128
# 定义权重
init = tf.keras.initializers.he_uniform()

# 先定义一层卷积，看看输出的东西是什么
weights = {
    'conv1':tf.Variable(init([8,1,128]))
}
biases = {
    'conv1':tf.Variable(init(128))
}

X = tf.placeholder(tf.float32,[None,1,30])
# Y = tf.placeholder(tf.fl)


def con1d(x,w,b):
    x = tf.transpose(x,[0,2,1])
    print(np.array(x).shape)
    conv1d = tf.nn.conv1d(x,weights['conv1'],1,'SAME')
    conv1d = tf.nn.bias_add(conv1d,biases['conv1'])
    bn_conv1d = tf.keras.layers.BatchNormalization()(conv1d)
    act_conv1d = tf.nn.relu(bn_conv1d)
    return act_conv1d,conv1d,bn_conv1d





act_conv1d,conv1d,bn_conv1d =  con1d(X,weights,biases)
all_init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    