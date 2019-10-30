import tensorflow as tf
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from tensorflow.contrib.layers import fully_connected



def weight_variable(shape,name):
    # init = tf.keras.initializers.he_uniform()
    init = tf.keras.initializers.lecun_normal()
    weight_ = tf.get_variable(name,dtype=tf.float32,initializer=init(shape))
    return weight_

def bais_variable(shape,name):
    # init = tf.keras.initializers.he_uniform()
    init = tf.keras.initializers.lecun_normal()
    bia_ = tf.get_variable(name,dtype=tf.float32,initializer=init(shape)) 
    return bia_

def conv1d(x,filters,kernel_size,name,stride=1,activation=tf.nn.relu,padding="SAME"):
    conv1d_ = tf.layers.conv1d(x,filters,kernel_size,strides=stride,padding=padding,kernel_initializer=tf.keras.initializers.lecun_normal(),use_bias=True,name=name)  
    return conv1d_

def maxpool1d(x,pool_size,strides,name,padding="SAME"):
    maxpool1d_ = tf.layers.max_pooling1d(x,pool_size,strides,padding=padding,name=name)
    return maxpool1d_

def res_block(x,filters,kernel_size,name):
    # print(x)
    res_block_ = conv1d(x,filters,kernel_size,stride=1,name=name+"_conv1")
    # print(res_block_)
    res_block_ = tf.layers.batch_normalization(res_block_,name=name+"_batch1")
    res_block_ = tf.nn.relu(res_block_,name=name+"_relu")

    res_block_ = conv1d(res_block_,filters,kernel_size,stride=1,name=name+"_conv2")
    res_block_ = tf.layers.batch_normalization(res_block_,name=name+'_batch2')
    addition = tf.nn.relu(res_block_+x)
    print(x)
    print(res_block_)
    return addition

def fullylayer(x,units,name):
    dense_ = tf.layers.dense(x,units,activation=tf.nn.relu,name=name)
    return dense_

def conv2ser(x, params_fc):
    return tf.reshape(x, [-1, params_fc.get_shape().as_list()[0]])

def resblock_s(x, filters, kernel_size, name):
    resblock=conv1d(x, filters, kernel_size, stride=2, name=name+'_conv1')
    shorcut=tf.layers.batch_normalization(resblock, name=name+'_batch1')
    resblock=tf.nn.relu(shorcut, name=name+'_relu')
    resblock=conv1d(resblock, filters, kernel_size, stride=1, name=name+'_conv2')
    resblock=tf.layers.batch_normalization(resblock, name=name+'_batch2')
    print(shorcut)
    print(res_block)

    return tf.nn.relu(resblock+shorcut)