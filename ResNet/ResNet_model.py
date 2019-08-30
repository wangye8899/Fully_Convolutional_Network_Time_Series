from tensorflow import keras
import tensorflow as tf
import numpy as np
import pandas as pd


class resnet_model:
    def __init__(self,x,weights,biases,num_hidden,strides):
        self.x = x
        self.weights = weights
        self.biases = biases
        self.num_hidden = num_hidden
        self.strides = strides

    def build_model(self):
        print("构建第一层")
        x = keras.layers.BatchNormalization()(self.x)
        conv1 = tf.nn.conv2d(x,self.weights['conv1_w'],strides=[1,self.strides,self.strides,1],padding='SAME')
        conv1 = keras.layers.BatchNormalization()(conv1)
        conv1 = tf.nn.relu(conv1)

        print("构建第二层")
        conv2 = tf.nn.conv2d(conv1,self.weights['conv2_w'],strides=[1,self.strides,self.strides,1],padding='SAME')
        conv2 = keras.layers.BatchNormalization()(conv2)
        conv2 = tf.nn.relu(conv2)

        print("构建第三层")
        conv3 = tf.nn.conv2d(conv2,self.weights['conv3_w'],strides=[1,self.strides,self.strides,1],padding='SAME')
        conv3 = keras.layers.BatchNormalization()(conv3)
        conv3 = tf.nn.relu(conv3)    