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
        # 第二层卷积
        conv2 = tf.nn.conv1d(conv1,self.weights['conv2'],1,'SAME')
        conv2 = tf.nn.bias_add(conv2,self.biases['conv2'])
        conv2 = tf.keras.layers.BatchNormalization()(conv2)
        conv2 = tf.nn.relu(conv2)
        # 第三层卷积
        conv3 = tf.nn.conv1d(conv2,self.weights['conv3'],1,'SAME')
        conv3 = tf.nn.bias_add(conv3,self.biases['conv3'])
        conv3 = tf.keras.layers.BatchNormalization()(conv3)
        conv3 = tf.nn.relu(conv3)
        # 第四层卷积

        conv4 = tf.nn.conv1d(conv3,self.weights['conv4'],1,'SAME')
        conv4 = tf.nn.bias_add(conv4,self.biases['conv4'])
        conv4 = tf.nn.relu(conv4)
        # # # # 第五层卷积
        conv5 = tf.nn.conv1d(conv4,self.weights['conv5'],1,'SAME')
        conv5 = tf.nn.bias_add(conv5,self.biases['conv5'])
        # # # 第六层卷积
        # conv6 = tf.nn.conv1d(conv5,self.weights['conv6'],1,'SAME')
        # conv6 = tf.nn.bias_add(conv6,self.biases['conv6'])
        # # # 全局池化层和softmax分类器
        FCN_out = tf.keras.layers.GlobalAveragePooling1D()(conv5)
        # conv4 = tf.nn.conv1d(conv3,self.weights['conv4'],1,'SAME')
        # conv4 = tf.nn.bias_add(conv4,self.biases['conv4'])
        # conv4 = tf.nn.relu(conv4)
        # # 第五层卷积
        # conv5 = tf.nn.conv1d(conv4,self.weights['conv5'],1,'SAME')
        # conv5 = tf.nn.bias_add(conv5,self.biases['conv5'])
        # 全局池化层和softmax分类器
        # FCN_out = tf.keras.layers.GlobalAveragePooling1D()(conv5)
        print("全局池化之后的形状")
        print(np.array(FCN_out))
        # FCN_out = tf.add(tf.matmul(FCN_out,self.weights['out_w']),self.biases['out_b'])
        return FCN_out

    def LSTM_(self):
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
        fcn_out = self.FCN_()
        lstm_out = self.LSTM_()
        connect_out = tf.concat([fcn_out,lstm_out],1)
        print("连接后的矩阵形状")
        print(np.array(connect_out))
        print("最终的形状")
        print(np.array(tf.matmul(connect_out,self.weights['out_w'])))
        connect_out = tf.add(tf.matmul(connect_out,self.weights['out_w']),self.biases['out_b'])
        return connect_out

    def Bi_LSTM_(self):
        model = Sequential()
        model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.num_hidden)))(self.x)
        return model
        x = model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.num_hidden)))(self.x)
        return x
    # def ResNet_(self):

# 以下为优化模型的可能探索方式：
'''
1. 加深网络，不过可能出现网络的层数越深而网络将会出现退化能力，所以模型的左侧可不可以考虑改成残差网络，在残差块的作用下，加深网络，增强模型的学习能力
2. 右侧lstm模型，改成双向LSTM？
3. 去掉Dropout参数，因为我们的数据量够大，过拟合问题不会太严重。
4. 模型在训练时训练集的准确率和损失仍然有些大，分别为0.88,0.25，是不是模型欠拟合了呢？
5. 欠拟合的解决：
    1.获得更多的数据特征
    2.能不能考虑去掉BatchNormation?虽然说BN并不是直接针对过拟合，但是在一定作用上还是减轻了过拟合的效果，所以只能试试
    3.扩大模型的规模，增加层数
'''
'''
改进顺序：
1. 取消Dropout参数 （有效果，继续保留）
2. 取消BN操作       (无效果，反而有稍微下降)
3. 尝试右侧改为双向LSTM
4. 尝试左侧改为多层神经网络，先改成4,5,6层试一下
5. 左侧模型改成多层残差网络
'''