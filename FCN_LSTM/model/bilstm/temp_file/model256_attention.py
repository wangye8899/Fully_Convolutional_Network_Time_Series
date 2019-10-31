import tensorflow as tf
import tensorflow.contrib as contrib
import numpy as np
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn
class model_:
    def __init__(self,x,num_hiddens,weights,biases):
        # self.x = tf.unstack(x,256,1)
        self.x = x 
        self.num_hiddens = num_hiddens
        self.weights = weights
        self.biases = biases


    def modeling(self):
        '''
        # 模型1.0，仅包含三层的lstm神经网络。不过对于最简单的数据，效果还可以。
        lstm_layer = contrib.rnn.MultiRNNCell([contrib.rnn.BasicLSTMCell(self.num_hiddens,activation=tf.nn.tanh)  for i in range(3)])
        lstm_layer_drop = contrib.rnn.DropoutWrapper(lstm_layer,input_keep_prob=1.0,output_keep_prob=0.5)
        lstm_outs,state = tf.nn.dynamic_rnn(lstm_layer_drop,self.x,dtype=tf.float32)
        # 初始化全连接层的权重
        w_fcn = tf.Variable(tf.random_normal([self.num_hiddens,2]))
        b_fcn = tf.Variable(tf.random_normal([2]))
        
        out_fcn = tf.matmul(tf.reshape(lstm_outs,[-1,128]),w_fcn)+b_fcn

        return out_fcn

        '''
        # 模型2.0
        '''
        # 模型2.0,
        在模型2.0中，lstm将替换成Bi-lstm,使用双向循环神经网络的目的是：前后两个方向去学习信号点向量
        同时，在双向循环神经网络处理后，得到的输出将送至Attention层，使用Attention注意力机制为每个点向量分配权重，找到关键词
        lstm_layer = contrib.rnn.MultiRNNCell([contrib.rnn.DropoutWrapper(contrib.rnn.BasicLSTMCell(self.num_hiddens,activation=tf.nn.tanh),input_keep_prob=1.0,output_keep_prob=0.5)  for i in range(3)])
        lstm_outs,state = tf.nn.dynamic_rnn(lstm_layer,self.x,dtype=tf.float32)
        # lstm_outs = outputs[-1]
        # print("双向lstm的输出shape")
        # print(lstm_outs.get_shape()) # [batch_size,256,128]
        # outputs_after_tanh = tf.nn.tanh(lstm_outs) # [batch_size,256,128]
        # print("激活后：")
        # print(outputs_after_tanh.get_shape())
        # # 初始化Attention权重
        # w_attention = tf.Variable(tf.random_normal([self.num_hiddens])) #[1,128]
        # print("w_attention的shape")
        # print(w_attention.get_shape())
        # # 求解注意力权重
        # alpha_wights = tf.nn.softmax(tf.matmul(tf.reshape(outputs_after_tanh,[-1,self.num_hiddens]),tf.reshape(w_attention,[-1,1]))) #[batch_size*256,1]
        # print("alpha的shape")
        # print(alpha_wights.get_shape())
        # r = tf.matmul(tf.transpose(lstm_outs,[0,2,1]),tf.reshape(alpha_wights,[-1,256,1])) #[batch_size*128,1]
        # print("r的形状")
        # print(r.get_shape())
        # final_r = tf.reshape(r,[-1,self.num_hiddens])
        # print("最终r的形状")
        # print(final_r.get_shape())
        # activation_r = tf.tanh(final_r)
        '''
        

        '''
        模型3.0 在模型1.0的基础上，将三层lstm改为三层双向lstm，看看在256_1_5的数据下，是否有提升。
        '''
        input_ = self.x
        for _ in range(3):
            with tf.variable_scope(None, default_name="bidirectional-rnn"):
                fw_lstm_cell = contrib.rnn.LSTMCell(self.num_hiddens,initializer=tf.orthogonal_initializer())
                bw_lstm_cell = contrib.rnn.LSTMCell(self.num_hiddens,initializer=tf.orthogonal_initializer())
                # 目前均不加Dropout层
                outputs,state = tf.nn.bidirectional_dynamic_rnn(fw_lstm_cell,bw_lstm_cell,input_,dtype=tf.float32)
                input_ = tf.concat(outputs,2) 

        #此时3层BLSTM输出的形状为 [?,256,256]
        # 使用全连接层进行分类
        input_reshape = tf.reshape(input_,[-1,256])
        fcn_return = tf.matmul(input_reshape,self.weights['out'])+self.biases['out']
        return fcn_return