import tensorflow as tf
import tensorflow.contrib as contrib
import numpy as np
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn
class model_:
    def __init__(self,x,num_hiddens,weights,biases,batch_size):
        # self.x = tf.unstack(x,256,1)
        self.x = x 
        self.num_hiddens = num_hiddens
        self.weights = weights
        self.biases = biases
        self.batch_size = batch_size


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
        input_ = tf.keras.layers.BatchNormalization()(input_)
        for _ in range(3):
            with tf.variable_scope(None, default_name="bidirectional-rnn"):
                fw_lstm_cell = contrib.rnn.LSTMCell(self.num_hiddens,initializer=tf.initializers.orthogonal(),activation=tf.nn.tanh)
                bw_lstm_cell = contrib.rnn.LSTMCell(self.num_hiddens,initializer=tf.initializers.orthogonal(),activation=tf.nn.tanh)
                # 目前均不加Dropout层
                outputs,state = tf.nn.bidirectional_dynamic_rnn(fw_lstm_cell,bw_lstm_cell,input_,dtype=tf.float32)
                input_ = tf.concat(outputs,2) 

        #此时3层BLSTM输出的形状为 [?,256,256]
        # 使用全连接层进行分类
        input_reshape = tf.reshape(input_,[-1,256])
        fcn_return = tf.matmul(input_reshape,self.weights['out'])+self.biases['out']
        return fcn_return
                        
        '''
        drop_out = 0.3
        bi_lstm_input = self.x
        bi_lstm_input = tf.keras.layers.BatchNormalization()(bi_lstm_input)
        fw_list = []
        for _ in range(3):
            fw_lstm_cell = contrib.rnn.BasicLSTMCell(self.num_hiddens,activation=tf.nn.tanh) 
            fw_lstm_cell_drop = tf.nn.rnn_cell.DropoutWrapper(fw_lstm_cell,output_keep_prob=1-drop_out)
            fw_list.append(fw_lstm_cell_drop)

        # contrib.rnn.BasicLSTMCell()
        bw_list = []
        for __ in range(3):
            bw_lstm_cell =contrib.rnn.BasicLSTMCell(self.num_hiddens,activation=tf.nn.tanh)
            bw_lstm_cell_drop = tf.nn.rnn_cell.DropoutWrapper(bw_lstm_cell,input_keep_prob=1-drop_out,output_keep_prob=1-drop_out)
            bw_list.append(bw_lstm_cell_drop)
        # init = tf.initializers.orthogonal()
        fw_list_states = [ fw.zero_state(self.batch_size,tf.float32) for fw in fw_list]
        bw_list_states = [ bw.zero_state(self.batch_size,tf.float32) for bw in bw_list]
        outputs,_,__= contrib.rnn.stack_bidirectional_dynamic_rnn(fw_list,bw_list,bi_lstm_input,initial_states_fw=fw_list_states,initial_states_bw=bw_list_states)
        outputs_ = tf.reshape(outputs,[-1,256])
        fcn_return = tf.matmul(outputs_,self.weights['out'])+self.biases['out']
        return fcn_return
        '''
        '''
        模型4.0 LSTM+Attention机制
        

        # 网络底层是三层的双向lstm
        input_ = self.x
        for _ in range(3):
            with tf.variable_scope(None, default_name="bidirectional-rnn"):
                fw_lstm_cell = contrib.rnn.LSTMCell(self.num_hiddens,initializer=tf.initializers.orthogonal(),activation=tf.nn.tanh)
                bw_lstm_cell = contrib.rnn.LSTMCell(self.num_hiddens,initializer=tf.initializers.orthogonal(),activation=tf.nn.tanh)
                # 目前均不加Dropout层
                outputs,state = tf.nn.bidirectional_dynamic_rnn(fw_lstm_cell,bw_lstm_cell,input_,dtype=tf.float32)
                input_ = tf.concat(outputs,2)
        # 此时3层Bilstm的输出形状是：[?,256,256]
        # 而三成Bi-LSTM的输出将作为整体全文向量C，交给Attention机制使用

        # 网络顶层是单向的LSTM，目的是按照由前至后的时间顺序学习，并结合Attention机制给出的关系权重，一同学习。

        predict_lstm = contrib.rnn.LSTMCell(self.num_hiddens,initializer=tf.initializers.orthogonal(),activation=tf.nn.tanh)
        
        '''
        
        '''
        模型4.0 双向lstm+lstm，当前点信息加全文信息学习
        '''
        # 首先是三层双向循环神经网络，用作encoder编码器，拿到最终的outputs和hidden
        
        # init = tf.random_normal()
        input_x = self.x
        for _ in range(3):
            with tf.variable_scope(None,default_name="bidirectional-rnn"):
                fw_lstm_cell = contrib.rnn.LSTMCell(self.num_hiddens,initializer=tf.initializers.orthogonal(),activation=tf.nn.tanh)
                bw_lstm_cell = contrib.rnn.LSTMCell(self.num_hiddens,initializer=tf.initializers.orthogonal(),activation=tf.nn.tanh)
                outputs,states = tf.nn.bidirectional_dynamic_rnn(fw_lstm_cell,bw_lstm_cell,input_x,dtype=tf.float32)
                input_x = tf.concat(outputs,2)
                fw_states ,bw_states = states
                

        # 双层LSTM最终的输出为[?,256,256]
        final_outputs = input_x
        final_states = tf.concat((fw_states.h,bw_states.h),1)

        # 返回outputs和states给Attention计算权重系数和上下文向量
        # return final_outputs,final_states
        return self.BahdanauAttention(self.num_hiddens,final_outputs,final_states)

    def BahdanauAttention(self,num_hiddens,final_outputs,final_states):
        # 计算Attention系数
        # 声明用到的权重，权重可由网络自动学习更新达到最优

        # final_outputs 表示所有点的输出，也就是h_1、h_2....h_T 形状为[batch_size,256,256]
        # final_states 表示encoder编码后的最后一个隐藏state，应该是关于全局点的隐藏state 现状为[batch_size,256]
        # 插入1个维度 1 ,state 的形状变为[batch_size,1,256]
        final_states = tf.expand_dims(final_states,1)
        # w1 = tf.random_normal([256,128])
        # w2 = tf.random_normal([256,128])
        # v1 = tf.random_normal([128,1])
        # score = tf.matmul(tf.nn.tanh(tf.matmul(final_states,w2)+tf.matmul(final_outputs,w1)),v1)
        # 接下来我们求解Attention-Score 它的形状为[batch_size,256,1] 
        score = tf.layers.dense(tf.nn.tanh(tf.layers.dense(final_states+final_outputs,128)),1) 
        print(score)
        attention_weights = tf.nn.softmax(score,1)
        print(attention_weights)
        # 计算上下文向量
        context_vector = tf.matmul(score,final_states)
        print(context_vector)
        context_vector = tf.reduce_sum(context_vector,1)
        print(context_vector) 

        context_vector = tf.expand_dims(context_vector,1)
        print(context_vector)
        attention_lstm_output = tf.nn.tanh(context_vector+final_outputs)  
        print("最终")
        print(attention_lstm_output)
        attention_lstm_output = tf.reshape(attention_lstm_output,[-1,256])
        return tf.matmul(attention_lstm_output,self.weights['out'])+self.biases['out']

        