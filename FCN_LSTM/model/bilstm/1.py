'''
        # 模型1.0，仅包含三层的lstm神经网络。不过对于最简单的数据，效果还可以。
        lstm_cell = contrib.rnn.BasicLSTMCell(self.num_hiddens,activation=tf.nn.tanh)
        predict,state = tf.nn.dynamic_rnn(lstm_cell,self.x,dtype=tf.float32)
        # 此时predict的维度是[batch_size,256,num_hiddens]
        # 使用全连接层，将predict转换为label的格式，即[batch_size,1024,2]
        # predict = np.array(predict)
        predict = tf.reshape(predict,[-1,self.num_hiddens])
        fully_connected_layer_outputs = tf.layers.dense(predict,2)
        return fully_connected_layer_outputs
        '''
        # 模型2.0
        '''
        # 模型2.0,
        在模型2.0中，lstm将替换成Bi-lstm,使用双向循环神经网络的目的是：前后两个方向去学习信号点向量
        同时，在双向循环神经网络处理后，得到的输出将送至Attention层，使用Attention注意力机制为每个点向量分配权重，找到关键词
        '''
        forward_lstm = contrib.rnn.BasicLSTMCell(self.num_hiddens,activation=tf.nn.tanh)
        backward_lstm = contrib.rnn.BasicLSTMCell(self.num_hiddens,activation=tf.nn.tanh)
        outputs,_,_ = contrib.rnn.static_bidirectional_rnn(forward_lstm,backward_lstm,self.x,dtype=tf.float32)
        tf.unstack()