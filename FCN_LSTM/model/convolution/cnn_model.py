import tensorflow as tf
import tensorflow.contrib as contrib
import numpy as np
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn

class model_:
    def __init__(self,x,num_hiddens,weights,biases):
        self.x = x 
        self.num_hiddens = num_hiddens
        self.weights = weights
        self.biases = biases

    
    def modeling(self):
        '''
        使用卷积的思想设置模型
        '''
        # 第一层卷积
        conv1_layer = tf.nn.conv1d(self.x,self.weights['conv1'],1,"SAME")
        conv1_layer = tf.nn.bias_add(conv1_layer,self.biases['conv1'])
        conv1_layer = tf.keras.layers.BatchNormalization()(conv1_layer)
        conv1_layer = tf.nn.relu(conv1_layer)
        # conv1_layer = tf.nn.dropout(conv1_layer,keep_prob=0.7)
        # 第一层池化
        conv1_layer = tf.layers.average_pooling1d(conv1_layer,2,1,'same')
        # print(conv1_layer)
        # 第二层卷积
        conv2_layer = tf.nn.conv1d(conv1_layer,self.weights['conv2'],1,"SAME")
        conv2_layer = tf.nn.bias_add(conv2_layer,self.biases['conv2'])
        conv2_layer = tf.keras.layers.BatchNormalization()(conv2_layer)
        conv2_layer = tf.nn.relu(conv2_layer)
        conv2_layer = tf.layers.average_pooling1d(conv2_layer,3,1,'same')
        # print(conv2_layer)
        # 第三层卷积
        conv3_layer = tf.nn.conv1d(conv2_layer,self.weights['conv3'],1,"SAME")
        conv3_layer = tf.nn.bias_add(conv3_layer,self.biases['conv3'])
        conv3_layer = tf.keras.layers.BatchNormalization()(conv3_layer)
        conv3_layer = tf.nn.relu(conv3_layer)
        conv3_layer = tf.layers.average_pooling1d(conv3_layer,2,1,'same')
        # 第四层卷积
        conv4_layer = tf.nn.conv1d(conv3_layer,self.weights['conv4'],1,"SAME")
        conv4_layer = tf.nn.bias_add(conv4_layer,self.biases['conv4'])
        conv4_layer = tf.keras.layers.BatchNormalization()(conv4_layer)
        conv4_layer = tf.nn.relu(conv4_layer)
        conv4_layer = tf.layers.average_pooling1d(conv4_layer,3,1,'same')
        # 第五层卷积
        conv5_layer = tf.nn.conv1d(conv4_layer,self.weights['conv5'],1,"SAME")
        conv5_layer = tf.nn.bias_add(conv5_layer,self.biases['conv5'])
        conv5_layer = tf.keras.layers.BatchNormalization()(conv5_layer)
        conv5_layer = tf.nn.relu(conv5_layer)
        conv5_layer = tf.layers.average_pooling1d(conv5_layer,2,1,'same')
        # 第六城卷积
        conv6_layer = tf.nn.conv1d(conv5_layer,self.weights['conv6'],1,"SAME")
        conv6_layer = tf.nn.bias_add(conv6_layer,self.biases['conv6'])
        conv6_layer = tf.keras.layers.BatchNormalization()(conv6_layer)
        conv6_layer = tf.nn.relu(conv6_layer)
        conv6_layer = tf.layers.average_pooling1d(conv6_layer,3,1,'same')
        # 第七层卷积
        conv7_layer = tf.nn.conv1d(conv6_layer,self.weights['conv7'],1,"SAME")
        conv7_layer = tf.nn.bias_add(conv7_layer,self.biases['conv7'])
        conv7_layer = tf.keras.layers.BatchNormalization()(conv7_layer)
        conv7_layer = tf.nn.relu(conv7_layer)
        conv7_layer = tf.layers.average_pooling1d(conv7_layer,2,1,'same')
        
        # 第八层卷积    
        conv8_layer = tf.nn.conv1d(conv7_layer,self.weights['conv8'],1,"SAME")
        conv8_layer = tf.nn.bias_add(conv8_layer,self.biases['conv8'])
        conv8_layer = tf.keras.layers.BatchNormalization()(conv8_layer)
        conv8_layer = tf.nn.relu(conv8_layer)
        conv8_layer = tf.layers.average_pooling1d(conv8_layer,3,1,'same')
        global_avg_pool = tf.keras.layers.GlobalAveragePooling1D()(conv8_layer)
        print(global_avg_pool)
        

        # 右侧为双向lstm
        lstm_input = tf.reshape(self.x,[-1,256,30])

        for _ in range(4):
            with tf.variable_scope(None, default_name="bidirectional-rnn"):
                fw_lstm_cell = contrib.rnn.LSTMCell(self.num_hiddens,initializer=tf.initializers.orthogonal(),activation=tf.nn.tanh)
                bw_lstm_cell = contrib.rnn.LSTMCell(self.num_hiddens,initializer=tf.initializers.orthogonal(),activation=tf.nn.tanh)
                # 目前均不加Dropout层
                outputs,state = tf.nn.bidirectional_dynamic_rnn(fw_lstm_cell,bw_lstm_cell,lstm_input,dtype=tf.float32)
                lstm_input = tf.concat(outputs,2) 

        #此时3层BLSTM输出的形状为 [?,256,256]
        # 使用全连接层进行分类

        input_reshape = tf.reshape(lstm_input,[-1,256])

        outputs = tf.concat((input_reshape,global_avg_pool),1)
        # print(outputs)
        # fcn_return = tf.matmul(input_reshape,self.weights['out'])+self.biases['out']
        outputs = tf.layers.dense(outputs,2,activation=tf.nn.relu)
        return outputs

