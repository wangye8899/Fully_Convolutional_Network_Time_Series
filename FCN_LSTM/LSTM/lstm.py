import tensorflow as tf
import numpy as np
import tensorflow.contrib as contrib

class basic_lstm:
    def __init__(self,x,weights,biases,num_hiddens):
        self.x = x 
        self.weights = weights
        self.biases = biases
        self.num_hiddens = num_hiddens
    

    def cnn_model(self):
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
        # pool = tf.keras.layers.GlobalAveragePooling1D()(conv3)
        return conv3

    def lstm_model(self):
        # 每次输入的数据格式为：[1024*30]
        lstm_cell = contrib.rnn.BasicLSTMCell(num_units=self.num_hiddens)
        muti_lstm = contrib.rnn.MultiRNNCell([lstm_cell for i in range(3)])
        init_state = muti_lstm.zero_state(1024,dtype=tf.float32)
        outputs , state = tf.nn.dynamic_rnn(muti_lstm,inputs=self.x,initial_state=init_state)
        
        # # x = tf.transpose(self.x,[0,2,1])
        # lstm = tf.nn.rnn_cell.BasicLSTMCell(self.num_hiddens)
        # outputs,_ = tf.nn.dynamic_rnn(lstm,self.x,dtype=tf.float32)
        # # outputs = tf.matmul(ou)
        logits =tf.add( tf.matmul(outputs[:,-1,:],self.weights['out_w']),self.biases['out_b'])
        return logits
