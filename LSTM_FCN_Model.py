import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.contrib import rnn

class LSTM_FCN_Model:

    def __init__(self, num_cells,x,weights,biases,strides):
        self.num_cells = num_cells
        self.x = x
        self.weights = weights
        self.biases = biases
        self.strides = strides
        pass
    
    def LSTM_Model(self):
        lstm = rnn.BasicLSTMCell(self.num_cells)
        lstm = tf.nn.rnn_cell.DropoutWrapper(lstm,output_keep_prob=0.8)
        outputs,states = rnn.static_rnn(lstm,self.x,dtype=tf.float32)
        return outputs

    def FCN_Model(self):
        lstm = self.LSTM_Model()
        self.x = tf.keras.layers.Permute((2,1))(self.x)

        conv1_layer = tf.layers.Conv2D(self.x,filters=128,kernel_size=8,strides=1,kernel_initializer=tf.keras.initializers.he_uniform(),padding='same')
        # conv1_layer = tf.nn.bias_add(conv1_layer,self.biases['conv1_b'])
        conv1_layer = keras.layers.BatchNormalization()(conv1_layer)
        conv1_layer = tf.nn.relu(conv1_layer)

        # 第二层卷积层
        conv2_layer = tf.layers.Conv2D(conv1_layer,filters=256,kernel_size=5,strides=1,kernel_initializer=tf.keras.initializers.he_uniform(),padding='same')
        # conv2_layer = tf.nn.bias_add(conv2_layer,self.biases['conv2_b'])
        conv2_layer = keras.layers.BatchNormalization()(conv2_layer)
        conv2_layer = tf.nn.relu(conv2_layer) 

        # 第三层卷积层
        conv3_layer = tf.layers.Conv2D(conv2_layer,filters=128,kernel_size=5,strides=1,kernel_initializer=tf.keras.initializers.he_uniform(),padding='same')
        # conv3_layer = tf.nn.bias_add(conv3_layer,self.biases['conv3_b']) 
        conv3_layer = keras.layers.BatchNormalization()(conv3_layer)
        conv3_layer = tf.nn.relu(conv3_layer)

        # 全局池化
        gap = keras.layers.GlobalAveragePooling2D()(conv3_layer)
        
        out = tf.keras.layers.concatenate([lstm,gap])
        logits = tf.keras.layers.Softmax(out)