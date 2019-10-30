import tensorflow as tf
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from tensorflow.contrib.layers import fully_connected
from return_moudle import *


class Resnet1D:
    def __init__(self,x):
        self.x = x
        self.shape = [-1,60,1]

    def run(self):
        X_reshaped = tf.reshape(self.x,self.shape)
        # conv1d
        conv1 = conv1d(X_reshaped,64,7,name='conv1',stride=1)
        print(conv1)
        # maxpool1
        maxpool1 = maxpool1d(conv1,2,strides=1,name='maxpool1',padding='Valid')
        # print(maxpool1)

        # 残差块1、2 、3 
        res_block1 = res_block(maxpool1,64,3,name='resblock1')
        res_block2 = res_block(res_block1,64,3,name='resblock2')
        res_block3 = res_block(res_block2,64,3,name='resblock3')
        res_block4 = res_block(res_block3,64,3,name="resblock4")
        res_block5 = res_block(res_block4,64,3,name="resblock5")
        # res_block6 = res_block(res_block5,64,3,name="resblock6")
        # res_block7 = res_block(res_block6,64,3,name="resblock7")
        # res_block8 = res_block(res_block7,64,3,name="resblock8")
        maxpool2 = maxpool1d(res_block5,2,1,name='maxpool2',padding='VALID')
        print(maxpool2)
        
        res_block9 = resblock_s(maxpool2,128,3,name="resblock9")
        res_block10 = res_block(res_block9,128,3,name="resblock10")
        res_block11 = res_block(res_block10,128,3,name="resblock11")
        res_block12 = res_block(res_block11,128,3,name='resblock12')
        res_block13 = res_block(res_block12,128,3,name="resblock13")
        # res_block14 = res_block(res_block13,128,3,name="resblock14")
        # res_block15 = res_block(res_block14,128,3,name="resblock15")
        # res_block16 = res_block(res_block15,128,3,name="resblock16")
        maxpool3 = maxpool1d(res_block13,2,1,name="maxpool3",padding="Valid")
        print(maxpool3)



        res_block17 = resblock_s(maxpool3,256,3,name='resblock17')
        res_block18 = res_block(res_block17,256,3,name='resblock18')
        res_block19 = res_block(res_block18,256,3,name="resblock19")
        res_block20 = res_block(res_block19,256,3,name="resblock20")
        # res_block21 = res_block(res_block20,256,3,name="resblock21")
        # res_block22 = res_block(res_block21,256,3,name="resblock22")
        # res_block23 = res_block(res_block22,256,3,name="resblock23")
        # res_block24 = res_block(res_block23,256,3,name="resblock24")
        maxpool4 = maxpool1d(res_block20,2,1,name="maxpool4",padding="Valid")
        print(maxpool4)

        res_block25 = resblock_s(maxpool4,512,3,name="resblock25")
        res_block26 = res_block(res_block25,512,3,name="resblock26")
        res_block27 = res_block(res_block26,512,3,name="resblock27")
        res_block28 = res_block(res_block27,512,3,name="resblock28")
        # res_block29 = res_block(res_block28,512,3,name="resblock29")
        # res_block30 = res_block(res_block29,512,3,name= "resblock30")
        # res_block31 = res_block(res_block30,512,3,name="resblock31")
        # res_block32 = res_block(res_block31,512,3,name="resblock32")
        maxpool5 = maxpool1d(res_block28,2,1,name="maxpool5",padding="Valid")
        print(maxpool5)
        
        res_block33 = resblock_s(maxpool5,1024,3,name="resblock33")
        res_block34 = res_block(res_block33,1024,3,name="resblock34")
        res_block35 = res_block(res_block34,1024,3,name="resblock35")
        res_block36 = res_block(res_block35,1024,3,name="resblock36")
        # res_block37 = res_block(res_block36,1024,3,name="resblock37")
        # res_block38 = res_block(res_block37,1024,3,name="resblock38")
        # res_block39 = res_block(res_block38,1024,3,name="resblock39")
        # res_block40 = res_block(res_block39,1024,3,name="resblock40")
        maxpool6 = maxpool1d(res_block36,2,1,name="maxpool5",padding="Valid")
        print(maxpool6)


        # res_block20 = resblock_s(res_block19,)
        global_avg_pool = tf.keras.layers.GlobalAveragePooling1D()(maxpool6)
        print(global_avg_pool)

        # fcn1 = tf.matmul(global_avg_pool,weight_variable([1024,512],name="weights1"))+bais_variable([512],name="bias1")
        # fcn1 = tf.nn.relu(fcn1)
        # fcn1 = tf.layers.batch_normalization(fcn1)
        # print(fcn1)
        outputs = tf.matmul(global_avg_pool,weight_variable([1024,2],name="weights2"))+bais_variable([2],name="bias2")
        # outputs = tf.nn.dropout(outputs,keep_prob=0.8)
        print(outputs)
        return outputs


if __name__ == "__main__":
    x = tf.constant(1.0,shape=[1,60,1])
    sess = tf.Session()
    a = Resnet1D(x)
    a.run()
