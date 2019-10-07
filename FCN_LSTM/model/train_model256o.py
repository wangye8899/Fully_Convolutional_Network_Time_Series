import tensorflow as tf
import configparser
import os
import random
import numpy as np
from model256o import model_
from batch_read256 import read_csv_,TFRecordReader
file = './config/parameter_256_3_5.ini'

# 读取配置文件

def read_config(file):
    cf = configparser.ConfigParser()
    cf.read(file)
    learning_rate = cf.getfloat('model','learning_rate')
    training_steps = cf.getint('model','train_steps')
    batch_size = cf.getint('model','batch_size')    
    display_step = cf.getint('model','display_step')
    test_steps = cf.getint('model','test_steps')
    num_input = cf.getint('model','num_input')
    num_hidden = cf.getint('model','num_hidden')
    num_classes = cf.getint('model','num_classes')
    timesteps = cf.getint('model','timesteps')
    epoch_num = cf.getint('model','epoch_num')
    Is_Vali = cf.getint('model','Is_Vali')
    train_file = cf.get('file','train_file')
    test_file = cf.get('file','test_file')
    val_file  = cf.get('file','val_file')
    model_file = cf.get('file','model_file')
    test_file_path = cf.get('file','data_file_path')
    plot_train_step = cf.getint('model','plot_train_steps')
    strides = cf.getint('model','strides')
    num_channels = cf.getint('model','num_channels')
    model_file = cf.get('file','model_file')
    return num_hidden, strides,num_channels,num_input,learning_rate,train_file,epoch_num,training_steps,batch_size,display_step,num_classes,Is_Vali,val_file,plot_train_step,model_file

# 返回值
num_hidden,strides,num_channels,num_input,learning_rate,train_file,epoch_num,training_steps,batch_size ,display_step,num_classes,Is_Vali,val_file,plot_train_step,model_file= read_config(file)

weights={
    'conv1':tf.Variable(tf.random_normal([7,1,128])),
    'conv1_out':tf.Variable(tf.random_normal([128,1])),
    'out':tf.Variable(tf.random_normal([2*num_hidden,2]))
        }
biases ={            
    'conv1':tf.Variable(tf.random_normal([128])),
    'conv1_out':tf.Variable(tf.random_normal([1])),
    'out':tf.Variable(tf.random_normal([2]))
        }

# 定义输入data、label
X = tf.placeholder(tf.float32,[None,256,num_input],name='X')
Y = tf.placeholder(tf.float32,[None,num_classes],name='Y')

# 定义loss和优化函数
model_ = model_(X,num_hidden,weights,biases)
logits = model_.modeling()
prediction = tf.nn.softmax(logits)

# 计算损失
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=Y)) 
# 定义优化函数
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_process = optimizer.minimize(loss_op)
# 定义准确率
acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(prediction,1),tf.argmax(Y,1)),tf.float32))

# 保存模型
meraged = tf.summary.merge_all()
tf.add_to_collection('loss',loss_op)
tf.add_to_collection('accuracy',acc)
tf.add_to_collection('prediction',prediction)
sess = tf.Session()
# 初始化变量
init = tf.global_variables_initializer()
# tf.get_default_graph().finalize() 
data,label,_ = TFRecordReader(train_file,0)
label = tf.cast(label,tf.int32)
label = tf.one_hot(label,2)
data  = tf.reshape(data,[5000,256,30])
val_data , val_label,_ = TFRecordReader(val_file,0)
val_data = val_data[:20]
val_label = val_label[:20]
val_label = tf.cast(val_label,tf.int32)
val_label = tf.one_hot(val_label,2)
val_label = tf.reshape(val_label,[20*256,2])
val_data =  tf.reshape(val_data,[20,256,num_input])

with tf.Session() as sess:

    count = 0
    temp_acc = 0
    sess.run(init)
    # 读取数据，调整shape，保证输入无误
    data = sess.run(data)
    label = sess.run(label)
    val_data = sess.run(val_data)
    val_label = sess.run(val_label)

    for epoch in range(epoch_num):
        print("Epoch"+str(epoch))
        for step in range(1,training_steps+1):
            # 每次随机选一个csv训练
            random_index = random.randint(0,4999)
            batch_data = data[random_index]
            batch_label = label[random_index]
            batch_data  = np.reshape(batch_data,[batch_size,256,num_input])
            batch_label = np.reshape(batch_label,[batch_size*256,2])
            # 训练
            sess.run(train_process,feed_dict={X:batch_data,Y:batch_label})
            # 计算损失和准确率
            loss,accu = sess.run([loss_op,acc],feed_dict={X:batch_data,Y:batch_label})
            # 输出
            if step%display_step==0:
                print("Step " + str(step) + ", Minibatch Loss= " + \
                    "{:.4f}".format(loss) + ", Training Accuracy= " + \
                    "{:.3f}".format(accu))

        #每结束一轮epoch，便开始使用验证集验证 
        Val_loss,Val_acc = sess.run([loss_op,acc],feed_dict={X:val_data,Y:val_label})
        
        if epoch==0:
            temp_loss = Val_loss
        else:
            if (Val_loss-temp_loss) < 0:
                # 说明损失仍然在下降
                print("loss改善:"+str(temp_loss)+"--->"+str(Val_loss)+"||"+"学习率:"+str(learning_rate))

                temp_loss = Val_loss
            else:
                # 说明经过一个epoch之后模型并没有改善
                count+=1
                print("loss未改善:"+str(count)+"次"+"--->"+str(Val_loss))
    
        if Val_acc>temp_acc:
            # temp_loss = Val_loss
            tf.train.Saver().save(sess,model_file+'model')
        else:
            pass
        print("validation acc"+"{:.4f}".format(Val_acc)+"validation loss"+"{:.4f}".format(Val_loss)) 
        