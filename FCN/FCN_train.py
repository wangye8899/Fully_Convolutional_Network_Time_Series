import tensorflow as tf
import tensorflow.keras as keras
import configparser
from FCN_model import FCN_model
from file_utils import read_file_and_to_numpy
file = './config/parameter.ini'

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import random

# 读取配置文件，拿到参数
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
    return strides,num_channels,num_input,learning_rate,train_file,epoch_num,training_steps,batch_size,display_step,num_classes,Is_Vali,val_file,plot_train_step

# 返回值
strides,num_channels,num_input,learning_rate,train_file,epoch_num,training_steps,batch_size ,display_step,num_classes,Is_Vali,val_file,plot_train_step= read_config(file)

# 定义权重和偏置
init = tf.keras.initializers.he_uniform()
weights = {
<<<<<<< HEAD:FCN/FCN_train.py
'conv1_w':tf.Variable(init([8,8,1,128])),
'conv2_w':tf.Variable(init([5,5,128,256])),
'conv3_w':tf.Variable(init([3,3,256,128])),
# 'conv4_w':tf.Variable(tf.random_normal([3,1,256,128])),
'out_w':tf.Variable(init([128,2]))
}
biases = {
'conv1_b':tf.Variable(init([128])),
'conv2_b':tf.Variable(init([256])),
=======
'conv1_w':tf.Variable(tf.random_normal([8,8,1,128])),
'conv2_w':tf.Variable(tf.random_normal([5,5,128,256])),
'conv3_w':tf.Variable(tf.random_normal([3,3,256,128])),
# 'conv4_w':tf.Variable(tf.random_normal([3,1,256,128])),
'out_w':tf.Variable(tf.random_normal([128,2]))
}
biases = {
'conv1_b':tf.Variable(tf.random_normal([128])),
'conv2_b':tf.Variable(tf.random_normal([256])),
>>>>>>> 90200e08ccc8078e6456de3c9d879e4da2e082cf:FCN_train.py
'conv3_b':tf.Variable(tf.random_normal([128])),
# 'conv4_b':tf.Variable(tf.random_normal([128])),
'out_b':tf.Variable(init([2]))
}
# 定义X，Y的占位符
X = tf.placeholder(tf.float32,[None,1,num_input,1],name='X')
Y = tf.placeholder(tf.float32,[None,num_classes],name='Y')

FCN_model_ = FCN_model(X,weights,biases,strides,num_channels)
logits = FCN_model_.Conv2D()
prediction = tf.nn.softmax(logits)
# ??????
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=Y))
       
# loss_op = tf.reduce_mean(tf.square(Y-logits))/2
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_steps = optimizer.minimize(loss_op)

# 定义准确率
acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(prediction,1),tf.argmax(Y,1)),tf.float32))
meraged = tf.summary.merge_all()

tf.add_to_collection('loss',loss_op)
tf.add_to_collection('accuracy',acc)
tf.add_to_collection('prediction',prediction)

init = tf.global_variables_initializer()
# 定义训练过程

with tf.Session() as sess:
    label , data = read_file_and_to_numpy(train_file,1)
    vali_label_list,vali_data_list = read_file_and_to_numpy(val_file,10000)
    vali_data_list = vali_data_list[:12800]
    vali_label_list = vali_label_list[:12800]
    sess.run(init)
    p = 0 
    temp_acc = 0
    acc_list = []
    loss_list = []
    epoch_list = []
    for epoch in range(epoch_num):
        idx = np.random.permutation(len(data))
        data = data[idx]
        label = label[idx]
        print("Epoch"+str(epoch))
        count_pre = 0 
        count = batch_size
        for step in range(1,training_steps+1):
            batch_data = []
            batch_label = []
            random_index = random.sample(range(0,len(label)),128)
            for num in random_index:
                batch_data.append(data[num])
                batch_label.append(label[num])
            batch_data = np.array(batch_data)
            batch_label = np.array(batch_label)
                # 将tensor的形状重组成神经网络的输入要求
            batch_data = batch_data.reshape((batch_size,1,num_input,1))
            # 现在开始训练
            sess.run(train_steps,feed_dict={X:batch_data,Y:batch_label})
            loss,accu = sess.run([loss_op,acc],feed_dict={X:batch_data,Y:batch_label})
            if p%plot_train_step==0:
                acc_list.append(accu)  
                loss_list.append(loss)  
                epoch_list.append(p)
            p+=1 
            if step%display_step==0:
                print("Step " + str(step) + ", Minibatch Loss= " + \
                    "{:.4f}".format(loss) + ", Training Accuracy= " + \
                    "{:.3f}".format(accu))
            # if step%Is_Vali==0:
                    # 每训练1000次进行一次验证
        vali_data_list = vali_data_list.reshape(len(vali_data_list),1,num_input,1)
        Val_loss,Val_acc = sess.run([loss_op,acc],feed_dict={X:vali_data_list,Y:vali_label_list})
        if Val_acc>temp_acc:
            temp_loss = Val_loss
            tf.train.Saver().save(sess,'./FCN_checkpoint/model')
        else:
            pass
        print("validation acc"+"{:.4f}".format(Val_acc)+"validation loss"+"{:.4f}".format(Val_loss)) 

    print("训练完成，请加载模型进行测试")
    print("绘制图形中.......")
    print(np.array(epoch_list))
    print(np.array(acc_list))
    plt.figure()
    plt.plot(np.array(epoch_list),np.array(acc_list),label="训练集准确率")
    plt.xlabel("epoch-iter")
    plt.ylabel("train-acc")
    plt.title("训练集准确率曲线图")
    plt.legend()
    plt.savefig('FCN_train_acc.png')
    plt.show()
    
    plt.figure()
    plt.plot(np.array(epoch_list),np.array(loss_list),label="训练集损失值")
    plt.xlabel("epoch-iter")
    plt.ylabel("train_loss")
    plt.title("训练集损失值")
    plt.legend()
    plt.savefig('FCN_train_loss.png')
    plt.show()