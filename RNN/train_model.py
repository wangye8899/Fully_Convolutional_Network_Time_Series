import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import pandas as pd
import csv
from file_utils import read_file_and_to_numpy
import configparser
from model import model
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from read_model_and_test import read_model
import random
file = './config/parameter.ini'
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
    plot_step = cf.getint('model','plot_train_steps')
    num_hidden_1 = cf.getint('model','num_hidden_1')
    num_hidden_2 = cf.getint('model','num_hidden_2')
    return learning_rate,training_steps,batch_size,display_step,test_steps,num_input,\
        num_hidden,num_classes,timesteps,epoch_num,Is_Vali,train_file,test_file,val_file,\
            model_file,plot_step,num_hidden_1,num_hidden_2

learning_rate,training_steps,batch_size,display_step,test_steps,\
    num_input,num_hidden,num_classes,timesteps,epoch_num,Is_Vali,\
        train_file,test_file,val_file,model_file,plot_step,num_hidden_1,num_hidden_2 =  read_config(file) 
model_file = str(model_file)+'model'
# 实例化模型

# # 定义Graph placeholder
X = tf.placeholder(tf.float32,[None,timesteps,num_input],name='X')

# X = tf.placeholder(tf.float32,[None,num_input],name='X')
# 对于数字序列数据，应该提前对label进行onehot编码
Y = tf.placeholder(tf.float32,[None,2],name='Y')

# he初始化
# init = tf.keras.initializers.he_uniform()
# init = tf.glorot_normal_initializer()
init = tf.glorot_uniform_initializer()
# # 定义权重 Rnn所用
weights ={
    'out':tf.Variable(init([num_hidden,num_classes]))
    # 'out':tf.Variable(tf.random_normal([2*num_hidden,num_classes]))/(tf.sqrt(float(num_hidden/2)))
    # 'out':tf.Variable(tf.zeros([num_hidden,num_classes]))
    # 'out':tf.Variable(init([num_hidden,num_classes]))
}
biases ={
    # 'out':tf.Variable(tf.random_normal([num_classes]))
    # 'out':tf.Variable(tf.zeros([num_classes]))
    'out':tf.Variable(init([num_classes]))
}

#全连接神经网络
# weights = {
#     'h1':tf.Variable(tf.random_normal([num_input,num_hidden_1])),
#     'h2':tf.Variable(tf.random_normal([num_hidden_1,num_hidden_2])),
#     'out':tf.Variable(tf.random_normal([num_hidden_2,num_classes]))
# }
# biases = {
#     'b1':tf.Variable(tf.random_normal([num_hidden_1])),
#     'b2':tf.Variable(tf.random_normal([num_hidden_2])),
#     'out':tf.Variable(tf.random_normal([num_classes]))
# }
model_ = model(X,weights,biases,timesteps,num_hidden)

# logits = model_.MulTi_LSTM()
# logits = model_.MulTi_LSTM()
# logits = model_.BiRNN()
logits = model_.Basic_Rnn()
prediction = tf.nn.softmax(logits)

# # 定义损失和优化器
# 损失函数优先考虑二次代价函数又称均方误差
# loss_op = tf.reduce_mean(tf.square(Y-logits))/2

loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=Y))
tf.summary.scalar('loss',loss_op)
# loss_op = tf.reduce_mean(tf.square(logits-Y)+tf.contrib.layers.l1_regularizer(0.1))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_steps = optimizer.minimize(loss_op)
# # 定义准确率
acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(prediction,1) ,tf.argmax(Y,1)),tf.float32))
tf.summary.scalar('acc',acc)
# # 初始化变量
meraged = tf.summary.merge_all()
init = tf.global_variables_initializer()

tf.add_to_collection('loss',loss_op)
tf.add_to_collection('accuracy',acc)
tf.add_to_collection('prediction',prediction)

with tf.Session() as sess:
    # print(train_file)
    label , data = read_file_and_to_numpy(train_file,10000)
    vali_label_list,vali_data_list = read_file_and_to_numpy(val_file,10000)
    temp_acc = 0
    num = 0 
    # temp_list = []
    sess.run(init)
    write = tf.summary.FileWriter('./tensorboard/',sess.graph)
    acc_list = []
    loss_list = []
    epoch_list= []
    p = 0
    for epoch in range(epoch_num):
        idx = np.random.permutation(len(data))
        data = data[idx]
        label = label[idx]
        print("epoch:"+str(epoch))  
        # count_pre = 0 
        # count = batch_size
        # num+=1   
        for step in range(1,training_steps+1):
            batch_data = []
            batch_label = []
            random_idx =  random.sample(range(0,len(label)),batch_size)
            for num in random_idx:
                batch_data.append(data[num])
                batch_label.append(label[num])
            batch_data = np.array(batch_data)
            batch_label = np.array(batch_label)
            # if len(data) - count>=128:
            #     batch_label = label[count_pre:count]
            #     batch_data = data[count_pre:count]
            #     count_pre = count
            #     count+=batch_size
                # 将tensor的形状重组成神经网络的输入要求
            batch_data = batch_data.reshape((batch_size,timesteps,num_input))
                # 现在开始训练
            sess.run(train_steps,feed_dict={X:batch_data,Y:batch_label})
            loss,accu = sess.run([loss_op,acc],feed_dict={X:batch_data,Y:batch_label})
            if p%plot_step==0:
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
        vali_data_list = vali_data_list.reshape(len(vali_data_list),timesteps,num_input)
        Val_loss,Val_acc = sess.run([loss_op,acc],feed_dict={X:vali_data_list,Y:vali_label_list})
        if Val_acc>temp_acc:
            temp_loss = Val_loss
            tf.train.Saver().save(sess,model_file)
        else:
            pass
        print("validation acc"+"{:.4f}".format(Val_acc)+"validation loss"+"{:.4f}".format(Val_loss)) 

        summary = sess.run(meraged,feed_dict={X:batch_data,Y:batch_label})
        write.add_summary(summary,epoch)

   
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
    plt.savefig('train_acc.png')
    plt.show()
    
    plt.figure()
    plt.plot(np.array(epoch_list),np.array(loss_list),label="训练集损失值")
    plt.xlabel("epoch-iter")
    plt.ylabel("train_loss")
    plt.title("训练集损失值")
    plt.legend()
    plt.savefig('train_loss.png')
    plt.show()

