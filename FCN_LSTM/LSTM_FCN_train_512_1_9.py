import tensorflow as tf
import tensorflow.keras as keras
import configparser
from LSTM_FCN_model import LSTM_FCN
# from file_utils import read_file_and_to_numpy
from file_utils512 import read_file_and_to_numpy
from file_utils512 import read_file_and_to_numpy_val
file = './config/parameter_512_1_9.ini'


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import random
import os
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
    model_file = cf.get('file','model_file')
    return num_hidden, strides,num_channels,num_input,learning_rate,train_file,epoch_num,training_steps,batch_size,display_step,num_classes,Is_Vali,val_file,plot_train_step,model_file

# 返回值
num_hidden,strides,num_channels,num_input,learning_rate,train_file,epoch_num,training_steps,batch_size ,display_step,num_classes,Is_Vali,val_file,plot_train_step,model_file= read_config(file)

# 定义权重和偏置
init = tf.keras.initializers.he_uniform()
# init = tf.keras.initializers.he_normal()
# init = tf.keras.initializers.random_uniform()
# init = tf.keras.initializers.random_normal()

weights = {
'conv1':tf.Variable(init([8,1,128])),
'conv2':tf.Variable(init([5,128,256])),
'conv3':tf.Variable(init([3,256,128])),
'conv4':tf.Variable(init([3,128,128])),
'conv5':tf.Variable(init([3,128,128])),
# 'conv6':tf.Variable(init([3,64,32])),
# 'conv4_w':tf.Variable(tf.random_normal([3,1,256,128])),
'out_w':tf.Variable(init([256,2]))
}
biases = {
'conv1':tf.Variable(init([128])),
'conv2':tf.Variable(init([256])),
'conv3':tf.Variable(init([128])),
'conv4':tf.Variable(init([128])),
# 'conv5':tf.Variable(init([64])),
# 'conv6':tf.Variable(init([32])),
# 'conv4_b':tf.Variable(tf.random_normal([128])),
'out_b':tf.Variable(init([2]))
}
# 定义X，Y的占位符
X = tf.placeholder(tf.float32,[None,1,num_input],name='X')
Y = tf.placeholder(tf.float32,[None,num_classes],name='Y')
# batch_size = tf.Variable(128,dtype=tf.float32)
lr = tf.Variable(0.001,dtype=tf.float32)
LSTM_FCN = LSTM_FCN(X,weights,biases,num_hidden)
logits = LSTM_FCN.connect_FCN_LSTM()
prediction = tf.nn.softmax(logits)
# ??????
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=Y))

# loss_op = tf.reduce_mean(tf.square(Y-logits))/2
optimizer = tf.train.AdamOptimizer(learning_rate=lr)

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
    
    
    sess.run(init)
    p = 0 
    count = 0
    temp_acc = 0
    temp_loss = 0
    acc_list = []
    loss_list = []
    epoch_list = []
    val_acc = []
    val_loss= []
    val_epoch_list = []
    # print(len(val_file_list))
    vali_label_list,vali_data_list = read_file_and_to_numpy_val(val_file)
    vali_data_list = vali_data_list[:51200]
    vali_label_list = vali_label_list[:51200]
    vali_data_list = vali_data_list.reshape(len(vali_data_list),1,num_input) 
    
    label,data = read_file_and_to_numpy(train_file)
    # label
    # data = ?data.tolist()
    for epoch in range(epoch_num):
        if epoch==50:
            sess.run(tf.assign(lr,lr/10))
        elif epoch==150:
            sess.run(tf.assign(lr,lr/10))
        elif epoch==300:
            sess.run(tf.assign(lr,lr/10))
            

        # 每一个epoch，都要对数据进行洗牌，打乱顺序。
        # idx = np.random.permutation(len(data))
        # data = data[idx]
        # label = label[idx]
        print("Epoch"+str(epoch))
        for step in range(1,training_steps+1):
            # 读数据
            random_index = random.randint(0,9999)
            batch_data = data[random_index]
            batch_label = label[random_index]
            batch_data = np.array(batch_data)
            batch_label = np.array(batch_label)
            # print(batch_data.shape)
            # print(batch_label.shape)
            
            batch_data = batch_data.reshape((batch_size*512,1,num_input))
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

        Val_loss,Val_acc = sess.run([loss_op,acc],feed_dict={X:vali_data_list,Y:vali_label_list})
        
        if epoch==0:
            temp_loss = Val_loss
        else:
            if (Val_loss-temp_loss) < 0:
                # 说明损失仍然在下降
                print("loss改善:"+str(temp_loss)+"--->"+str(Val_loss)+"||"+"学习率:"+str(sess.run(lr)))

                temp_loss = Val_loss
            else:
                # 说明经过一个epoch之后模型并没有改善
                count+=1

                print("loss未改善:"+str(count)+"次"+"--->"+str(Val_loss))
        #         if count==10:
        #             count=0
        #             if float(sess.run(lr))<=0.000001:
        #                 sess.run(tf.assign(lr,lr))
        #                 print("学习率过低，恒定，不再变化")
        #             else:    
        #                 sess.run(tf.assign(lr,lr*(0.5)))
        #                 print("降低学习率为："+str(sess.run(lr)))
        if Val_acc>temp_acc:
            # temp_loss = Val_loss
            tf.train.Saver().save(sess,model_file+'model')
        else:
            pass
        print("validation acc"+"{:.4f}".format(Val_acc)+"validation loss"+"{:.4f}".format(Val_loss)) 
        val_acc.append(Val_acc)
        val_loss.append(Val_loss)
        val_epoch_list.append(epoch)
    # print("训练完成，请加载模型进行测试")
    # print("绘制图形中.......")
    # print(np.array(epoch_list))
    # print(np.array(acc_list))
    # plt.figure()
    # plt.plot(np.array(epoch_list),np.array(acc_list),label="训练集准确率")
    # plt.xlabel("epoch-iter")
    # plt.ylabel("train-acc")
    # plt.title("训练集准确率曲线图")
    # plt.legend()
    # plt.savefig('FCN_train_acc.png')
    # plt.show()
    
    # plt.figure()
    # plt.plot(np.array(epoch_list),np.array(loss_list),label="训练集损失值")
    # plt.xlabel("epoch-iter")
    # plt.ylabel("train_loss")
    # plt.title("训练集损失值")
    # plt.legend()
    # plt.savefig('FCN_train_loss.png')
    # plt.show()

    print("训练完成，请加载模型进行测试")
    print("绘制图形中.......")
    print(np.array(val_acc))
    print(np.array(val_loss))
    plt.figure()
    plt.plot(np.array(val_epoch_list),np.array(val_acc),label="验证集准确率")
    plt.xlabel("epoch-iter")
    plt.ylabel("val-acc")
    plt.title("验证集准确率曲线图")
    plt.legend()
    plt.savefig('FCN_val_acc.png')
    plt.show()
    
    plt.figure()
    plt.plot(np.array(val_epoch_list),np.array(val_loss),label="验证集损失值")
    plt.xlabel("epoch-iter")
    plt.ylabel("val_loss")
    plt.title("验证集损失值")
    plt.legend()
    plt.savefig('FCN_val_loss.png')
    plt.show()