import tensorflow as tf
from file_utils import read_file_and_to_numpy
# 加载模型文件，对测试数据进行测试
import configparser
import os
import pandas as pd
from data_process import generate_vector,write_vector_to_file
# from file_utils512 import  read_file_and_to_numpy_val
from batch_read256 import read_csv_,TFRecordReader
file = './config/parameter_256_3_5.ini'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import time

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
    plot_test_step = cf.getint('model','plot_test_steps')
    return learning_rate,training_steps,batch_size,display_step,test_steps,num_input,\
        num_hidden,num_classes,timesteps,epoch_num,Is_Vali,train_file,test_file,val_file,\
            model_file,test_file_path,plot_test_step



def read_model(filename,test_file,num_input,timesteps,batch_size,test_steps,test_file_path):
    # filename 为模型文件保存路径
    with tf.Session() as sess:
        model_file = filename+"model.meta"
        model_saver = tf.train.import_meta_graph(model_file)
        filename = tf.train.latest_checkpoint(filename)
        model_saver.restore(sess,filename)
    # 拿到损失函数和准确率指标
        loss = tf.get_collection('loss')[0]
        accuracy = tf.get_collection('accuracy')[0]
        prediction = tf.get_collection('prediction')[0]
        tf.summary.scalar('test_loss',loss)
        tf.summary.scalar('test_acc',accuracy)
        meraged = tf.summary.merge_all()
        write = tf.summary.FileWriter('./tensorboard/',sess.graph)
    # 加载placeholder
        graph = tf.get_default_graph()
        X = graph.get_operation_by_name('X').outputs[0]
        Y = graph.get_operation_by_name('Y').outputs[0]
        
        test_data,test_label,test_file_list = TFRecordReader(test_file,0)
        temp_label = test_label
        test_label = tf.cast(test_label,tf.int32)
        test_label = tf.one_hot(test_label,2)
        test_label = sess.run(test_label)
        acc_list = []
        loss_list = []
        epoch_list = []
        count = 0
        acc_sum = 0
        loss_sum = 0
        # test_data = test_data[:51200]
        # test_label = test_label[:51200]
        # 循环测试所有的csv测试文件
        start_time = time.time()
        for i in range(len(test_file_list)):
            file = str(test_file_list[i])
            batch_data = test_data[i]
            batch_label = test_label[i]
            batch_data = np.reshape(batch_data,[1,256,30])
            # batch_data = batch_data.reshape(len(batch_data),1,num_input)
            print(file+":"+"Final Test Accuracy:",\
                    sess.run(accuracy,feed_dict={X:batch_data,Y:batch_label}))
            print(file+":"+"Final Test Loss:",\
                    sess.run(loss,feed_dict={X:batch_data,Y:batch_label}))
            
            # pre = sess.run(tf.argmax(prediction,1),feed_dict={X:test_data})
            # print(sess.run(tf.argmax(prediction,1),feed_dict={X:test_data}))
        end_time = time.time()
        cost_time_sum = end_time - start_time
        print("测试文件的总时间为:"+str(cost_time_sum))
        print("平均的测试时间:"+str(cost_time_sum/len(test_file_list)))
        
        print("计算平均准确率中.................")
        for i in range(len(test_file_list)):
            file = str(test_file_list[i])
            batch_data = test_data[i]
            batch_label = test_label[i]
            batch_data = np.reshape(batch_data,[1,256,30])
            print(file+":"+"Final Test Accuracy:",\
                    sess.run(accuracy,feed_dict={X:batch_data,Y:batch_label}))
            print(file+":"+"Final Test Loss:",\
                    sess.run(loss,feed_dict={X:batch_data,Y:batch_label}))
            acc_sum  = acc_sum+sess.run(accuracy,feed_dict={X:batch_data,Y:batch_label})
            loss_sum = loss_sum+sess.run(loss,feed_dict={X:batch_data,Y:batch_label})
        print("平均准确率为："+str(acc_sum/len(test_file_list)))
        print("平均损失为："+str(loss_sum/len(test_file_list)))

        pre_list = np.array( sess.run(tf.argmax(prediction,1),feed_dict={X:batch_data,Y:batch_label}))
        wirte_csv = open('./pre.csv','a')
        # zeros_list = []
        # zeros_sum = 0 
        # one_sum = 0
        # two_sum = 0
        # temp_count_0 = 0
        # temp_count_1 = 0
        # temp_count_2 = 0
        # for i in pre_list:
        #     print(i)
        #     zeros_list.append(i)
        #     wirte_csv.write(str(i))
        #     wirte_csv.write('\n')
        # for i in range(len(zeros_list)):
        #     if temp_label[0][i] == 0:
        #         zeros_sum+=1
        #     if temp_label[0][i] == 1:
        #         one_sum+=1
        #     if temp_label[0][i] == 2:
        #         two_sum+=1
        # for i in range(len(zeros_list)):
        #     if zeros_list[i] == 0:
        #         if temp_label[0][i] == 0:
        #             temp_count_0+=1
        #     if zeros_list[i] == 1:
        #         if temp_label[0][i] == 1:
        #             temp_count_1+=1
        #     if zeros_list[i]==2:
        #         if temp_label[0][i] == 2:
        #             temp_count_2+=1
            
        # print(temp_count_0/zeros_sum)
        # print(temp_count_1/one_sum)
        # print(temp_count_2/two_sum)
        
def read_file_produce_vector(path,i):
    # 拿到文件夹下所有的csv文件
    filename = "test"+str(i)+"csv"
    csv_path = path+filename
    # 将当前文件转换成目标向量
    df = pd.read_csv(csv_path,usecols=["Label","Data"])
    data = list(df.get("Data"))
    label = list(df.get("Label"))
    # 此处参数只要不填1,2,3即可
    label_list , pri_list = generate_vector(filename,data,label,5)
    temp_file = 'one_test_vector.csv'
    # 将当前测试文件写入csv文件
    write_vector_to_file(filename,temp_file,label_list,pri_list,0)
    label,data = read_file_and_to_numpy(temp_file,0)
    return label,data,filename    





if __name__ == "__main__":
    learning_rate,training_steps,batch_size,display_step,test_steps,\
    num_input,num_hidden,num_classes,timesteps,epoch_num,Is_Vali,\
        train_file,test_file,val_file,model_file,test_file_path,plot_test_step =  read_config(file) 
    
    read_model(model_file,test_file,num_input,timesteps,batch_size,test_steps,test_file_path)
    
