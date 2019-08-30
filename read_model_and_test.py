import tensorflow as tf
from file_utils import read_file_and_to_numpy
# 加载模型文件，对测试数据进行测试
import configparser
import os
import pandas as pd
from data_process import generate_vector,write_vector_to_file
from file_utils import  read_file_and_to_numpy_val
file = './config/parameter.ini'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

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
        
        test_label,test_data = read_file_and_to_numpy_val(test_file)
        acc_list = []
        loss_list = []
        epoch_list = []
        count = 0
        test_data = test_data[:64000]
        test_label = test_label[:64000]
        test_data = test_data.reshape(len(test_data),1,num_input)
        print("Final Test Accuracy:",\
                sess.run(accuracy,feed_dict={X:test_data,Y:test_label}))
        print("Final Test Loss:",\
                sess.run(loss,feed_dict={X:test_data,Y:test_label}))
        # print(sess.run(tf.argmax(prediction,1),feed_dict={X:test_data}))
        plt.figure()
        plt.plot(np.array(epoch_list),np.array(acc_list))
        plt.xlabel("epoch_iter")
        plt.ylabel("test_acc")
        plt.title("The accuracy of test")
        plt.legend()
        plt.savefig('test_acc.png')
        plt.show()

        plt.figure()
        plt.plot(np.array(epoch_list),np.array(loss_list))
        plt.xlabel("epoch_iter")
        plt.ylabel("test_losss")
        plt.title("The loss of test")
        plt.legend()
        plt.savefig('test_loss.png')
        plt.show()

        
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
    
