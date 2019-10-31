import numpy as np
import pandas as pd
import tensorflow as tf
import os
import configparser
file = 'parameter_256_3_5.ini'
def read_config(file):
    cf = configparser.ConfigParser()
    cf.read(file)
    time_steps = cf.getint('model','time_steps')
    learning_rate = cf.getfloat('model','learning_rate')
    training_steps = cf.getint('model','train_steps')
    batch_size = cf.getint('model','batch_size')    
    display_step = cf.getint('model','display_step')
    test_steps = cf.getint('model','test_steps')
    num_input = cf.getint('model','num_input')
    num_hidden = cf.getint('model','num_hidden')
    num_classes = cf.getint('model','num_classes')
    timesteps = cf.getint('model','time_steps')
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
    return num_hidden, strides,num_channels,num_input,learning_rate,train_file,epoch_num,training_steps,batch_size,display_step,num_classes,Is_Vali,val_file,plot_train_step,model_file,time_steps

# 返回值
num_hidden,strides,num_channels,num_input,learning_rate,train_file,epoch_num,training_steps,batch_size ,display_step,num_classes,Is_Vali,val_file,plot_train_step,model_file,time_steps= read_config(file)



def read_data_return(is_training,file):
    print("数据读取，请稍后")
    all_data = []
    all_label = []
    file_list = os.listdir(file)
    for file_name in file_list:
        file_name = file+file_name
        content = pd.read_csv(file_name,header=None).values
        content = np.array(content)
        # label = np.array([ con[0] for con in content])
        # data = np.array([ con[1:] for con in content])
        label = [int(con[0]) for con in content]
        data = [ float(c) for con in content for c in con[1:] ]
        all_data.append(data)
        all_label.append(label)
    # 将数据reshape
    all_data = np.reshape(np.array(all_data),[len(file_list),time_steps,num_input])
    all_label = np.reshape(np.array(all_label),[len(file_list),time_steps])
    data = all_data
    label = tf.one_hot(all_label,num_classes)
    
    # print(dataset.output_types)
    # print(dataset.output_shapes)
    if is_training:
        # dataset = tf.data.Dataset.from_tensor_slices((data,label))
        # dataset = dataset.shuffle(buffer_size=1000).batch(256).repeat(10000)
        # print(dataset.output_shapes)
        # iterator = dataset.make_one_shot_iterator()
        # data_,label_ = iterator.get_next()
        # return data_,label_
        sess = tf.Session()
        data = np.reshape(all_data,[-1,batch_size,time_steps,num_input])
        label = sess.run(label)
        label = np.reshape(label,[-1,batch_size,time_steps,num_classes])
        return data,label
        # return shuffle_data(data,label)
        
        
    else:
        sess = tf.Session()
        data = np.reshape(all_data,[-1,time_steps,num_input])
        label = sess.run(label)
        label = np.reshape(label,[-1,time_steps,num_classes])
        return data,label
    # sess = tf.Session()
    # for i in range(10000):
    #     sess.run([data_,label_])
    #     print(sess.run(data_).shape)
    #     print(sess.run(label_).shape)
    

def shuffle_data(data,label):
    # 同时对data、label按照同种顺序进行了shuffle
    print("数据读取完毕，shuffle数据中。。。。")
    label_list = []
    data_list = []
    for i in range(epoch_num*training_steps):
        state = np.random.get_state()
        np.random.shuffle(data)
        np.random.set_state(state)
        np.random.shuffle(label)
        data = np.reshape(data,[-1,batch_size,time_steps,num_input])
        label = np.reshape(label,[-1,batch_size,time_steps,num_classes])
        label_list.append(label)
        data_list.append(data)
        
    label_iter = iter(label_list)
    data_iter = iter(data_list)

    return data_iter,label_iter
# read_data_return('data/')