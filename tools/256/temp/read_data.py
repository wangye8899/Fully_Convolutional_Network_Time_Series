import numpy as np
import pandas as pd
import csv
import os
import tensorflow as tf

def read_csv_(file_name):
    # 传入需要读取的文件名
    content = pd.read_csv(file_name,header=None).values
    content = np.array(content)
    label = [int(con[0]) for con in content]
    data = [ float(c) for con in content for c in con[1:] ]
    step = 30
    data = [ data[i:i+step] for i in range(0,len(data),step)]
    data = np.array(data)
    label = np.array(label)
    # print("信号数据如下:")
    # print(data)
    # print(label)
    # print(data)
    # print(label)
    return data,label

def return_value(path):
    file_lsit = os.listdir(path)
    data_list = []
    lable_list = []
    sess = tf.Session()
    for file in file_lsit:
        data,label = read_csv_(path+file)
        data_list.append(data)
        lable_list.append(sess.run(tf.one_hot(tf.cast(label,tf.int32),4)))   
    data_list = np.array(data_list)
    # print(data_list[0].shape)
    # print(np.array(lable_list[0]).shape)
    return data_list,lable_list


# return_value('../val_/')