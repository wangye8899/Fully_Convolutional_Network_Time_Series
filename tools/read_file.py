'''
读文件，分别取label和data，将str格式转换成int或者float格式，并分别保存成list形式，最后将list
转换成numpy数组的形式
'''
import numpy as np
import pandas as pd
import csv
import tensorflow as tf
import os

def read_file_and_to_numpy(path):
    # 参数说明：filename代表文件名 amount：代表读取的batchsize
    # 逐行读取
    label = []
    data = []   
   
    for path,dir_list,file_list in os.walk(path):
        for file in file_list:
            csv_file = open(path+file)
            csv_reader_lines = np.array(list(csv.reader(csv_file)))
            for oneline in csv_reader_lines:
                label.append(int(oneline[0]))
                new_data = [float(oneline[1:][num]) for num in range(len(oneline[1:]))]
                data.append(new_data)
                
    print(len(data))        
    label_numpy = np.array(label)
    data_numpy = np.array(data)
        
    target = tf.one_hot(label,2)
    # print(target)
    sess = tf.Session()
    target = sess.run(target)
    print(len(data_numpy))
    return target,data_numpy

if __name__ == "__main__":
   read_file_and_to_numpy('../data/val/') 