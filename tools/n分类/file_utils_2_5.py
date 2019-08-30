'''
读文件，分别取label和data，将str格式转换成int或者float格式，并分别保存成list形式，最后将list
转换成numpy数组的形式
'''
import numpy as np
import pandas as pd
import csv
import tensorflow as tf

def read_file_and_to_numpy(filename,amount):
    # 参数说明：filename代表文件名 amount：代表读取的batchsize
    # 逐行读取
    csv_file = open(filename)
    csv_reader_lines = np.array(list(csv.reader(csv_file)))
    # random_index = np.random.choice(len(csv_reader_lines),amount)
    # print(random_index)
    # lines = csv_reader_lines[random_index]
    label = []
    data = []
    for oneline in csv_reader_lines:
        label.append(int(oneline[0]))
        new_data = [float(oneline[1:][num]) for num in range(len(oneline[1:]))]
        data.append(new_data)
    label_numpy = np.array(label)
    data_numpy = np.array(data)
    # print(label_numpy)
    # target = np.eye()
    target = tf.one_hot(label,3)
    print(target)
    sess = tf.Session()
    target = sess.run(target[:20])
    print(target)
    
    # print(data_numpy.shape)
    # print(type(target))
    # 返回数据和标签
    # print(len(target[:amount]))
    # print(len(data_numpy[:amount]))
    # print(target[:10])
    # print(data_numpy[:10])
    # print(len(target))
    # print(len(data_numpy))
    return target,data_numpy

if __name__ == "__main__":
    read_file_and_to_numpy('all_data_vector_val.csv',10000)
