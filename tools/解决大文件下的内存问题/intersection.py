import numpy as np
import pandas as pd
import os 
import tensorflow as tf
from itertools import product
def read_csv_processed(file_name):
    # 读处理过的csv文件，拿到雷达信号、噪声标识符和向量数据
    vector = pd.read_csv(file_name,header=None).values
    vector = np.array(vector)
    rador_or_noise = [int(v[0]) for v in vector]
    data = [float(v) for vec in vector for v in vec[1:]]
    step = 30
    data = [data[i:i+step] for i in range(0,len(data),step)]
    data = np.array(data)
    
    return data ,rador_or_noise


def process_node_generate_edge(data,rador_or_noise):
    # 处理每个点之间的关系，生成邻接矩阵
    # num = 0
    edge = []
    edge_vector = []
    edge_vector_ = []
    for i,j in product(range(len(rador_or_noise)),range(len(rador_or_noise))):
            # 噪声
        if rador_or_noise[i] == 0:
            # print(num)
            edge.append(0)
            intersect_ = np.intersect1d(data[i],data[j])
            intersect_ = np.pad(intersect_,(0,30-len(intersect_)),'constant',constant_values=0)
            edge_vector.append(intersect_)
        else:    
            # 出噪声外 一起处理
            if rador_or_noise[i]==rador_or_noise[j]:
                edge.append(rador_or_noise[i])
                intersect_ = np.intersect1d(data[i],data[j])
                intersect_ = np.pad(intersect_,(0,30-len(intersect_)),'constant',constant_values=0)
                edge_vector.append(intersect_)
            else:
                edge.append(0)
                intersect_ = np.intersect1d(data[i],data[j])
                intersect_ = np.pad(intersect_,(0,30-len(intersect_)),'constant',constant_values=0)
                edge_vector.append(intersect_)
    
    print(len(edge_vector))
    edge_vector_ = [e for ed in edge_vector for e in ed]
    print(len(edge_vector_))
    edge = np.array(edge)
    edge_vector_ = np.array(edge_vector_)
    return edge,edge_vector_



def csv_convert_TFRecord(data,label,num):
    # 此函数实现csv文件数据转换成TFRecord格式的数据
    print(len(data))
    floder = "./TFtest/"
    writer = tf.python_io.TFRecordWriter(floder+str(num)+".tfrecords")
    # label,data = read_csv_(filename)
    example = tf.train.Example(
        features = tf.train.Features(
            feature = {
                'label': tf.train.Feature(int64_list = tf.train.Int64List(value=label)),
                'data': tf.train.Feature(float_list = tf.train.FloatList(value=data.flatten()))}))

    serialized = example.SerializeToString()
    writer.write(serialized)
    print(num)
   

def TFRecordReader(tfrecord_file,flag):
    # 读取TFRecord文件
    # tfrecord_file = ['data153.csv.tfrecords']
    file_list_ =  os.listdir(tfrecord_file)
    print(file_list_)
    if flag==0:
        num = len(file_list_)
    else:
        num = flag
    sess = tf.Session()
    file_list_  =  [ tfrecord_file+f for f in file_list_]
    # print(file_list_)
    filename_queues = tf.train.string_input_producer(file_list_)
    reader_none = tf.TFRecordReader(options=None)
    # reader_none = tf.WholeFileReader()
    init = tf.local_variables_initializer()
    sess.run(init)
    _,serialized_example_none = reader_none.read(filename_queues)
    
    features = {
        "data":tf.FixedLenFeature([30*1048576],tf.float32),
        "label":tf.FixedLenFeature([1048576],tf.int64)
    }

    init = tf.global_variables_initializer()
    sess.run(init)
    # print(sess.run(serialized_example_none))
    parsed_features = tf.parse_single_example(serialized_example_none,features)
    
    label = tf.cast(parsed_features["label"],tf.int64)
    data = tf.cast(parsed_features["data"],tf.float32)
    data = tf.reshape(data,[1048576,30])
    print(data)
    print(label)
    label_batch,data_batch = tf.train.shuffle_batch([label, data], batch_size=num,capacity=10000, min_after_dequeue=200, num_threads=5)
    coord=tf.train.Coordinator()
    threads= tf.train.start_queue_runners(sess=sess,coord=coord)
    print(data)
    # coord.request_stop()
    # coord.join(threads)
    data ,label = sess.run([data_batch,label_batch])
    print(len(data))
    print(len(label))
    return data,label 

if __name__ == "__main__":
    file_list = os.listdir("./test_1/")
    # print(file_list)
    # i = 0
    for file in file_list:
    #     i+=1
        file = "./test_1/"+file
        data , rador_or_noise = read_csv_processed(file)
    #     # print("的撒多")
        edge , edge_vector = process_node_generate_edge(data,rador_or_noise)
        # print(edge_vector)
        # print(len(edge))
        print("邻接矩阵形状1024*1024:"+str(len(edge)))
        print("邻接矩阵中1的个数:")
        print(sum(edge))
        print("1的比重:")
        print(sum(edge)/len(edge))
        # print(edge.reshape([128,128]))
        # for line in edge.reshape([128,128]):
            # print(line)
    #     csv_convert_TFRecord(edge_vector,edge,i)
    # data , label = TFRecordReader('./TFtest',0)            

    
    
   
    