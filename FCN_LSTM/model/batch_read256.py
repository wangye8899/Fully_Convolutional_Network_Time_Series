import numpy as np
import pandas as pd
import tensorflow as tf
import os

def read_csv_(file_name):
    # 传入需要读取的文件名
    content = pd.read_csv(file_name,header=None).values
    content = np.array(content)
    # label = np.array([ con[0] for con in content])
    # data = np.array([ con[1:] for con in content])
    label = [int(con[0]) for con in content]
    data = [ float(c) for con in content for c in con[1:] ]
    # step = 30
    # data = [ data[i:i+step] for i in range(0,len(data),step)]
    data = np.array(data)
    # print(label)
    
    return label ,data


def csv_convert_TFRecord(filename,num):
    # 此函数实现csv文件数据转换成TFRecord格式的数据
    
    floder = "./TFval/"
    writer = tf.python_io.TFRecordWriter(floder+str(num)+".tfrecords")
    label,data = read_csv_(filename)
    example = tf.train.Example(
        features = tf.train.Features(
            feature = {
                'label': tf.train.Feature(int64_list = tf.train.Int64List(value=label)),
                'data': tf.train.Feature(float_list = tf.train.FloatList(value=data.flatten()))}))

    serialized = example.SerializeToString()
    writer.write(serialized)
    print(num)
    print(filename+"处理完毕")


def TFRecordReader(tfrecord_file,flag):
    # 读取TFRecord文件
    # tfrecord_file = ['data153.csv.tfrecords']
    print("开始读文件")
    file_list_ =  os.listdir(tfrecord_file)
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
        "data":tf.FixedLenFeature([30*256],tf.float32),
        "label":tf.FixedLenFeature([256],tf.int64)
    }

    init = tf.global_variables_initializer()
    sess.run(init)
    # print(sess.run(serialized_example_none))
    parsed_features = tf.parse_single_example(serialized_example_none,features)
    
    label = tf.cast(parsed_features["label"],tf.int64)
    data = tf.cast(parsed_features["data"],tf.float32)
    data = tf.reshape(data,[256,30])
    
    label_batch,data_batch = tf.train.shuffle_batch([label, data], batch_size=num,capacity=10000, min_after_dequeue=200, num_threads=5)
    coord=tf.train.Coordinator()
    threads= tf.train.start_queue_runners(sess=sess,coord=coord)
    
    data ,label = sess.run([data_batch,label_batch])
    # print(np.array(data).shape)
    # label = tf.cast(label,tf.int32)
    # label = tf.one_hot(label,4)
    # print(np.array(sess.run(label)))
    
    return data,label ,file_list_


if __name__ == "__main__":
    # read_csv_('data153.csv')
    file_list = os.listdir('./val/')
    i = 0
    for file in file_list:
        i+=1
        file="./val/"+file
        csv_convert_TFRecord(file,i)

    # TFRecordReader('./TFval/',0)