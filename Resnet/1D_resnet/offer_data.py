import numpy as np
import pandas as pd
import tensorflow as tf
import os



def read_data_return(is_training,file):
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
    all_data = np.reshape(np.array(all_data),[len(file_list),256,60,1])
    all_label = np.reshape(np.array(all_label),[len(file_list),256])
    data = all_data
    label = tf.one_hot(all_label,2)
    label = tf.cast(label,tf.int32)
    data = tf.cast(data,tf.float32)
    
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
        data = np.reshape(all_data,[-1,256,60,1])
        label = sess.run(label)
        return data,label
        
    else:
        sess = tf.Session()
        data = np.reshape(all_data,[-1,60,1])
        label = sess.run(label)
        label = np.reshape(label,[-1,2])
        return data,label
    # sess = tf.Session()
    # for i in range(10000):
    #     sess.run([data_,label_])
    #     print(sess.run(data_).shape)
    #     print(sess.run(label_).shape)
    



# read_data_return('data/')