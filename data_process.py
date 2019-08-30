import numpy as np
import pandas as pd
import csv
from collections import Counter
import os

data_in = []
label_in = []
class data_pro:

    def __init__(self,window_size,intersection,filename,num_):
        # 滑动窗口的大小和交集的概率值alpha,文件
        self.window_size = window_size
        self.intersection = intersection
        self.filename = filename
        self.num_ = num_
        

    def read_csv_file(self):
        # 读取文件夹中所有的csv文件，然后对每个csv文件进行处理，按照论文中的相应数据处理方式生成向量
        for path,dir_list,file_list in os.walk(self.filename):
            # file_list 为所有的文件
            for file in file_list:
                # 四行语句获取到所有的Label、Data数据
                csv_name = os.path.join(path,file)
                df = pd.read_csv(csv_name,usecols=["Label","Data"])
                data = list(df.get("Data"))
                label = list(df.get("Label"))
                # 将所有的data、label数据转成numpy_array的形式，交给第二个函数处理
                self.process_data(data,label)


    def process_data(self,data,label):
        # pri_为pri向量
        pri_ = []
        for i in range(len(data)):
            # 生成向量
            if i<=117:
                pri_.append([data[i+1]-data[i],data[i+2]-data[i],data[i+3]-data[i],data[i+4]-data[i],data[i+5]-data[i],data[i+6]-data[i],data[i+7]-data[i],data[i+8]-data[i],data[i+9]-data[i],data[i+10]-data[i]])
            elif i==118:
                pri_.append([data[i+1]-data[i],data[i+2]-data[i],data[i+3]-data[i],data[i+4]-data[i],data[i+5]-data[i],data[i+6]-data[i],data[i+7]-data[i],data[i+8]-data[i],data[i+9]-data[i],0])
            elif i==119:
                pri_.append([data[i+1]-data[i],data[i+2]-data[i],data[i+3]-data[i],data[i+4]-data[i],data[i+5]-data[i],data[i+6]-data[i],data[i+7]-data[i],data[i+8]-data[i],0,0])
            elif i==120:
                pri_.append([data[i+1]-data[i],data[i+2]-data[i],data[i+3]-data[i],data[i+4]-data[i],data[i+5]-data[i],data[i+6]-data[i],data[i+7]-data[i],0,0,0])
            elif i==121:
                pri_.append([data[i+1]-data[i],data[i+2]-data[i],data[i+3]-data[i],data[i+4]-data[i],data[i+5]-data[i],data[i+6]-data[i],0,0,0,0])
            elif i==122:
                pri_.append([data[i+1]-data[i],data[i+2]-data[i],data[i+3]-data[i],data[i+4]-data[i],data[i+5]-data[i],0,0,0,0,0])
            elif i==123:
                pri_.append([data[i+1]-data[i],data[i+2]-data[i],data[i+3]-data[i],data[i+4]-data[i],0,0,0,0,0,0])
            elif i==124:
                pri_.append([data[i+1]-data[i],data[i+2]-data[i],data[i+3]-data[i],0,0,0,0,0,0,0])
            elif i==125:
                pri_.append([data[i+1]-data[i],data[i+2]-data[i],0,0,0,0,0,0,0,0])
            elif i==126:
                pri_.append([data[i+1]-data[i],0,0,0,0,0,0,0,0,0])
            elif i==127:
                pri_.append([0,0,0,0,0,0,0,0,0,0])

        # 返回特征矩阵进行归一化1处理
        # self.normalizition_data(pri_,label,self.num_)
        self.normalizition_data_normal(pri_,label,self.num_)
    
    def normalizition_data(self,pri_,label,num_):
        # 对数据向量进行归一化处理，共有正态归一化，平均归一化，max-min归一化
        # 1.平均化
        pri_ = [p/12500 for pri in pri_ for p in pri]
        if num_==0:
            filename = "all_data_vector_train.csv"
            self.write_vector_to_file(pri_,label,filename)
        elif num_ ==1:
            filename = "all_data_vector_test.csv"
            self.write_vector_to_file(pri_,label,filename)
        else:
            filename = "all_data_vector_val.csv"
            self.write_vector_to_file(pri_,label,filename)
    def normalizition_data_normal(self,pri_,label,num_):
        global data_in 
        global label_in
        steps = 10
        for la in label:
            label_in.append(la)
        for pri in pri_:
            for p in pri:
                data_in.append(p)
        print(len(data_in))
        print(len(label_in))
        if len(data_in) == 2560000:
            data_in_ = np.array(data_in)
            mean = data_in_.mean()
            std = data_in_.std()
            pri_ = [ ((data-mean)/std) for data in data_in_ ]
            # pri_ = [ pri_[i:i+steps] for i in range(0,len(pri_),steps) ]
        # print(len(pri_))
            if num_==0:
                filename = "all_data_vector_train.csv"
                self.write_vector_to_file(pri_,label_in,filename)
            elif num_ ==1:
                filename = "all_data_vector_test.csv"
                self.write_vector_to_file(pri_,label_in,filename)
            else:
                filename = "all_data_vector_val.csv"
                self.write_vector_to_file(pri_,label_in,filename)


    def write_vector_to_file(self,data,label,filename):

        # 写入文件  
        f = open(filename,'a')
        # 先将数据转成str类型
        data_list = [str(da) for da in data]
        # 每十个为一组向量
        step = 10 
        data_list = [data_list[i:i+step] for i in range(0,len(data),step)]
        
        for num in range(len(label)):
            if label[num]==100:
                label[num]=0
            else:
                label[num] = 1
            f.write(str(label[num])+","+",".join(data_list[num])+"\n")
            

if __name__ == "__main__":
    data_pro_ = data_pro(5,0.1,"/home/wang/Radar数据/Deinterleaving/data/5参差/val/",2) 
    data,label = data_pro_.read_csv_file()
    # pri_,label = data_pro_.process_data(data,label)
    # data_pro_.normalizition_data(pri_,label,1)


