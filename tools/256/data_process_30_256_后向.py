'''
函数作用：遍历生成的模拟数据文件,同时对每个csv生成128个特征向量数据。每个特征向量为[1,10]
'''

import csv 
import pandas as pd
import os 
from collections import Counter
import numpy as np

# dict_data = {}
all_pri=[]
all_label = []
te_min = 0
te_max = 0
def read_file_and_get_data(filename,num):
    count = 0
    ab_path = os.walk(filename)
    for path,dir_list,file_list in ab_path:
        print("文件数")
        print(len(file_list))
        i = 0
        for file in file_list:
            i +=1 
            dict_data = {}
            # count+=1
            # print("正在处理"+file)
            # 此时拿到所有的csv文件名
            csv_name = os.path.join(path,file)
            # print(csv_name)
            df = pd.read_csv(csv_name,usecols=["Label","Data"])
            data = list(df.get("Data"))
            label = list(df.get("Label"))
            # print(data)
            count+=1
            generate_vector(file,data,label,num,len(file_list),i)

            # print(count)
            
def generate_vector(file,data_list,label_list,num,num_file,j):
    
    pri_ = []
    
    # 循环key值，生成向量
    # print(len(data_list))
    # print(label_list[0])
    for i in range(len(data_list)): 
        # print(len(data_list))
        # print(data_list[127])
        # print(i)
        if i<=225:
            # print(i)
            pri_.append([data_list[i+1]-data_list[i],data_list[i+2]-data_list[i],data_list[i+3]-data_list[i],\
                    data_list[i+4]-data_list[i],data_list[i+5]-data_list[i],data_list[i+6]-data_list[i],data_list[i+7]-data_list[i],data_list[i+8]-data_list[i],data_list[i+9]-data_list[i],data_list[i+10]-data_list[i],data_list[i+11]-data_list[i],data_list[i+12]-data_list[i],data_list[i+13]-data_list[i],data_list[i+14]-data_list[i],data_list[i+15]-data_list[i],\
                        data_list[i+16]-data_list[i],data_list[i+17]-data_list[i],data_list[i+18]-data_list[i],data_list[i+19]-data_list[i],data_list[i+20]-data_list[i],data_list[i+21]-data_list[i],data_list[i+22]-data_list[i],data_list[i+23]-data_list[i],data_list[i+24]-data_list[i],data_list[i+25]-data_list[i],data_list[i+26]-data_list[i],data_list[i+27]-data_list[i],data_list[i+28]-data_list[i],data_list[i+29]-data_list[i],data_list[i+30]-data_list[i]])   
            # all_pri.append(pri_)label_list
            # pri_ = []label_list
            # print(i)
            # print(len(pri_[i]))
        elif i==226:
            pri_.append([data_list[i+1]-data_list[i],data_list[i+2]-data_list[i],data_list[i+3]-data_list[i],\
                    data_list[i+4]-data_list[i],data_list[i+5]-data_list[i],data_list[i+6]-data_list[i],data_list[i+7]-data_list[i],data_list[i+8]-data_list[i],data_list[i+9]-data_list[i],data_list[i+10]-data_list[i],data_list[i+11]-data_list[i],data_list[i+12]-data_list[i],data_list[i+13]-data_list[i],data_list[i+14]-data_list[i],data_list[i+15]-data_list[i],\
                        data_list[i+16]-data_list[i],data_list[i+17]-data_list[i],data_list[i+18]-data_list[i],data_list[i+19]-data_list[i],data_list[i+20]-data_list[i],data_list[i+21]-data_list[i],data_list[i+22]-data_list[i],data_list[i+23]-data_list[i],data_list[i+24]-data_list[i],data_list[i+25]-data_list[i],data_list[i+26]-data_list[i],data_list[i+27]-data_list[i],data_list[i+28]-data_list[i],data_list[i+29]-data_list[i],0])   # print(i)
            if len(pri_[i])!=30:
                print(i)
            # print(len(pri_[i]))
            # pri_ = []
        elif i==227:
            pri_.append([data_list[i+1]-data_list[i],data_list[i+2]-data_list[i],data_list[i+3]-data_list[i],\
                    data_list[i+4]-data_list[i],data_list[i+5]-data_list[i],data_list[i+6]-data_list[i],data_list[i+7]-data_list[i],data_list[i+8]-data_list[i],data_list[i+9]-data_list[i],data_list[i+10]-data_list[i],data_list[i+11]-data_list[i],data_list[i+12]-data_list[i],data_list[i+13]-data_list[i],data_list[i+14]-data_list[i],data_list[i+15]-data_list[i],\
                        data_list[i+16]-data_list[i],data_list[i+17]-data_list[i],data_list[i+18]-data_list[i],data_list[i+19]-data_list[i],data_list[i+20]-data_list[i],data_list[i+21]-data_list[i],data_list[i+22]-data_list[i],data_list[i+23]-data_list[i],data_list[i+24]-data_list[i],data_list[i+25]-data_list[i],data_list[i+26]-data_list[i],data_list[i+27]-data_list[i],data_list[i+28]-data_list[i],0 ,0])   # print(i)
            # print(len(pri_[i]))
            if len(pri_[i])!=30:
                print(i)
        elif i==228:
            pri_.append([data_list[i+1]-data_list[i],data_list[i+2]-data_list[i],data_list[i+3]-data_list[i],\
        data_list[i+4]-data_list[i],data_list[i+5]-data_list[i],data_list[i+6]-data_list[i],data_list[i+7]-data_list[i],data_list[i+8]-data_list[i],data_list[i+9]-data_list[i],data_list[i+10]-data_list[i],data_list[i+11]-data_list[i],data_list[i+12]-data_list[i],data_list[i+13]-data_list[i],data_list[i+14]-data_list[i],data_list[i+15]-data_list[i],\
            data_list[i+16]-data_list[i],data_list[i+17]-data_list[i],data_list[i+18]-data_list[i],data_list[i+19]-data_list[i],data_list[i+20]-data_list[i],data_list[i+21]-data_list[i],data_list[i+22]-data_list[i],data_list[i+23]-data_list[i],data_list[i+24]-data_list[i],data_list[i+25]-data_list[i],data_list[i+26]-data_list[i],data_list[i+27]-data_list[i],0,0 ,0])
            if len(pri_[i])!=30:
                print(i)
            
        elif i==229:
            pri_.append([data_list[i+1]-data_list[i],data_list[i+2]-data_list[i],data_list[i+3]-data_list[i],\
        data_list[i+4]-data_list[i],data_list[i+5]-data_list[i],data_list[i+6]-data_list[i],data_list[i+7]-data_list[i],data_list[i+8]-data_list[i],data_list[i+9]-data_list[i],data_list[i+10]-data_list[i],data_list[i+11]-data_list[i],data_list[i+12]-data_list[i],data_list[i+13]-data_list[i],data_list[i+14]-data_list[i],data_list[i+15]-data_list[i],\
            data_list[i+16]-data_list[i],data_list[i+17]-data_list[i],data_list[i+18]-data_list[i],data_list[i+19]-data_list[i],data_list[i+20]-data_list[i],data_list[i+21]-data_list[i],data_list[i+22]-data_list[i],data_list[i+23]-data_list[i],data_list[i+24]-data_list[i],data_list[i+25]-data_list[i],data_list[i+26]-data_list[i],0,0,0,0])
            if len(pri_[i])!=30:

                print(i)
            
        elif i==230:
            pri_.append([data_list[i+1]-data_list[i],data_list[i+2]-data_list[i],data_list[i+3]-data_list[i],\
        data_list[i+4]-data_list[i],data_list[i+5]-data_list[i],data_list[i+6]-data_list[i],data_list[i+7]-data_list[i],data_list[i+8]-data_list[i],data_list[i+9]-data_list[i],data_list[i+10]-data_list[i],data_list[i+11]-data_list[i],data_list[i+12]-data_list[i],data_list[i+13]-data_list[i],data_list[i+14]-data_list[i],data_list[i+15]-data_list[i],\
            data_list[i+16]-data_list[i],data_list[i+17]-data_list[i],data_list[i+18]-data_list[i],data_list[i+19]-data_list[i],data_list[i+20]-data_list[i],data_list[i+21]-data_list[i],data_list[i+22]-data_list[i],data_list[i+23]-data_list[i],data_list[i+24]-data_list[i],data_list[i+25]-data_list[i],0,0,0,0,0])
            if len(pri_[i])!=30:
                print(i)
            
        elif i==231:
            pri_.append([data_list[i+1]-data_list[i],data_list[i+2]-data_list[i],data_list[i+3]-data_list[i],\
        data_list[i+4]-data_list[i],data_list[i+5]-data_list[i],data_list[i+6]-data_list[i],data_list[i+7]-data_list[i],data_list[i+8]-data_list[i],data_list[i+9]-data_list[i],data_list[i+10]-data_list[i],data_list[i+11]-data_list[i],data_list[i+12]-data_list[i],data_list[i+13]-data_list[i],data_list[i+14]-data_list[i],data_list[i+15]-data_list[i],\
            data_list[i+16]-data_list[i],data_list[i+17]-data_list[i],data_list[i+18]-data_list[i],data_list[i+19]-data_list[i],data_list[i+20]-data_list[i],data_list[i+21]-data_list[i],data_list[i+22]-data_list[i],data_list[i+23]-data_list[i],data_list[i+24]-data_list[i],0,0,0,0,0,0])
            if len(pri_[i])!=30:
                print(i)
            
        elif i==232:
            pri_.append([data_list[i+1]-data_list[i],data_list[i+2]-data_list[i],data_list[i+3]-data_list[i],\
        data_list[i+4]-data_list[i],data_list[i+5]-data_list[i],data_list[i+6]-data_list[i],data_list[i+7]-data_list[i],data_list[i+8]-data_list[i],data_list[i+9]-data_list[i],data_list[i+10]-data_list[i],data_list[i+11]-data_list[i],data_list[i+12]-data_list[i],data_list[i+13]-data_list[i],data_list[i+14]-data_list[i],data_list[i+15]-data_list[i],\
            data_list[i+16]-data_list[i],data_list[i+17]-data_list[i],data_list[i+18]-data_list[i],data_list[i+19]-data_list[i],data_list[i+20]-data_list[i],data_list[i+21]-data_list[i],data_list[i+22]-data_list[i],data_list[i+23]-data_list[i],0,0,0,0,0,0,0])    
            if len(pri_[i])!=30:
                print(i)
            
        elif i==233:
            pri_.append([data_list[i+1]-data_list[i],data_list[i+2]-data_list[i],data_list[i+3]-data_list[i],\
        data_list[i+4]-data_list[i],data_list[i+5]-data_list[i],data_list[i+6]-data_list[i],data_list[i+7]-data_list[i],data_list[i+8]-data_list[i],data_list[i+9]-data_list[i],data_list[i+10]-data_list[i],data_list[i+11]-data_list[i],data_list[i+12]-data_list[i],data_list[i+13]-data_list[i],data_list[i+14]-data_list[i],data_list[i+15]-data_list[i],\
            data_list[i+16]-data_list[i],data_list[i+17]-data_list[i],data_list[i+18]-data_list[i],data_list[i+19]-data_list[i],data_list[i+20]-data_list[i],data_list[i+21]-data_list[i],data_list[i+22]-data_list[i],0,0,0,0,0,0,0,0])
            if len(pri_[i])!=30:
                print(i)
            
        elif i==234:
            pri_.append([data_list[i+1]-data_list[i],data_list[i+2]-data_list[i],data_list[i+3]-data_list[i],\
        data_list[i+4]-data_list[i],data_list[i+5]-data_list[i],data_list[i+6]-data_list[i],data_list[i+7]-data_list[i],data_list[i+8]-data_list[i],data_list[i+9]-data_list[i],data_list[i+10]-data_list[i],data_list[i+11]-data_list[i],data_list[i+12]-data_list[i],data_list[i+13]-data_list[i],data_list[i+14]-data_list[i],data_list[i+15]-data_list[i],\
            data_list[i+16]-data_list[i],data_list[i+17]-data_list[i],data_list[i+18]-data_list[i],data_list[i+19]-data_list[i],data_list[i+20]-data_list[i],data_list[i+21]-data_list[i],0,0,0,0,0,0,0,0,0])
            if len(pri_[i])!=30:
                print(i)
            
        elif i==235:
            pri_.append([data_list[i+1]-data_list[i],data_list[i+2]-data_list[i],data_list[i+3]-data_list[i],\
        data_list[i+4]-data_list[i],data_list[i+5]-data_list[i],data_list[i+6]-data_list[i],data_list[i+7]-data_list[i],data_list[i+8]-data_list[i],data_list[i+9]-data_list[i],data_list[i+10]-data_list[i],data_list[i+11]-data_list[i],data_list[i+12]-data_list[i],data_list[i+13]-data_list[i],data_list[i+14]-data_list[i],data_list[i+15]-data_list[i],\
            data_list[i+16]-data_list[i],data_list[i+17]-data_list[i],data_list[i+18]-data_list[i],data_list[i+19]-data_list[i],data_list[i+20]-data_list[i],0,0,0,0,0,0,0,0,0,0])
            if len(pri_[i])!=30:
                print(i)
            
        elif i==236:
            pri_.append([data_list[i+1]-data_list[i],data_list[i+2]-data_list[i],data_list[i+3]-data_list[i],\
        data_list[i+4]-data_list[i],data_list[i+5]-data_list[i],data_list[i+6]-data_list[i],data_list[i+7]-data_list[i],data_list[i+8]-data_list[i],data_list[i+9]-data_list[i],data_list[i+10]-data_list[i],data_list[i+11]-data_list[i],data_list[i+12]-data_list[i],data_list[i+13]-data_list[i],data_list[i+14]-data_list[i],data_list[i+15]-data_list[i],\
            data_list[i+16]-data_list[i],data_list[i+17]-data_list[i],data_list[i+18]-data_list[i],data_list[i+19]-data_list[i],0,0,0,0,0,0,0,0,0,0,0])
            if len(pri_[i])!=30:
                print(i)
            
        elif i==237:
            pri_.append([data_list[i+1]-data_list[i],data_list[i+2]-data_list[i],data_list[i+3]-data_list[i],\
        data_list[i+4]-data_list[i],data_list[i+5]-data_list[i],data_list[i+6]-data_list[i],data_list[i+7]-data_list[i],data_list[i+8]-data_list[i],data_list[i+9]-data_list[i],data_list[i+10]-data_list[i],data_list[i+11]-data_list[i],data_list[i+12]-data_list[i],data_list[i+13]-data_list[i],data_list[i+14]-data_list[i],data_list[i+15]-data_list[i],\
            data_list[i+16]-data_list[i],data_list[i+17]-data_list[i],data_list[i+18]-data_list[i],0,0,0,0,0,0,0,0,0,0,0,0])
            if len(pri_[i])!=30:
                print(i)
            
        elif i==238:
            pri_.append([data_list[i+1]-data_list[i],data_list[i+2]-data_list[i],data_list[i+3]-data_list[i],\
        data_list[i+4]-data_list[i],data_list[i+5]-data_list[i],data_list[i+6]-data_list[i],data_list[i+7]-data_list[i],data_list[i+8]-data_list[i],data_list[i+9]-data_list[i],data_list[i+10]-data_list[i],data_list[i+11]-data_list[i],data_list[i+12]-data_list[i],data_list[i+13]-data_list[i],data_list[i+14]-data_list[i],data_list[i+15]-data_list[i],\
            data_list[i+16]-data_list[i],data_list[i+17]-data_list[i],0,0,0,0,0,0,0,0,0,0,0,0,0])
            if len(pri_[i])!=30:
                print(i)
            
        elif i==239:
            pri_.append([data_list[i+1]-data_list[i],data_list[i+2]-data_list[i],data_list[i+3]-data_list[i],\
        data_list[i+4]-data_list[i],data_list[i+5]-data_list[i],data_list[i+6]-data_list[i],data_list[i+7]-data_list[i],data_list[i+8]-data_list[i],data_list[i+9]-data_list[i],data_list[i+10]-data_list[i],data_list[i+11]-data_list[i],data_list[i+12]-data_list[i],data_list[i+13]-data_list[i],data_list[i+14]-data_list[i],data_list[i+15]-data_list[i],\
            data_list[i+16]-data_list[i],0,0,0,0,0,0,0,0,0,0,0,0,0,0])
            if len(pri_[i])!=30:
                print(i)
            
        elif i==240:
            pri_.append([data_list[i+1]-data_list[i],data_list[i+2]-data_list[i],data_list[i+3]-data_list[i],\
        data_list[i+4]-data_list[i],data_list[i+5]-data_list[i],data_list[i+6]-data_list[i],data_list[i+7]-data_list[i],data_list[i+8]-data_list[i],data_list[i+9]-data_list[i],data_list[i+10]-data_list[i],data_list[i+11]-data_list[i],data_list[i+12]-data_list[i],data_list[i+13]-data_list[i],data_list[i+14]-data_list[i],data_list[i+15]-data_list[i],\
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
            if len(pri_[i])!=30:
                print(i)
            
        # 处理后四个
            
        elif i==241:
            pri_.append([data_list[i+1]-data_list[i],data_list[i+2]-data_list[i],data_list[i+3]-data_list[i],\
        data_list[i+4]-data_list[i],data_list[i+5]-data_list[i],data_list[i+6]-data_list[i],data_list[i+7]-data_list[i],data_list[i+8]-data_list[i],data_list[i+9]-data_list[i],data_list[i+10]-data_list[i],data_list[i+11]-data_list[i],data_list[i+12]-data_list[i],data_list[i+13]-data_list[i],data_list[i+14]-data_list[i],0,\
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
            if len(pri_[i])!=30:
                print(i)
            
        elif i==242:
            pri_.append([data_list[i+1]-data_list[i],data_list[i+2]-data_list[i],data_list[i+3]-data_list[i],\
        data_list[i+4]-data_list[i],data_list[i+5]-data_list[i],data_list[i+6]-data_list[i],data_list[i+7]-data_list[i],data_list[i+8]-data_list[i],data_list[i+9]-data_list[i],data_list[i+10]-data_list[i],data_list[i+11]-data_list[i],data_list[i+12]-data_list[i],data_list[i+13]-data_list[i],0,0,\
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
            if len(pri_[i])!=30:
                print(i)
            
        elif i==243:
            pri_.append([data_list[i+1]-data_list[i],data_list[i+2]-data_list[i],data_list[i+3]-data_list[i],\
        data_list[i+4]-data_list[i],data_list[i+5]-data_list[i],data_list[i+6]-data_list[i],data_list[i+7]-data_list[i],data_list[i+8]-data_list[i],data_list[i+9]-data_list[i],data_list[i+10]-data_list[i],data_list[i+11]-data_list[i],data_list[i+12]-data_list[i],0,0,0,\
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
            if len(pri_[i])!=30:
                print(i)
            
        elif i==244:
            pri_.append([data_list[i+1]-data_list[i],data_list[i+2]-data_list[i],data_list[i+3]-data_list[i],\
        data_list[i+4]-data_list[i],data_list[i+5]-data_list[i],data_list[i+6]-data_list[i],data_list[i+7]-data_list[i],data_list[i+8]-data_list[i],data_list[i+9]-data_list[i],data_list[i+10]-data_list[i],data_list[i+11]-data_list[i],0,0,0,0,\
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
            if len(pri_[i])!=30:
                print(i)
            
        elif i==245:
            pri_.append([data_list[i+1]-data_list[i],data_list[i+2]-data_list[i],data_list[i+3]-data_list[i],\
        data_list[i+4]-data_list[i],data_list[i+5]-data_list[i],data_list[i+6]-data_list[i],data_list[i+7]-data_list[i],data_list[i+8]-data_list[i],data_list[i+9]-data_list[i],data_list[i+10]-data_list[i],0,0,0,0,0,\
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
            if len(pri_[i])!=30:
                print(i)
            
        elif i==246:
            pri_.append([data_list[i+1]-data_list[i],data_list[i+2]-data_list[i],data_list[i+3]-data_list[i],\
        data_list[i+4]-data_list[i],data_list[i+5]-data_list[i],data_list[i+6]-data_list[i],data_list[i+7]-data_list[i],data_list[i+8]-data_list[i],data_list[i+9]-data_list[i],0,0,0,0,0,0,\
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
            # print(i)
            # print(len(pri_[i]))
            if len(pri_[i])!=30:
                print(i)
           
        elif i==247:
            pri_.append([data_list[i+1]-data_list[i],data_list[i+2]-data_list[i],data_list[i+3]-data_list[i],\
        data_list[i+4]-data_list[i],data_list[i+5]-data_list[i],data_list[i+6]-data_list[i],data_list[i+7]-data_list[i],data_list[i+8]-data_list[i],0,0,0,0,0,0,0,\
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
            # print(i)
            # print(len(pri_[i]))
            if len(pri_[i])!=30:
                print(i)
            
        elif i==248:
            pri_.append([data_list[i+1]-data_list[i],data_list[i+2]-data_list[i],data_list[i+3]-data_list[i],\
        data_list[i+4]-data_list[i],data_list[i+5]-data_list[i],data_list[i+6]-data_list[i],data_list[i+7]-data_list[i],0,0,0,0,0,0,0,0,\
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
            # print(i)
            # print(len(pri_[i]))
            if len(pri_[i])!=30:
                print(i)
            
        elif i==249:
            pri_.append([data_list[i+1]-data_list[i],data_list[i+2]-data_list[i],data_list[i+3]-data_list[i],\
        data_list[i+4]-data_list[i],data_list[i+5]-data_list[i],data_list[i+6]-data_list[i],0,0,0,0,0,0,0,0,0,\
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
            # print(i)
            # print(len(pri_[i]))
            if len(pri_[i])!=30:
                print(i)
            
        elif i==250:
            pri_.append([data_list[i+1]-data_list[i],data_list[i+2]-data_list[i],data_list[i+3]-data_list[i],\
        data_list[i+4]-data_list[i],data_list[i+5]-data_list[i],0,0,0,0,0,0,0,0,0,0,\
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
            # print(i)
            # print(len(pri_[i]))
            if len(pri_[i])!=30:
                print(i)
            
        elif i==251:
            pri_.append([data_list[i+1]-data_list[i],data_list[i+2]-data_list[i],data_list[i+3]-data_list[i],\
        data_list[i+4]-data_list[i],0,0,0,0,0,0,0,0,0,0,0,\
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
            # print(i)
            # print(len(pri_[i]))
            if len(pri_[i])!=30:
                print(i)
            
        elif i==252:
            pri_.append([data_list[i+1]-data_list[i],data_list[i+2]-data_list[i],data_list[i+3]-data_list[i],\
        0,0,0,0,0,0,0,0,0,0,0,0,\
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
            # print(i)
            # print(len(pri_[i]))
            if len(pri_[i])!=30:
                print(i)
            
        elif i==253:
            pri_.append([data_list[i+1]-data_list[i],data_list[i+2]-data_list[i],0,\
        0,0,0,0,0,0,0,0,0,0,0,0,\
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])            # all_pri.append(pri_)
            # print(i)
            # print(len(pri_[i]))
            if len(pri_[i])!=30:
                print(i)
            
        elif i==254:
            pri_.append([data_list[i+1]-data_list[i],0,0,\
        0,0,0,0,0,0,0,0,0,0,0,0,\
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
            # print(i)
            # print(len(pri_[i]))
            if len(pri_[i])!=30:
                print(i)
            
        elif i==255:   
            pri_.append([0,0,0,\
        0,0,0,0,0,0,0,0,0,0,0,0,\
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
            # print(i)
            # print(len(pri_[i]))
            if len(pri_[i])!=30:
                print(i)
            
    # 首先去除pri<=50和pri>=2500的值
    # print(pri_)
   
    # 此处对每一个向量进行正态分布归一化处理
    # pri = normal_process(pri_)
    # print(len(pri_))
    pri = [data/12500  for pri in pri_ for data in pri]
    all_pri.append(pri)
    all_label.append(label_list)
    # print(all_pri)
    print(file+"pri阈值处理、归一化完成")
    print(file+"生成向量完成")
    print(j)
    if num==0:
        # print("dasd")
        foldername = "./train/"
        write_vector_to_file(file,foldername,label_list,pri,1)
    elif num==1:
        foldername = "./test/"
        write_vector_to_file(file,foldername,label_list,pri,1)
    elif num==2 :
        foldername = "./val/"
        write_vector_to_file(file,foldername,label_list,pri,1)

def max_min(all_pri,all_label):
    all_pri_ = [ pri_ for pri in all_pri for pri_ in pri]
    all_label = [ l for label in all_label for l in label] 
    # print(len(all_pri_))
    all_pri_ = np.array(all_pri_)
    max_ = all_pri_.max()
    min_ = all_pri_.min() 
    all_pri_ = [ (pri-min_)/(max_-min_) for pri in all_pri_]
    print(len(all_pri_))
    return all_pri_,all_label

def normal_process(all_pri,all_label):
    # 第一步先计算一组向量的平均值，然后就算方差，在逐一归一化。
    # average_value = sum(pri_)/len(pri_)
    # 平均值
    mean_ = np.array(all_pri).mean()
    std_ = np.array(all_pri).std()
    all_pri_ = [ (pri_-mean_)/std_ for pri in all_pri for pri_ in pri]
    all_label_ = [l for label in all_label for l in label]
    return all_pri_,all_label_
def write_vector_to_file(file,filename,label_list,data_list,temp):

    if temp==1:
        # if语句中的内容只有在train中使用
        f = open(filename+file,'a')
        step = 30
        # 还原回原来的格式
        data_list = [ data_list[i:i+step]  for i in range(0,len(data_list),step) ]
        sorted_list = []
        sorted_list_= []
        sorted_list=[sorted(da) for da in data_list]
        # print()
        for s in sorted_list:
            sorted_list_.append([str(da) for da in s])
        num = 0
        data_list_ = []
        label_list_ = []
        for d in data_list:
            # print(sum(d))filename
            # if sum(d)!=0.0:
            data_list_.append(d)
            # print(num)
            label_list_.append(label_list[num])
            num+=1
        # 此时拿到了所有的合格的数据
        # print(len(label_list_))
        final_data_str=[]
        sorted_list = []
        for data in data_list_:
            final_data_str.append([str(da) for da in data])
        con = ','
        for num in range(len(final_data_str)):
            # print(label_list_[num])
            if label_list_[num] == 100:
                label_list_[num] = 0
            else:
                label_list_[num] = 1
            # print(con.join(final_data_str[num]))
            f.write(str(label_list_[num])+","+con.join(final_data_str[num])+"\n")

    
if __name__ == "__main__":

    # read_file_and_get_data('/home/wang/Radar数据/Deinterleaving/data/256_3_5_30/train45000/',0)
    # read_file_and_get_data('/home/wang/Radar数据/Deinterleaving/data/256_3_5_30/test/',1)
    read_file_and_get_data('/home/wang/Radar数据/Deinterleaving/data/256_3_5_30/val0/',2)
    # read_file_and_get_data('/home/wang/Radar数据/Deinterleaving/data/5test/',3)
    # read_file_and_get_data('/home/wang/Radar数据/Deinterleaving/data/5参差/two/',4)
    
    