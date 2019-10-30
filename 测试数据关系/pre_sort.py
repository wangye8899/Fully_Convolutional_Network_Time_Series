import numpy as np
import pandas as pd
import csv
import os


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
    return data,label

def process(data,label,file):
    data = list(data)
    label = list(label)
    file = "./train_"+"/"+str(str(file).split('.')[0])
    file = "./val_"+"/"+str(str(file).split('.')[0])
    con = ','
    for i in range(len(data)):
        fw_0 = open(file+"_0.csv",'a+')
        fw_1 = open(file+"_1.csv",'a+')
        fw_2 = open(file+"_2.csv",'a+')
        fw_3 = open(file+"_3.csv",'a+')
        if label[i] == 0:
            # 添加噪声数据
            data[i] = [str(da) for da in data[i]]
            fw_0.write(str(label[i])+","+con.join(data[i])+"\n")
        elif label[i]==1:
            # 添加类型1雷达数据
            data[i] = [str(da) for da in data[i]]
            fw_1.write(str(label[i])+","+con.join(data[i])+"\n")
        elif label[i]==2:
            # 添加类型2雷达数据
            data[i] = [str(da) for da in data[i]]
            fw_2.write(str(label[i])+","+con.join(data[i])+"\n")
        else:
            # 添加类型3雷达数据
            data[i] = [str(da) for da in data[i]]
            fw_3.write(str(label[i])+","+con.join(data[i])+"\n")
    
            




label_zero = []
label_one = []
label_two = []
label_three = []

path = './val/'
# path = './train'
file_list = os.listdir(path)
i = 0
for file in file_list:
    i+=1
    file=path+file
    data,label = read_csv_(file)
    process(data,label,file)