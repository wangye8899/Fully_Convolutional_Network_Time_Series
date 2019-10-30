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
    # step =100
    data = [ data[i:i+step] for i in range(0,len(data),step)]
    data = np.array(data)
    label = np.array(label)
    # print("信号数据如下:")
    # print(data)
    # print(label)WWW
    return data,label


def intersection(data,label):
    # 求解256个信号向量互相的交集
    intersection_list = []
    intersection_list1 = [] 
    for i in range(len(data)):
        for j in range(len(data)):
            a = np.intersect1d(data[i],data[j])
            a_length = len(a)
            if a_length ==1 :
                a_length = 0

            # if label[i]!=0:
            #     if label[j]!=0:
            #         print(str(label[i])+","+str(label[j])+)
            intersection_list.append(a_length)
            intersection_list1.append(str(label[i])+','+str(label[j])+','+str(a_length))
            
    intersection_list = np.array(intersection_list)
    intersection_list = np.reshape(intersection_list,[256,256])
    intersection_list1 = np.array(intersection_list1)
    intersection_list1 = np.reshape(intersection_list1,[256,256])
    # print("交集的矩阵形式")
    # print(intersection_list)
    # print(intersection_list1)
    return intersection_list,intersection_list1,label

def accounting_none_zero(a,b):
    temp_length = []
    fw = open('./no_zero_and_one_1024.csv','a+')
    for i in range(len(a)):
        a_index = np.nonzero(a[i])
        # print(b[i])
        # print(a[i][a_index])
        no_zero_one_lsit = [ temp  for temp in a[i][a_index] if temp!=1 and temp!=2  ]
        # print(no_zero_one_lsit)
        fw.write(str(b[i])+"    "+str(no_zero_one_lsit)+"\n")
        temp_length.append("("+str(b[i])+"  "+str(len(no_zero_one_lsit))+")")
    print(temp_length)

def write(x,y,z):
    fw = open('output.csv','a')
    fw.write("  ")
    for c in range(len(z)):
        fw.write(str(z[c])+',')
    fw.write("\n")
    for i in range(len(x)):
        fw.write(str(z[i])+",")
        for j in range(len(x[i])):
            fw.write(str(x[i][j])+",")
        fw.write("\n")

def time_range(inter):
    print(inter)
    time_range = []
    count = 0
    for i in range(len(inter)):
        count +=1
        for j in range(len(inter[i])-1,-1,-1):
            if inter[i][j]!=0 and inter[i][j]!=1 and count==1:
                time_range.append(j)
                count+=1
        count = 0     
    print(len(time_range))
    return time_range



if __name__ == "__main__":

    # data,label =  read_csv_('./256_3_5_0.8/data151.csv')
    data,label = read_csv_('data0.csv')
    intersection_list,intersection_list1,label =  intersection(data,label)
    # print(np.array(intersection_list1).shape)
    # print(len(intersection_list[0]))
    # print()
    # inter =  np.reshape(intersection_list)
    # accounting_none_zero(intersection_list,label)
    # for i in range(len(intersection_list)):
    #     print(label[i])
    #     print(intersection_list[i])
    # for j in range(len(intersection_list1)):
    #     print(intersection_list1[j])
    intersection_list1 = np.reshape(intersection_list1,[256,256])
    print(intersection_list1)
    for da in intersection_list1:
        print(da)
    # label = np.reshape(label,[16,16])
    # for i in range(len(label)):
        
    #     print("inter"+str(inter[i]))
    #     print("label"+str(label[i]))
    # print(label[75:209])
    # time = time_range(intersection_list)
    # write(intersection_list,intersection_list1,label)
    # print(intersection_list)
    # time = 1
    