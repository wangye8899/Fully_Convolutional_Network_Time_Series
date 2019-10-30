
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
global temp_count 
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
        if i>=60 and i<=195:
            # print(i)
            pri_.append([data_list[i]-data_list[i-60],data_list[i]-data_list[i-58],data_list[i]-data_list[i-56],data_list[i]-data_list[i-54],data_list[i]-data_list[i-52],data_list[i]-data_list[i-50],data_list[i]-data_list[i-48],data_list[i]-data_list[i-46],data_list[i]-data_list[i-44],\
                data_list[i]-data_list[i-42],data_list[i]-data_list[i-40],data_list[i]-data_list[i-38],data_list[i]-data_list[i-36],data_list[i]-data_list[i-34],data_list[i]-data_list[i-32],data_list[i]-data_list[i-30],data_list[i]-data_list[i-28],data_list[i]-data_list[i-26],data_list[i]-data_list[i-24],data_list[i]-data_list[i-22],data_list[i]-data_list[i-20],data_list[i]-data_list[i-18],data_list[i]-data_list[i-16],data_list[i]-data_list[i-14],\
                data_list[i]-data_list[i-12],data_list[i]-data_list[i-10],data_list[i]-data_list[i-8],data_list[i]-data_list[i-6],data_list[i]-data_list[i-4],data_list[i]-data_list[i-2],data_list[i+2]-data_list[i],data_list[i+4]-data_list[i],data_list[i+6]-data_list[i],\
                    data_list[i+8]-data_list[i],data_list[i+10]-data_list[i],data_list[i+12]-data_list[i],data_list[i+14]-data_list[i],data_list[i+16]-data_list[i],data_list[i+18]-data_list[i],data_list[i+20]-data_list[i],data_list[i+22]-data_list[i],data_list[i+24]-data_list[i],data_list[i+26]-data_list[i],data_list[i+28]-data_list[i],data_list[i+30]-data_list[i],data_list[i+32]-data_list[i],data_list[i+34]-data_list[i],data_list[i+36]-data_list[i],\
                    data_list[i+38]-data_list[i],data_list[i+40]-data_list[i],data_list[i+42]-data_list[i],data_list[i+44]-data_list[i],data_list[i+46]-data_list[i],data_list[i+48]-data_list[i],data_list[i+50]-data_list[i],data_list[i+52]-data_list[i],data_list[i+54]-data_list[i],data_list[i+56]-data_list[i],data_list[i+58]-data_list[i],data_list[i+60]-data_list[i]])   
            
            # pri_.append([data_list[i]-data_list[i-30],data_list[i]-data_list[i-29],data_list[i]-data_list[i-28],data_list[i]-data_list[i-27],data_list[i]-data_list[i-26],data_list[i]-data_list[i-25],data_list[i]-data_list[i-24],data_list[i]-data_list[i-23],data_list[i]-data_list[i-22],\
            #     data_list[i]-data_list[i-21],data_list[i]-data_list[i-20],data_list[i]-data_list[i-19],data_list[i]-data_list[i-18],data_list[i]-data_list[i-17],data_list[i]-data_list[i-16],data_list[i]-data_list[i-15],data_list[i]-data_list[i-14],data_list[i]-data_list[i-13],data_list[i]-data_list[i-12],data_list[i]-data_list[i-11],data_list[i]-data_list[i-10],data_list[i]-data_list[i-9],data_list[i]-data_list[i-8],data_list[i]-data_list[i-7],\
            #     data_list[i]-data_list[i-6],data_list[i]-data_list[i-5],data_list[i]-data_list[i-4],data_list[i]-data_list[i-3],data_list[i]-data_list[i-2],data_list[i]-data_list[i-1],data_list[i+1]-data_list[i],data_list[i+2]-data_list[i],data_list[i+3]-data_list[i],\
            #         data_list[i+4]-data_list[i],data_list[i+5]-data_list[i],data_list[i+6]-data_list[i],data_list[i+7]-data_list[i],data_list[i+8]-data_list[i],data_list[i+9]-data_list[i],data_list[i+10]-data_list[i],data_list[i+11]-data_list[i],data_list[i+12]-data_list[i],data_list[i+13]-data_list[i],data_list[i+14]-data_list[i],data_list[i+15]-data_list[i],data_list[i+16]-data_list[i],data_list[i+17]-data_list[i],data_list[i+18]-data_list[i],\
            #         data_list[i+19]-data_list[i],data_list[i+20]-data_list[i],data_list[i+21]-data_list[i],data_list[i+22]-data_list[i],data_list[i+23]-data_list[i],data_list[i+24]-data_list[i],data_list[i+25]-data_list[i],data_list[i+26]-data_list[i],data_list[i+27]-data_list[i],data_list[i+28]-data_list[i],data_list[i+29]-data_list[i],data_list[i+30]-data_list[i]])   
            # # pri_.append([data_list[i]-data_list[i-50],data_list[i]-data_list[i-49],data_list[i]-data_list[i-48],data_list[i]-data_list[i-47],data_list[i]-data_list[i-46],data_list[i]-data_list[i-45],data_list[i]-data_list[i-44],data_list[i]-data_list[i-43],\
            #     data_list[i]-data_list[i-42],data_list[i]-data_list[i-41],data_list[i]-data_list[i-40],data_list[i]-data_list[i-39],data_list[i]-data_list[i-38],data_list[i]-data_list[i-37],data_list[i]-data_list[i-36],data_list[i]-data_list[i-35],data_list[i]-data_list[i-34],data_list[i]-data_list[i-33],data_list[i]-data_list[i-32],data_list[i]-data_list[i-31],data_list[i]-data_list[i-30],data_list[i]-data_list[i-29],data_list[i]-data_list[i-28],data_list[i]-data_list[i-27],data_list[i]-data_list[i-26],data_list[i]-data_list[i-25],data_list[i]-data_list[i-24],data_list[i]-data_list[i-23],data_list[i]-data_list[i-22],\
            #     data_list[i]-data_list[i-21],data_list[i]-data_list[i-20],data_list[i]-data_list[i-19],data_list[i]-data_list[i-18],data_list[i]-data_list[i-17],data_list[i]-data_list[i-16],data_list[i]-data_list[i-15],data_list[i]-data_list[i-14],data_list[i]-data_list[i-13],data_list[i]-data_list[i-12],data_list[i]-data_list[i-11],data_list[i]-data_list[i-10],data_list[i]-data_list[i-9],data_list[i]-data_list[i-8],data_list[i]-data_list[i-7],\
            #     data_list[i]-data_list[i-6],data_list[i]-data_list[i-5],data_list[i]-data_list[i-4],data_list[i]-data_list[i-3],data_list[i]-data_list[i-2],data_list[i]-data_list[i-1],data_list[i+1]-data_list[i],data_list[i+2]-data_list[i],data_list[i+3]-data_list[i],\
            #         data_list[i+4]-data_list[i],data_list[i+5]-data_list[i],data_list[i+6]-data_list[i],data_list[i+7]-data_list[i],data_list[i+8]-data_list[i],data_list[i+9]-data_list[i],data_list[i+10]-data_list[i],data_list[i+11]-data_list[i],data_list[i+12]-data_list[i],data_list[i+13]-data_list[i],data_list[i+14]-data_list[i],data_list[i+15]-data_list[i],data_list[i+16]-data_list[i],data_list[i+17]-data_list[i],data_list[i+18]-data_list[i],\
            #         data_list[i+19]-data_list[i],data_list[i+20]-data_list[i],data_list[i+21]-data_list[i],data_list[i+22]-data_list[i],data_list[i+23]-data_list[i],data_list[i+24]-data_list[i],data_list[i+25]-data_list[i],data_list[i+26]-data_list[i],data_list[i+27]-data_list[i],data_list[i+28]-data_list[i],data_list[i+29]-data_list[i],data_list[i+30]-data_list[i],data_list[i+31]-data_list[i],data_list[i+32]-data_list[i],data_list[i+33]-data_list[i],data_list[i+34]-data_list[i],data_list[i+35]-data_list[i],data_list[i+36]-data_list[i],data_list[i+37]-data_list[i],data_list[i+38]-data_list[i],\
            #         data_list[i+39]-data_list[i],data_list[i+40]-data_list[i],data_list[i+41]-data_list[i],data_list[i+42]-data_list[i],data_list[i+43]-data_list[i],data_list[i+44]-data_list[i],data_list[i+45]-data_list[i],data_list[i+46]-data_list[i],data_list[i+47]-data_list[i],data_list[i+48]-data_list[i],data_list[i+49]-data_list[i],data_list[i+50]-data_list[i]])   
            # pri_.append([data_list[i]-data_list[i-5],data_list[i]-data_list[i-4],\
            # data_list[i]-data_list[i-3],data_list[i]-data_list[i-2],data_list[i]-data_list[i-1],\
            # data_list[i+1]-data_list[i],data_list[i+2]-data_list[i],data_list[i+3]-data_list[i],\
            #         data_list[i+4]-data_list[i],data_list[i+5]-data_list[i]])   
            
            # all_pri.append(pri_)label_list
            # pri_ = []label_list
            # print(i)
            # print(len(pri_[i]))

        elif i<60:
            temp = []
            for j in range(30-int(i/2)):
                temp.append(0)
            # print(len(temp))
            if i%2==0:
                # 偶数从0开始
                for h in range(0,i,2):
                    temp.append(data_list[i]-data_list[h])
            else:
                for h in range(1,i,2):
                    temp.append(data_list[i]-data_list[h])

            for k in range(i+2,i+32,2):
                temp.append(data_list[k]-data_list[i])
            pri_.append(temp)
            

        else:
            temp_255 = []
            for a in range(i-30,i,2):
                temp_255.append(data_list[i]-data_list[a])
            for b in range(i+2,256,2):
                temp_255.append(data_list[b]-data_list[i])
            for c in range(30-255+i):
                temp_255.append(0)
            pri_.append(temp_255)

    pri = [data/12500  for pri in pri_ for data in pri]

    # all_pri.append(pri)
    # all_label.append(label_list)
    # pri,_ = max_min(pri_,all_label)
    # print(a)
    # print(all_pri)
    print(file+"pri阈值处理、归一化完成")
    print(file+"生成向量完成")
    # print(temp_count)
    # temp_count+=1
    
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
        # f = open(file,'a')
        step = 60
        # 还原回原来的格式
        data_list = [ data_list[i:i+step]  for i in range(0,len(data_list),step) ]
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
        # print(num)
        for temp_num in range(len(final_data_str)):
           
            if label_list_[temp_num] == 100:
                label_list_[temp_num] = 0
            else:
                # label_list_[temp_num] = int(label_list_[temp_num])
                label_list_[temp_num] = 1
            # print(con.join(final_data_str[num]))
            f.write(str(label_list_[temp_num])+","+con.join(final_data_str[temp_num])+"\n")

    
if __name__ == "__main__":

    # read_file_and_get_data('./data/',0)
    read_file_and_get_data('/home/wang/Radar数据/Deinterleaving/data/256_3_5/train1/',0)
    # read_file_and_get_data('/home/wang/Radar数据/Deinterleaving/data/256_1_5_30/test0/',1)
    # read_file_and_get_data('/home/wang/Radar数据/Deinterleaving/data/256_1_5_1h/256_1_5参差_1_恒参_验证0/',2)
    # read_file_and_get_data('/home/wang/Radar数据/Deinterleaving/data/5test/',3)
    # read_file_and_get_data('/home/wang/Radar数据/Deinterleaving/data/5参差/two/',4)
    