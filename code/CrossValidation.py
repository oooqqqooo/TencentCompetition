'''
Created on May 26, 2017
Logistic Regression Working Module
@author: frx
'''
import  os
import numpy as np

def splitDataSet(split_size):

    fr = open('E:\\数据集\\pre\\2\\train.csv','r')#open fileName to read
    #next(fr)
    num_line = 0
    onefile = fr.readlines()
    num_line = len(onefile)
    #print(num_line)
    arr = np.arange(num_line) #get a seq and set len=numLine
    #np.random.shuffle(arr) #产生随机数列，把数据随机生成10份。。。但是这里涉及到时间，所以不用随机generate a random seq from arr
    list_all = arr.tolist()
    #print(list_all[0])
    each_size = (num_line+1) / split_size #size of each split sets
    #print(each_size)
    split_all = []; each_split = []
    count_num = 0; count_split = 0  #count_num 统计每次遍历的当前个数
                                    #count_split 统计切分次数
    each_split.append(onefile[int(list_all[0])].strip())
    print('@@@',each_split[0])
    for i in range(int(each_size),int(each_size)): #遍历整个数字序列
        #print(int(each_size*6),i)
        #print(onefile[int(list_all[i])].strip())

        each_split.append(onefile[int(list_all[i])].strip())

        #print(each_split[0:2])
        count_num += 1
        if count_num == int(each_size):
            count_split += 1
            array_ = np.array(each_split)
            # print('@@@',array_[0])
            # print('###',np.shape(array_))
            np.savetxt('E:\\数据集\\pre\\2' + "\\split_" + str(count_split) + '.csv',\
                        array_,fmt="%s", delimiter=',')  #输出每一份数据
            split_all.append(each_split) #将每一份数据加入到一个list中
            each_split = []
            count_num = 0
    return split_all

splitDataSet(80)

