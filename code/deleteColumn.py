#f = open('F:\\数据集\\pre\\浪费\\tupac.csv')
# coding=utf-8

#删除指定列
with open('E:\\数据集\\pre\\数据预处理\\v7.csv') as reader, open('E:\\数据集\\pre\\数据预处理\\v11.csv', 'w') as writer:
    for line in reader:
        items = line.strip().split(',')
        print(','.join(items[1:]), file=writer)
        #print(','.join(items[:1]+items[6:]), file=writer)