# -*- coding:utf-8 -*-

import csv as csv
import numpy as np

# -------------
# csv读取表格数据
# -------------
'''
csv_file_object = csv.reader(codecs.open('ReaderRentRecode.csv', 'rb'))
header = csv_file_object.next()
print header
print type(header)
print header[1]

data = []
for row in csv_file_object:
    data.append(row)
data = np.array(data)

print data[0::, 0]
'''
# -------------
# pandas读取表格数据
# -------------
import pandas as pd

df = pd.read_csv('E:\\数据集\\pre\\数据预处理\\v10.csv')

dd = pd.read_csv('E:\\数据集\\pre\\数据预处理\\v11.csv')

data = pd.merge(df, dd, on=['userID'], how='left')  # pandas csv表左连接
data = data[['userID', 'age','gender','education','marriageStatus','haveBaby','hometown','residence','countNum','actionNum']]
#data = data[['label','clickTime','conversionTime','userID', 'age','gender','education','marriageStatus','haveBaby','hometown','residence','positionID','positionType','connectionType',
             #'telecomsOperator','creativeID','adID','camgaignID','advertiserID','appID','appPlatform','appCategory']]
#data = data[['creativeID','adID','camgaignID','advertiserID','appID','appPlatform','appCategory']]
#data = data[['userID', 'age','gender','education','marriageStatus','haveBaby','hometown','residence','actionNum']]
print (data)
print ('------------------------------------------------------------------')


# -------------
# pandas写入表格数据
# -------------
data.to_csv(r'E:\\数据集\\pre\\数据预处理\\v7.csv', encoding='gbk')