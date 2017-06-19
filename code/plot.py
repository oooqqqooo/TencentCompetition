import pandas as pd #数据分析
import matplotlib.pyplot as plt


data_train = pd.read_csv("E:\\数据集\\pre\\数据预处理\\train+user+type+ad+cata.csv")
dd = pd.read_csv("E:\\数据集\\pre\\测试及结果\\userCount.csv")

data_train = pd.merge(data_train, dd, on="userID")

fig = plt.figure()
fig.set(alpha=0.2)  # 设定图表颜色alpha参数

label_0 = data_train.creativeID[data_train.label == 0].value_counts()
label_1 = data_train.creativeID[data_train.label == 1].value_counts()*5
df=pd.DataFrame({u'label0':label_0, u'label1':label_1})
df.plot(kind='bar', stacked=True)
plt.title(u"各乘客等级的获救情况")
plt.xlabel(u"creativeID")
plt.ylabel(u"number")
plt.show()

# pwd = os.getcwd()
# os.chdir(os.path.dirname(trainFile))
# data_train  = pd.read_csv(os.path.basename(trainFile))
# os.chdir(pwd)

# data_train = pd.read_csv("")
#
# fig = plt.figure()
# fig.set(alpha=0.2)  # 设定图表颜色alpha参数

# plt.subplot2grid((2,3),(0,0))             # 在一张大图里分列几个小图
# data_train['label'].value_counts().plot(kind='bar')# 柱状图
# plt.title(u"获救情况 (1为获救)") # 标题
# plt.ylabel(u"人数")
# plt.show()

#统计每个特征不同类别的0,1对比


#统计每个特征中不同类别的人数
# data_train.camgaignID[data_train.label == 1].value_counts().plot(kind='bar')
#
# plt.xlabel(u"camgaignID")# plots an axis lable
# plt.ylabel(u"number")
# plt.title(u"sadf")
# plt.legend((u'asd'),loc='best') # sets our legend for our graph.
# plt.show()