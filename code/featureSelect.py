from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

from sklearn.ensemble import RandomForestRegressor

from collections import defaultdict
from sklearn.cross_validation import ShuffleSplit
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd

from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression

from sklearn.linear_model import RandomizedLasso

# load data
data_root = "."
dfTrain = pd.read_csv("%s/train.csv"%data_root)
dfTest = pd.read_csv("%s/test.csv"%data_root)
dfAd = pd.read_csv("%s/userChangeAd.csv"%data_root)
dfUser = pd.read_csv("%s/userChangeAge.csv"%data_root)
dfPosition = pd.read_csv("%s/position.csv"%data_root)

# process data
dfTrain = pd.merge(dfTrain, dfAd, on="creativeID")
dfTrain = pd.merge(dfTrain, dfUser, on="userID")
dfTrain = pd.merge(dfTrain, dfPosition, on="positionID")
dfTest = pd.merge(dfTest, dfAd, on="creativeID")
dfTest = pd.merge(dfTest, dfUser, on="userID")
dfTest = pd.merge(dfTest, dfPosition, on="positionID")
y_train = dfTrain["label"]


#enc = OneHotEncoder()
scaler = StandardScaler()
names = ["creativeID", "camgaignID",'age','appID','positionID','gender','education',
         'marriageStatus','haveBaby','hometown','countNum','actionNum','residence',
         'positionType','connectionType','telecomsOperator','appCategory', "advertiserID",
         "appPlatform"]
#X = scaler.fit_transform(dfTrain[names])   #只有L1正则化需要对数据进行归一化处理
X = dfTrain[names]  #随机森林不需要归一化
Y = dfTrain["label"]




# boston = load_boston()
# scaler = StandardScaler()
# X = scaler.fit_transform(boston["data"])
# Y = boston["target"]
# names = boston["feature_names"]

'''
1、特征选取--L1正则化

结果：Lasso model:  -0.007 * positionType + 0.007 * appCategory + -0.006 * connectionType + 0.004 * advertiserID + 0.003 * age + -0.003 * gender + -0.002 * appPlatform +
 0.002 * hometown + -0.001 * creativeID + 0.001 * education + 0.0 * haveBaby + 0.0 * telecomsOperator + 0.0 * camgaignID + -0.0 * residence + 0.0 * marriageStatus


增大alpha（步长），得到的模型就会越来越稀疏，即越来越多的特征系数会变成0。
'''
#
def lasso1(X,Y):
    lasso = Lasso(alpha=0.0001)
    lasso.fit(X, Y)

    #A helper method for pretty-printing linear models
    def pretty_print_linear(coefs, names = None, sort = False):
        if names == None:
            names = ["X%s" % x for x in range(len(coefs))]
        lst = zip(coefs, names)
        if sort:
            lst = sorted(lst,  key = lambda x:-np.abs(x[0]))
        return " + ".join("%s * %s" % (round(coef, 3), name)
                                       for coef, name in lst)

    return pretty_print_linear(lasso.coef_, names, sort = True)

print(lasso1(X,Y))

'''
2、特征选取--随机森林之重要性衡量

1、这种方法存在偏向，对具有更多类别的变量会更有利；2、对于存在关联的多个特征，其中任意一个都可以作为指示器（优秀的特征），并且一旦某个特征被选择之后，
其他特征的重要度就会急剧下降，因为不纯度已经被选中的那个特征降下来了，其他的特征就很难再降低那么多不纯度了，这样一来，只有先被选中的那个特征重要度很高，
其他的关联特征重要度往往较低。

结果：[(0.25950000000000001, 'residence'), (0.1898, 'hometown'), (0.16020000000000001, 'age'), (0.096600000000000005, 'education'),
(0.065000000000000002, 'telecomsOperator'), (0.059900000000000002, 'marriageStatus'), (0.047600000000000003, 'creativeID'),
(0.033599999999999998, 'haveBaby'), (0.026100000000000002, 'camgaignID'), (0.019900000000000001, 'gender'), (0.015900000000000001, 'appCategory'),
(0.010500000000000001, 'positionType'), (0.0083000000000000001, 'advertiserID'), (0.0048999999999999998, 'connectionType'), (0.0023, 'appPlatform')]

'''
def RFR(X,Y):
    rf = RandomForestRegressor()
    rf.fit(X, Y)
    # print ("Features sorted by their score:")
    return sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), names),
                 reverse=True)

#print(RFR(X,Y))
'''
3、特征选取--随机森林之对精度的影响
'''
# rf = RandomForestRegressor()
# scores = defaultdict(list)
#
# #crossvalidate the scores on a number of different random splits of the data
# for train_idx, test_idx in ShuffleSplit(len(X), 5, .3):
#     print(train_idx,test_idx)
#     X_train, X_test = X[train_idx], X[test_idx]
#     Y_train, Y_test = Y[train_idx], Y[test_idx]
#     r = rf.fit(X_train, Y_train)
#     acc = r2_score(Y_test, rf.predict(X_test))
#     for i in range(X.shape[1]):
#         X_t = X_test.copy()
#         np.random.shuffle(X_t[:, i])
#         shuff_acc = r2_score(Y_test, rf.predict(X_t))
#         scores[names[i]].append((acc-shuff_acc)/acc)
# print ("Features sorted by their score:")
# print (sorted([(round(np.mean(score), 4), feat) for
#               feat, score in scores.items()], reverse=True))

'''
4、特征提取--递归特征消除

找最优特征子集的贪心算法

结果:未规范化处理：[(1, 'positionType'), (2, 'connectionType'), (3, 'appPlatform'), (4, 'gender'), (5, 'telecomsOperator'), (6, 'haveBaby'),
(7, 'education'), (8, 'age'), (9, 'advertiserID'), (10, 'appCategory'), (11, 'marriageStatus'), (12, 'hometown'), (13, 'camgaignID'),
 (14, 'creativeID'), (15, 'residence')]

规范化处理过：[(1, 'positionType'), (2, 'appCategory'), (3, 'connectionType'), (4, 'advertiserID'), (5, 'age'), (6, 'gender'), (7, 'appPlatform'), (8, 'hometown'),
(9, 'creativeID'), (10, 'education'), (11, 'haveBaby'), (12, 'telecomsOperator'), (13, 'residence'), (14, 'camgaignID'), (15, 'marriageStatus')]


'''
# #use linear regression as the model
def RFE1(X,Y):
    lr = LinearRegression()
    #rank all features, i.e continue the elimination until the last one
    rfe = RFE(lr, n_features_to_select=1)
    rfe.fit(X,Y)

    #print ("Features sorted by their rank:")
    return  sorted(zip(map(lambda x: round(x, 4), rfe.ranking_), names))

#print(RFE1(X,Y))
'''
5、特征选取--稳定性选择

结果：[(1.0, 'positionType'), (1.0, 'connectionType'), (1.0, 'appCategory'), (0.73999999999999999, 'advertiserID'), (0.014999999999999999, 'creativeID'),
(0.0, 'telecomsOperator'), (0.0, 'residence'), (0.0, 'marriageStatus'), (0.0, 'hometown'), (0.0, 'haveBaby'), (0.0, 'gender'), (0.0, 'education'), (0.0, 'camgaignID'), (0.0, 'appPlatform'), (0.0, 'age')]

'''
def rlasso():
    rlasso = RandomizedLasso(alpha=0.000001)
    rlasso.fit(X, Y)

    #print ("Features sorted by their score:")
    return sorted(zip(map(lambda x: round(x, 4), rlasso.scores_),
                     names), reverse=True)

#print(rlasso(X,Y))